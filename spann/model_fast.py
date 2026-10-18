"""
SPANN - fast version (target: whole pipeline in ~5 minutes on a Kaggle GPU).

Drop-in replacement for ``model_origin.SPANN_model``:
    from spann.model_fast import SPANN_model
    spann = SPANN_model(x_dim, z_dim, enc, dec, class_num, device)
    adata_source, adata_target, thr = spann.train(...same args...)        # + time_budget=...
    adata_source, adata_target, thr = spann.train_eval(...same args...)

What changed vs. the original (same losses / same maths, different execution):
  * no DataLoader / workers: every matrix lives on the GPU, batches are sampled on the GPU
  * DSBatchNorm without numpy/`.cpu()` round trips (the original synced GPU->CPU in every layer)
  * Encoder: one fused (mu, logvar) head, one fc for source+target; Decoder: BCE-with-logits (no sigmoid+log)
  * 3 Adam optimizers -> 1 fused Adam (mathematically identical, Adam is per-parameter)
  * unbalanced-OT Sinkhorn on GPU (float64) instead of POT/numpy on CPU (3 calls per iteration)
  * spatial adjacency loss vectorised (original: python loop over every cell, building a huge autograd graph)
  * label smoothing CE via F.cross_entropy, KL in closed form, mixup without CPU Beta sampling
  * evaluation only when needed (at the dip-test point and at the end), large batches, no python loops
  * dip-test null distribution: 300 simulations instead of 10000
  * wall-clock `time_budget`: the 3 training phases are scheduled by max(iter/maxiter, elapsed/time_budget),
    so training always finishes in time whatever the GPU is.
"""
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diptest import calculate_bimodality_coefficient, dip
except ImportError:  # imported as a plain module (sys.path.append('.../spann'))
    from diptest import calculate_bimodality_coefficient, dip

LOG_EPS = 1e-10


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------
class DomainBN(nn.Module):
    """Domain-specific BatchNorm: one BN per domain, applied to the rows of that domain.
    Rows must be ordered by domain: ``sizes`` = number of rows of domain 0, 1, ... (no numpy, no sync)."""

    def __init__(self, num_features, n_domain=2):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(n_domain)])

    def forward(self, x, sizes):
        outs = [bn(c) for bn, c in zip(self.bns, torch.split(x, sizes, dim=0))]
        return torch.cat(outs, 0)


class FastEncoder(nn.Module):
    """fc -> BN -> ReLU -> fused [mu | logvar] head.  Source and target go through the same fc in one matmul,
    BN statistics are still computed per domain batch (exactly like the original, which called the encoder
    separately on the source and on the target batch)."""

    def __init__(self, in_dim, hidden, z_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = nn.Linear(hidden, 2 * z_dim)
        self.z_dim = z_dim

    def forward(self, x_src, x_tgt=None, sample=True):
        sizes = [len(x_src)] if x_tgt is None else [len(x_src), len(x_tgt)]
        x = x_src if x_tgt is None else torch.cat([x_src, x_tgt], 0)
        h = self.fc(x)
        h = torch.cat([self.bn(c) for c in torch.split(h, sizes, 0)], 0) if len(sizes) > 1 else self.bn(h)
        h = F.relu(h)
        mu, logvar = self.head(h).chunk(2, dim=1)
        var = torch.exp(logvar.clamp(-10.0, 10.0))
        z = mu + var.sqrt() * torch.randn_like(mu) if sample else mu
        return z, mu, var


class FastDecoder(nn.Module):
    """Three one-layer decoders (as in the original). Returns *logits* (sigmoid is folded into the BCE loss).
      common decoder : fc -> domain BN (2 domains)        (source rows use BN #1, target rows BN #0)
      rna decoder    : fc -> BN
      spatial decoder: fc -> BN
    """

    def __init__(self, z_dim, n_common, n_rna, n_spa):
        super().__init__()
        self.fc_c = nn.Linear(z_dim, n_common)
        self.bn_c = DomainBN(n_common, 2)
        self.fc_s = nn.Linear(z_dim, n_rna)
        self.bn_s = nn.BatchNorm1d(n_rna)
        self.fc_t = nn.Linear(z_dim, n_spa)
        self.bn_t = nn.BatchNorm1d(n_spa)

    def forward(self, z_src, z_tgt):
        ns, nt = len(z_src), len(z_tgt)
        hc = self.fc_c(torch.cat([z_tgt, z_src], 0))  # domain 0 (target) first, domain 1 (source) second
        hc = self.bn_c(hc, [nt, ns])
        rec_c_t, rec_c_s = hc[:nt], hc[nt:]
        rec_s = self.bn_s(self.fc_s(z_src))
        rec_t = self.bn_t(self.fc_t(z_tgt))
        return rec_c_s, rec_c_t, rec_s, rec_t


class ProtoCLS(nn.Module):
    def __init__(self, in_dim, out_dim, temp=0.05):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        return self.fc(F.normalize(x)) / self.tmp

    @torch.no_grad()
    def weight_norm(self):
        w = self.fc.weight
        w.div_(w.norm(p=2, dim=1, keepdim=True))


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mlp=1024, feat_dim=16, temp=0.05):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, feat_dim)
        )
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        feat = self.projection_head(x)
        return feat, self.ProtoCLS(feat)


# --------------------------------------------------------------------------------------
# losses / OT helpers (all on GPU, no host sync unless stated)
# --------------------------------------------------------------------------------------
def kl_div(mu, var):
    """KL( N(mu, var) || N(0, 1) ), summed over latent dims, mean over batch (closed form)."""
    return (0.5 * (var + mu * mu - 1.0 - torch.log(var))).sum(1).mean()


def distance_gmm(mu_a, mu_b, var_a, var_b):
    """squared 2-Wasserstein-like distance between diagonal gaussians (same as the original)."""
    d_mu = ((mu_a.unsqueeze(1) - mu_b.unsqueeze(0)) ** 2).sum(2)
    d_sd = ((var_a.sqrt().unsqueeze(1) - var_b.sqrt().unsqueeze(0)) ** 2).sum(2)
    return d_mu + d_sd


def pdists(A, squared=False, eps=0):
    prod = A @ A.t()
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def mixup_target(x, device=None):
    """all pair-wise mixups x_ij = w_i*x_j + (1-w_i)*x_i, i != j, w~U(0,1)  (Beta(1,1) == Uniform)."""
    b = x.size(0)
    w = torch.rand(b, 1, 1, device=x.device)
    mix = w * x.unsqueeze(0) + (1 - w) * x.unsqueeze(1)  # [i, j, :]
    keep = ~torch.eye(b, dtype=torch.bool, device=x.device)
    return mix[keep]


def sinkhorn_unbalanced(a, b, M, reg=0.01, reg_m=0.5, stop_thr=1e-4, max_iter=1000, check_every=10):
    """Unbalanced Sinkhorn-Knopp, same algorithm / stopping rule as ``ot.unbalanced.sinkhorn_knopp_unbalanced``
    but on GPU. float64 is needed because K = exp(-M/reg) = exp(100*sim) overflows float32."""
    K = torch.exp(-M / reg)
    fi = reg_m / (reg_m + reg)
    u = torch.full_like(a, 1.0 / a.numel())
    v = torch.full_like(b, 1.0 / b.numel())
    for i in range(max_iter):
        u_prev, v_prev = u, v
        u = (a / (K @ v)) ** fi
        v = (b / (K.t() @ u)) ** fi
        if i % check_every == 0:  # one host sync every `check_every` iterations
            err_u = (u - u_prev).abs().max() / torch.stack([u.abs().max(), u_prev.abs().max(), torch.ones_like(u[0])]).max()
            err_v = (v - v_prev).abs().max() / torch.stack([v.abs().max(), v_prev.abs().max(), torch.ones_like(v[0])]).max()
            if bool(0.5 * (err_u + err_v) < stop_thr):
                break
    return u.unsqueeze(1) * K * v.unsqueeze(0)


def ubot_CCD(sim, beta, stop_thr=1e-4):
    """Unbalanced OT between a batch (uniform marginal) and the class prototypes (marginal beta).
    :sim: [n, K] cosine similarities, :beta: [K] float64 tensor
    :return: pseudo label, new beta (float64), normalised plan Q [n,K] (float32), confidence w_i [n]"""
    n = sim.size(0)
    alpha = torch.full((n,), 1.0 / n, dtype=torch.float64, device=sim.device)
    Q = sinkhorn_unbalanced(alpha, beta, -sim.detach().double(), stop_thr=stop_thr)
    Q = Q / Q.sum()
    wt, pseudo = Q.max(1)
    return pseudo, Q.sum(0), Q.float(), wt.float()


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def ensemble_score(logits, confidence_ot):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    softmax_cls = p.max(1)[0]
    entropy_cls = (p * logp).sum(-1) / np.log(p.size(1)) + 1
    return (entropy_cls + softmax_cls + confidence_ot) / 3, entropy_cls, softmax_cls, confidence_ot


def _to_gpu(x, device, dtype=torch.float32):
    if sp.issparse(x):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=dtype).to(device)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class SPANN_model:
    """
    Same constructor as the original (enc/dec config lists are only used to read the dimensions).
    :x_dim: dict {0: n_common_genes, 1: n_spatial_genes, 2: n_rna_genes}
    :enc_hidden: width of the encoder hidden layer (original 1024; 512 is ~2x cheaper)
    :cls_hidden: width of the classifier projection head (original 1024; 256 is plenty for a 16-d feature)
    """

    def __init__(self, x_dim, z_dim, enc, dec, class_num, device, enc_hidden=None, cls_hidden=1024):
        super().__init__()
        self.class_num = class_num
        self.device = torch.device(device)
        self.feat_dim = enc[-1][1]
        enc_hidden = enc_hidden or enc[0][1]
        self.encoder = FastEncoder(x_dim[0], enc_hidden, z_dim).to(self.device)
        self.decoder = FastDecoder(z_dim, x_dim[0], x_dim[2], x_dim[1]).to(self.device)
        self.classifier = CLS(z_dim, class_num, hidden_mlp=cls_hidden, feat_dim=self.feat_dim).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ data
    def _prepare(self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size=None):
        dev = self.device
        d = {}
        d["xs_c"] = _to_gpu(source_cm_dl.dataset.data, dev)
        d["xt_c"] = _to_gpu(target_cm_dl.dataset.data, dev)
        d["xs_s"] = _to_gpu(source_sp_ds.data, dev)
        d["xt_s"] = _to_gpu(target_sp_ds.data, dev)
        d["ys"] = torch.as_tensor(np.asarray(source_cm_dl.dataset.labels).astype(np.int64)).to(dev)
        coor = torch.as_tensor(np.array(spatial_coor), dtype=torch.float32).to(dev)
        d["coor"] = coor / coor.max(0)[0]
        # class balanced sampling of the source (== WeightedRandomSampler of the original)
        counts = torch.bincount(d["ys"], minlength=self.class_num).float().clamp(min=1)
        d["w_src"] = (1.0 / counts)[d["ys"]]
        d["bs"] = batch_size or getattr(source_cm_dl, "batch_size", None) or 256
        return d

    # ------------------------------------------------------------------ inference pass over all cells
    @torch.no_grad()
    def _infer(self, d, beta, beta_s, resolution, eval_bs=1024, with_source_ot=False, seed=0):
        dev = self.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        proto = self.classifier.ProtoCLS.fc.weight
        ns, nt = len(d["xs_c"]), len(d["xt_c"])
        out = {
            "src_feat": torch.zeros(ns, self.feat_dim, device=dev),
            "tgt_feat": torch.zeros(nt, self.feat_dim, device=dev),
            "escore": torch.zeros(nt, device=dev),
            "entropy": torch.zeros(nt, device=dev),
            "softmax": torch.zeros(nt, device=dev),
            "conf": torch.zeros(nt, device=dev),
            "pred_cls": torch.zeros(nt, dtype=torch.long, device=dev),
            "pred_ot": torch.zeros(nt, dtype=torch.long, device=dev),
            "src_pred_cls": torch.zeros(ns, dtype=torch.long, device=dev),
            "src_pred_ot": torch.zeros(ns, dtype=torch.long, device=dev),
        }
        # random order so that the (per batch) BN statistics are representative; scattered back by index
        perm_s = torch.randperm(ns, generator=g).to(dev)
        perm_t = torch.randperm(nt, generator=g).to(dev)

        for i in range(0, ns, eval_bs):
            idx = perm_s[i : i + eval_bs]
            if len(idx) < 2:
                continue
            z, _, _ = self.encoder(d["xs_c"][idx])
            feat, logit = self.classifier(z)
            out["src_feat"][idx] = feat
            out["src_pred_cls"][idx] = logit.argmax(1)
            if with_source_ot:
                sim = F.normalize(feat) @ proto.t()
                out["src_pred_ot"][idx] = ubot_CCD(sim, beta_s)[0]

        # threshold from mix-up of 256 random target cells (original: the first test batch)
        sub = perm_t[:256]
        z_sub, _, _ = self.encoder(d["xt_c"][sub])
        z_mix = mixup_target(z_sub)
        f_mix, l_mix = self.classifier(z_mix)
        _, _, _, w_mix = ubot_CCD(F.normalize(f_mix) @ proto.t(), beta_s)
        e_mix, _, _, _ = ensemble_score(l_mix, minmax(w_mix))
        thr = torch.minimum(e_mix.mean(), torch.quantile(e_mix, 1 - resolution))

        for i in range(0, nt, eval_bs):
            idx = perm_t[i : i + eval_bs]
            if len(idx) < 2:
                continue
            z, _, _ = self.encoder(d["xt_c"][idx])
            feat, logit = self.classifier(z)
            nf = F.normalize(feat)
            _, _, _, wt = ubot_CCD(nf @ proto.t(), beta)
            e, ent, smx, conf = ensemble_score(logit, minmax(wt))
            pred_cls = logit.argmax(1)
            pred_ot = pred_cls.clone()
            known = e >= thr
            kid = torch.nonzero(known).squeeze(1)
            if kid.numel() > 1:
                pred_ot[kid] = ubot_CCD(nf[kid] @ proto.t(), beta)[0]
            pred_cls[~known] = -1
            pred_ot[~known] = -1
            out["tgt_feat"][idx] = feat
            out["escore"][idx] = e
            out["entropy"][idx] = ent
            out["softmax"][idx] = smx
            out["conf"][idx] = conf
            out["pred_cls"][idx] = pred_cls
            out["pred_ot"][idx] = pred_ot
        out["thr"] = thr
        return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}

    # ------------------------------------------------------------------ core training loop
    def _fit(
        self, d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every, eval_fn=None,
    ):
        dev = self.device
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
        try:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4, fused=(dev.type == "cuda"))
        except (RuntimeError, TypeError):
            opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)

        bs, ns, nt = d["bs"], len(d["xs_c"]), len(d["xt_c"])
        K = self.class_num
        beta = torch.full((K,), 1.0 / K, dtype=torch.float64, device=dev)
        beta_s = beta.clone()
        threshold = None
        f1, f2, f3 = miditer1 / maxiter, miditer2 / maxiter, miditer3 / maxiter
        t0 = time.time()
        it, dip_done, perm, ptr = 0, False, torch.randperm(nt, device=dev), 0
        history = {}

        def progress(i):
            p = i / maxiter
            return max(p, (time.time() - t0) / time_budget) if time_budget else p

        while progress(it) < 1.0:
            p = progress(it)
            use_uot, use_spa, use_nb = p > f1, p > f3, p > f2

            # ---- batches (GPU sampling)
            si = torch.multinomial(d["w_src"], bs, replacement=True)
            if ptr + bs > nt:
                perm, ptr = torch.randperm(nt, device=dev), 0
            ti = perm[ptr : ptr + bs]
            ptr += bs
            xs_c, xt_c, xs_s, xt_s, ys, coor = d["xs_c"][si], d["xt_c"][ti], d["xs_s"][si], d["xt_s"][ti], d["ys"][si], d["coor"][ti]

            # ---- forward
            z_all, mu_all, var_all = self.encoder(xs_c, xt_c)
            z_s, z_t = z_all[:bs], z_all[bs:]
            mu_s, mu_t, var_s, var_t = mu_all[:bs], mu_all[bs:], var_all[:bs], var_all[bs:]
            rc_s, rc_t, r_s, r_t = self.decoder(z_s, z_t)
            feat_all, logit_all = self.classifier(z_all)
            feat_t, logit_t = feat_all[bs:], logit_all[bs:]
            nfeat_t = F.normalize(feat_t)

            bce = F.binary_cross_entropy_with_logits
            recon = bce(rc_s, xs_c) + bce(rc_t, xt_c) + 0.5 * (bce(r_s, xs_s) + bce(r_t, xt_s))
            kl = kl_div(mu_s, var_s) + kl_div(mu_t, var_t)
            cls_loss = F.cross_entropy(logit_all[:bs], ys, label_smoothing=0.1)

            zero = torch.zeros((), device=dev)
            spa_loss = ccd_loss = neighbor_loss = zero

            if use_uot:
                proto = self.classifier.ProtoCLS.fc.weight
                with torch.no_grad():
                    _, _, _, wti_all = ubot_CCD(nfeat_t @ proto.t(), beta)
                    esc_t, _, _, _ = ensemble_score(logit_t, minmax(wti_all))
                    if threshold is None or it % thr_freq == 0:
                        sub = torch.randperm(bs, device=dev)[:100]
                        z_mix = mixup_target(z_t[sub])
                        f_mix, l_mix = self.classifier(z_mix)
                        _, _, _, w_mix = ubot_CCD(F.normalize(f_mix) @ proto.t(), beta_s)
                        e_mix, _, _, _ = ensemble_score(l_mix, minmax(w_mix))
                        threshold = torch.minimum(e_mix.mean(), torch.quantile(e_mix, 1 - resolution))
                    hc = torch.nonzero(esc_t > threshold).squeeze(1)  # the only host sync of the step
                n_hc = hc.numel()

                if n_hc >= 2:
                    coor_hc = coor[hc]
                    nfeat_hc = nfeat_t[hc]
                    logit_hc = logit_t[hc]
                    with torch.no_grad():
                        _, new_beta, Q_t, wt_i = ubot_CCD(nfeat_hc @ proto.t(), beta)
                        weight_t = minmax(wt_i)
                        soft_q = Q_t / Q_t.sum(1, keepdim=True)
                        ok = (~(torch.isnan(soft_q).any() | torch.isnan(weight_t).any())).float()
                        soft_q = torch.nan_to_num(soft_q)
                        weight_t = torch.nan_to_num(weight_t)
                        beta = torch.where(torch.isnan(new_beta).any(), beta, mu * beta + (1 - mu) * new_beta)

                    if use_spa:
                        kk = min(k, n_hc)
                        sd = distance_gmm(mu_t[hc], mu_t[hc], var_t[hc], var_t[hc])
                        sd = sd / sd.max()
                        cd = pdists(coor_hc, squared=True)
                        cd = cd / cd.max()
                        index = torch.topk(cd, k=kk, largest=False)[1]
                        diff = sd.gather(1, index) - cd.gather(1, index)
                        spa_loss = diff.norm(dim=1).sum() / max(kk - 1, 1) / n_hc

                    pred_soft = torch.softmax(logit_hc, 1)
                    ccd_loss = ok * torch.mean(weight_t * (-soft_q * torch.log(pred_soft + LOG_EPS)).sum(1))

                    if use_nb:
                        lsm = F.log_softmax(logit_hc, 1)
                        dist = pdists(coor_hc)
                        dist = dist + dist.max() * torch.eye(n_hc, device=dev)
                        sp_idx = dist.argmin(1)
                        spatial_loss = (
                            -(Q_t[sp_idx] * lsm).sum() - (Q_t * F.log_softmax(logit_hc[sp_idx], 1)).sum()
                        ) / (2 * n_hc)
                        feat_mat = nfeat_hc @ nfeat_hc.t() / temp
                        feat_mat = feat_mat.masked_fill(torch.eye(n_hc, dtype=torch.bool, device=dev), -1 / temp)
                        lc_idx = feat_mat.argmax(1)
                        local_loss = (
                            -(Q_t[lc_idx] * lsm).sum() - (Q_t * F.log_softmax(logit_hc[lc_idx], 1)).sum()
                        ) / (2 * n_hc)
                        neighbor_loss = (spatial_loss + local_loss) / 2

            loss = (
                lambda_recon * recon + lambda_kl * kl + cls_loss
                + lambda_cd * ccd_loss + lambda_nb * neighbor_loss + lambda_spa * spa_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.classifier.ProtoCLS.weight_norm()
            it += 1

            if log_every and it % log_every == 0:
                print(
                    "#Iter %d (%.0fs): Recon %.4f | KL %.3f | CLS %.4f | Spatial %.4f | CCD %.4f | Neighbor %.4f"
                    % (it, time.time() - t0, recon.item(), kl.item(), cls_loss.item(),
                       float(spa_loss), float(ccd_loss), float(neighbor_loss))
                )

            # dip test (decides `resolution`) once, right when the UOT phase is about to start
            if not dip_done and novel_cell_test and progress(it) >= f1:
                dip_done = True
                res = self._infer(d, beta, beta_s, resolution)
                dp = dip(res["escore"], num_bins=100)
                dip_p, dip_ok = dp[-1], dp[-2]
                bc = calculate_bimodality_coefficient(np.sqrt(res["escore"]))
                bimodal = (not dip_ok) or (bc > 0.555)
                resolution = 0.5 if bimodal else 0.8
                print("[dip test] p=%.4f  BC=%.3f (>0.555 bimodal)  ood exists: %s  -> resolution=%.1f"
                      % (dip_p, bc, bimodal, resolution))
                if eval_fn is not None:
                    eval_fn(it, res)
                threshold = None

        print("training finished: %d iterations in %.1fs" % (it, time.time() - t0))
        res = self._infer(d, beta, beta_s, resolution, with_source_ot=eval_fn is not None)
        if eval_fn is not None:
            eval_fn(it, res)
        return res

    # ------------------------------------------------------------------ public API
    def train(
        self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
        test_source_cm_dl, test_target_cm_dl, source_labels, cell_types,
        lr=2e-4, lambda_recon=2000, lambda_kl=0.5, lambda_spa=0.1, lambda_cd=0.001, lambda_nb=0.1,
        mu=0.6, temp=0.1, k=20, resolution=0.5, novel_cell_test=True,
        maxiter=6000, miditer1=2000, miditer2=5000, miditer3=4000, test_freq=1000,
        time_budget=None, batch_size=None, thr_freq=10, log_every=100,
    ):
        """Annotate spatial data without ground truth. Arguments as in the original, plus
        :time_budget: seconds of wall-clock for the whole training loop (phases are rescaled to fit)
        :batch_size: override the dataloader batch size, :thr_freq: threshold refresh period
        :return: adata_source, adata_target, threshold_test"""
        d = self._prepare(source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size)
        res = self._fit(d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
                        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every)
        source_labels = np.asarray(source_labels).astype(int)
        adata_source = ad.AnnData(sp.csr_matrix(res["src_feat"]))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels]
        adata_source.obs["source"] = "scRNA"
        adata_target = ad.AnnData(sp.csr_matrix(res["tgt_feat"]))
        total = list(cell_types) + ["Unknown"]
        adata_target.obs["pred_cls"] = [total[i] for i in res["pred_cls"]]
        adata_target.obs["pred_ot"] = [total[i] for i in res["pred_ot"]]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = res["escore"]
        adata_target.obs["source"] = "Spatial"
        adata_target.obs.index = adata_target.obs.index.astype(str)
        return adata_source, adata_target, float(res["thr"])

    def train_eval(
        self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
        test_source_cm_dl, test_target_cm_dl, source_labels, target_labels, cell_types, common_cell_type,
        lr=2e-4, lambda_recon=2000, lambda_kl=0.5, lambda_spa=0.1, lambda_cd=0.001, lambda_nb=0.1,
        mu=0.6, temp=0.1, k=20, resolution=0.5, novel_cell_test=True,
        maxiter=6000, miditer1=2000, miditer2=5000, miditer3=4000, test_freq=1000,
        time_budget=None, batch_size=None, thr_freq=10, log_every=100,
    ):
        """Same as ``train`` but prints accuracies against the spatial ground truth `target_labels`."""
        d = self._prepare(source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size)
        source_labels = np.asarray(source_labels).astype(int)
        target_labels = np.asarray(target_labels).astype(int).copy()
        known_mask = np.isin(target_labels, np.unique(source_labels))
        target_eval = target_labels.copy()
        target_eval[~known_mask] = -1

        def report(it, r):
            s_cls = (r["src_pred_cls"] == source_labels).mean()
            s_ot = (r["src_pred_ot"] == source_labels).mean()
            for name in ("ot", "cls"):
                pred = r["pred_" + name]
                tot = (pred == target_eval).mean()
                kn = (pred[known_mask] == target_eval[known_mask]).mean()
                un = (pred[~known_mask] == -1).mean() if (~known_mask).any() else float("nan")
                print("#Iter %d: (%s) Source acc: %f, Target total acc: %f, known acc: %f, unknown acc: %f"
                      % (it, name.upper(), s_ot if name == "ot" else s_cls, tot, kn, un))

        res = self._fit(d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
                        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every,
                        eval_fn=report)
        adata_source = ad.AnnData(sp.csr_matrix(res["src_feat"]))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels]
        adata_source.obs["source"] = "scRNA"
        adata_target = ad.AnnData(sp.csr_matrix(res["tgt_feat"]))
        total = list(common_cell_type) + ["Unknown"]
        adata_target.obs["cell_type"] = [cell_types[i] for i in target_labels]
        adata_target.obs["pred_cls"] = [total[i] for i in res["pred_cls"]]
        adata_target.obs["pred_ot"] = [total[i] for i in res["pred_ot"]]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = res["escore"]
        adata_target.obs["source"] = "Spatial"
        adata_target.obs.index = adata_target.obs.index.astype(str)
        return adata_source, adata_target, float(res["thr"])
