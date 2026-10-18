# SPANN (spatial-aware version) - optimised for fast runs (e.g. Kaggle T4, ~5 minutes end-to-end)
#
# Main changes w.r.t. the original file (algorithm / losses / hyper-parameters are unchanged):
#   1. No torch DataLoader in the training loop: all matrices are densified ONCE and kept on the GPU,
#      mini-batches are gathered by index (the old code did per-cell sparse->dense in 4 worker processes).
#   2. `time_budget` (seconds): training phases (UOT / neighbour / spatial loss) are scheduled on
#      progress = max(iter / maxiter, elapsed / time_budget), so every phase is always reached
#      and the loop stops on time.
#   3. Periodic evaluation is skipped (in `train` it had no side effect); evaluation only runs at the
#      bimodality (dip) test point and once at the end.
#   4. dip test null distribution: 500 Monte-Carlo samples instead of 10000 (~35 s -> ~2 s).
#   5. Unbalanced Sinkhorn is implemented in numpy (identical to POT's sinkhorn_knopp_unbalanced) -> no POT dependency.
#   6. No GPU<->CPU syncs / numpy round trips in DSBatchNorm, label smoothing, mixup, threshold computation.
#   7. One fused Adam instead of three Adam optimisers, closed-form KL, no needless autograd graphs.
#   8. `easydl` (unused) import removed.
import time

import numpy as np
import scanpy as sc
import scipy
import scipy.sparse as sp
import scipy.stats
import torch
import torch.nn.functional as F
from torch import nn as nn

# ---- compatibility: anndata >= 0.11 removed AnnData.concatenate, which spann/preprocess.py still calls.
import anndata as _ad
import pandas as pd

if not hasattr(_ad.AnnData, "concatenate"):

    def _concatenate(self, *others, join="inner", batch_key="batch", batch_categories=None, index_unique="-", **kw):
        adatas = [self, *others]
        keys = [str(c) for c in (batch_categories if batch_categories is not None else range(len(adatas)))]
        res = _ad.concat(adatas, join=join, label=batch_key, keys=keys, index_unique=index_unique)
        # the old method kept the union of all obs columns (ad.concat keeps only shared ones)
        obs = pd.concat([a.obs for a in adatas], join="outer")
        obs.index = res.obs_names
        obs[batch_key] = res.obs[batch_key].values
        res.obs = obs
        return res

    _ad.AnnData.concatenate = _concatenate


def kl_div(mu, var, weight=None):
    # KL( N(mu, var) || N(0, 1) ), closed form, averaged over the batch
    loss = 0.5 * (var + mu.pow(2) - 1.0 - torch.log(var)).sum(dim=1)
    return loss.mean()


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    # Returns the matrix of ||x_i-y_j||_p^p.
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, var_src: torch.Tensor, var_dst: torch.Tensor):
    # Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    std_src = var_src.sqrt()
    std_dst = var_dst.sqrt()
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)
    return distance_mean + distance_var


def calculate_bimodality_coefficient(series=None):
    """
    calculate BC-coefficient, higher BC coef means more likely to be a bimodal distribution
    """
    skewness = scipy.stats.skew(series, axis=None, bias=False)
    kurtosis = scipy.stats.kurtosis(series, axis=None, fisher=True, bias=False)
    count = len(series)
    count_factor = ((count - 1) ** 2) / ((count - 2) * (count - 3))
    coefficient = ((skewness**2) + 1) / (kurtosis + (3 * count_factor))
    return coefficient


def dip(samples, num_bins=100, p=0.99, table=True, n_samples=10000):
    """
    dip test for distribution, to test is the distribution id multi-modality
    :n_samples: number of Monte-Carlo samples used to build the null distribution (10000 in the original code,
                500 is enough for the decision threshold and ~20x faster)
    """
    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 5e-2

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[: xl + 1] - cdf[: xl + 1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:] - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl : xr + 1]
        idxs = idxs[xl : xr + 1]
        pdf = pdf[xl : xr + 1]

    if table:
        p_threshold, p_value = p_table(p, ans, samples.size, n_samples)
        if ans < p_threshold:
            check = True
        return ans, p_threshold, check, p_value

    return ans


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1 - cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs(alpha[contact_points] - beta[contact_points])
    return diff.max(), diff


def p_table(p, ans, sample_size, n_samples):
    data = [np.random.randn(sample_size) for _ in range(n_samples)]
    dip_sample = [dip(samples, table=False) for samples in data]
    dips = np.hstack(dip_sample)
    dip_sample.append(ans)
    index = np.argsort(dip_sample)
    p_value = 1 - (np.argsort(index)[-1] + 1) / len(index)
    return np.percentile(dips, p * 100), p_value


### Networks
activation = {
    "relu": nn.ReLU(),
    "rrelu": nn.RReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "": None,
}


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization Layer
    :num_features: dimension of the features
    :n_domain: domain number

    forward(x, y): y can be a python int (whole batch belongs to one domain -> fast path, no GPU sync)
    or a tensor of per-sample domain ids (slow generic path).
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        if isinstance(y, int):  # fast path
            if x.size(0) > 1:
                return self.bns[y](x)
            bn = self.bns[y]
            was_training = bn.training
            bn.training = False
            out = bn(x)
            bn.training = was_training
            return out

        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]
            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                self.bns[i].training = False
                out[indices] = self.bns[i](x[indices])
                self.bns[i].training = True
        return out


class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    :input_dim: dimension of input
    : output_dim: dimension of output
    :norm: batch normalization, '' represent no batch normalization, 1 represent regular batch normalization, int>1 represent domain-specific batch normalization of n domain
    :act: activation function, relu -> nn.ReLU, rrelu -> nn.RReLU, sigmoid -> nn.Sigmoid(), leaky_relu -> nn.LeakyReLU(), tanh -> nn.Tanh(),  '' -> None
    dropout: dropout rate
    """

    def __init__(self, input_dim, output_dim, norm="", act="", dropout=0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if type(norm) == int:
            if norm == 1:  # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.act = activation[act]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y=None):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == "DSBatchNorm":
                h = self.norm(h, y)
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    :input_dim: input dimension
    :cfg: model structure configuration, example - [['fc', x_dim, n_domain, 'sigmoid']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i == 0:
                d_in = input_dim
            if layer[0] == "fc":
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)

    def forward(self, x, y=None):
        for layer in self.net:
            x = layer(x, y)
        return x


class Encoder(nn.Module):
    """
    VAE Encoder
    :input_dim: input dimension
    :cfg: encoder configuration, example - [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()

        enc = []
        mu_enc = []
        var_enc = []

        h_dim = cfg[-2][1]

        enc.append(NN(input_dim[0], cfg[:-1]))
        mu_enc.append(NN(h_dim, cfg[-1:]))
        var_enc.append(NN(h_dim, cfg[-1:]))

        self.enc = nn.ModuleList(enc)
        self.mu_enc = nn.ModuleList(mu_enc)
        self.var_enc = nn.ModuleList(var_enc)

    def reparameterize(self, mu, var):
        return mu + var.sqrt() * torch.randn_like(mu)

    def forward(self, x, domain, y=None):
        """ """
        q = self.enc[domain](x, y)
        mu = self.mu_enc[domain](q, y)
        var = torch.exp(self.var_enc[domain](q, y))
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    """
    VAE Decoder
    :z_dim: latent dimension
    :cfg: decoder configuration, example - [['fc', adatas[i].obsm[obsm[i]].shape[1], 1, 'sigmoid']]

    """

    def __init__(self, z_dim, cfg):
        super().__init__()

        dec = []
        for i in cfg.keys():
            dec.append(NN(z_dim, cfg[i]))

        self.dec = nn.ModuleList(dec)

    def forward(self, z, domain, y=None):
        """ """
        reconx_x = self.dec[domain](z, y)

        return reconx_x


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """

    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    @torch.no_grad()
    def weight_norm(self):
        w = self.fc.weight
        w.div_(w.norm(p=2, dim=1, keepdim=True))  # in-place: keeps the same Parameter object (fused Adam friendly)


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """

    def __init__(self, in_dim, out_dim, hidden_mlp=1024, feat_dim=16, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, feat_dim)
        )
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer (fully on-device, no CPU round trip)
    :num_classes (int): number of classes
    :epsilon (float): weight

    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        :inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        :targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
        onehot = (1 - self.epsilon) * onehot + self.epsilon / self.num_classes
        loss = (-onehot * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        return loss


# ------------------------------------------------------------------------------------------------
# optimal transport alignment
# ------------------------------------------------------------------------------------------------
# "kl"      : K = exp(-M/reg) * (a b^T)    (POT >= 0.9.5 default)
# "entropy" : K = exp(-M/reg)              (POT < 0.9.5 default)
UOT_REG_TYPE = "kl"


def _sinkhorn_unbalanced(a, b, M, reg=0.01, reg_m=0.5, stop_thr=1e-4, max_iter=1000, reg_type=None):
    """numpy port of ot.unbalanced.sinkhorn_knopp_unbalanced (same result, no POT dependency / overhead)"""
    reg_type = reg_type or UOT_REG_TYPE
    dim_a, dim_b = M.shape
    fi = reg_m / (reg_m + reg)
    K = np.exp(-M / reg)
    if reg_type == "kl":
        K = K * (a[:, None] * b[None, :])
    u = np.ones(dim_a)
    v = np.ones(dim_b)
    for _ in range(max_iter):
        uprev, vprev = u, v
        u = (a / (K @ v)) ** fi
        Ktu = K.T @ u
        v = (b / Ktu) ** fi
        if (Ktu == 0.0).any() or not np.isfinite(u).all() or not np.isfinite(v).all():
            u, v = uprev, vprev  # machine precision reached -> go back to previous solution
            break
        err_u = np.abs(u - uprev).max() / max(np.abs(u).max(), np.abs(uprev).max(), 1.0)
        err_v = np.abs(v - vprev).max() / max(np.abs(v).max(), np.abs(vprev).max(), 1.0)
        if 0.5 * (err_u + err_v) < stop_thr:
            break
    return u[:, None] * K * v[None, :]


def ubot_CCD(sim, beta, device, stopThr=1e-4):
    """
    Unbalanced optimal transport function
    :sim: similarity matrix between two distributions
    :beta: marginal distribution of the transport object
    :return: pseudo label computed from the tranport plan, updated marginal distribution of the transport object, normarlized transport plan, confidence score of the transport plan

    """
    n = sim.size(0)
    M = -sim.detach().float().cpu().numpy().astype(np.float64)  # the only GPU->CPU sync of this function
    a = np.full(n, 1.0 / n)
    b = np.asarray(beta, dtype=np.float64)
    Q_st = _sinkhorn_unbalanced(a, b, M, reg=0.01, reg_m=0.5, stop_thr=stopThr)

    # make sum equals to 1
    Q_bar = Q_st / Q_st.sum()
    new_beta = Q_bar.sum(0)  # computed on CPU, no extra sync
    Q_st_bar = torch.from_numpy(Q_bar).to(device=device, dtype=torch.float32)
    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_st_bar, 1)

    return pseudo_label, new_beta, Q_st_bar, wt_i


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def ensemble_score(logit_tensor, confidence_ot):
    """ """
    log_p = F.log_softmax(logit_tensor, dim=-1)
    label_pred_tensor = log_p.exp()
    softmax_cls, _ = torch.max(label_pred_tensor, 1)
    entropy_cls = (label_pred_tensor * log_p).sum(-1)
    entropy_cls = entropy_cls / np.log(label_pred_tensor.size()[1]) + 1

    ensemble = (entropy_cls + softmax_cls + confidence_ot) / 3
    return ensemble, entropy_cls, softmax_cls, confidence_ot


def mixup_target(x_batch, device=None):  ##batchsize*latent
    """all pairwise mixes (i != j) with weight ~ U(0,1) (= Beta(1,1)), generated on-device"""
    B, D = x_batch.shape
    w = torch.rand(B, 1, 1, device=x_batch.device)
    mix = w * x_batch.unsqueeze(0) + (1 - w) * x_batch.unsqueeze(1)  # [i, j] = w_i x_j + (1 - w_i) x_i
    keep = ~torch.eye(B, dtype=torch.bool, device=x_batch.device).reshape(-1)
    return mix.reshape(-1, D)[keep]  # B*(B-1) x D


def pdists(A, squared=False, eps=0):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)

    if squared:
        return res
    else:
        res = res.clamp(min=eps).sqrt()
        return res


# ------------------------------------------------------------------------------------------------
# data helpers (everything stays on the GPU during training)
# ------------------------------------------------------------------------------------------------
def _dense_f32(obj):
    """dataset / sparse / ndarray / tensor -> dense float32 CPU tensor"""
    if sp.issparse(obj) or isinstance(obj, (np.ndarray, torch.Tensor)):
        data = obj
    elif hasattr(obj, "data"):  # SingleCellDataset
        data = obj.data
    else:  # generic dataset: indexing with an array returns (x, label, idx)
        data = obj[np.arange(len(obj))][0]
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    elif sp.issparse(data):
        arr = data.toarray()
    else:
        arr = np.asarray(data)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class _Data:
    """dense matrices + label vectors + index samplers, resident on the training device"""

    MAX_GPU_BYTES = 8e9

    def __init__(self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, source_labels, device, bs):
        self.device = device
        self.bs = bs
        mats = [
            _dense_f32(getattr(source_cm_dl, "dataset", source_cm_dl)),
            _dense_f32(getattr(target_cm_dl, "dataset", target_cm_dl)),
            _dense_f32(source_sp_ds),
            _dense_f32(target_sp_ds),
        ]
        nbytes = sum(m.numel() * 4 for m in mats)
        if device.type == "cuda" and nbytes > self.MAX_GPU_BYTES:
            print("[SPANN] data too large for the GPU, keeping it in host memory")
        else:
            mats = [m.to(device) for m in mats]
        self.src_cm, self.tgt_cm, self.src_sp, self.tgt_sp = mats
        self.n_src = self.src_cm.size(0)
        self.n_tgt = self.tgt_cm.size(0)

        self.src_lab = torch.as_tensor(np.asarray(source_labels).astype(np.int64), device=device)
        # class-balanced sampling of scRNA cells (same as the WeightedRandomSampler of preprocess.py)
        _, inv, counts = torch.unique(self.src_lab, return_inverse=True, return_counts=True)
        self.src_w = (1.0 / counts.float())[inv]
        self.bs_t = min(bs, self.n_tgt)
        self._perm = None
        self._ptr = 0

    def take(self, t, idx):
        if t.device.type == self.device.type:
            return t[idx]
        return t[idx.to(t.device)].to(self.device)

    def sample_source(self):
        return torch.multinomial(self.src_w, self.bs, replacement=True)

    def sample_target(self):
        # shuffled epochs without replacement, drop_last=True (as the DataLoader did)
        if self._perm is None or self._ptr + self.bs_t > self.n_tgt:
            self._perm = torch.randperm(self.n_tgt, device=self.device)
            self._ptr = 0
        idx = self._perm[self._ptr : self._ptr + self.bs_t]
        self._ptr += self.bs_t
        return idx


def _f(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)


class SPANN_model:
    """
    SPANN model, using scRNA reference dataset to annotate spatial transcriptome dataset

    :x_dim: input dimension for the encoder
    :z_dim: latent feature dimension
    :enc: encoder parameters, for example, [['fc', 1024, 1, 'relu'], ['fc', 16, '', '']]
    :dec: decoder parameters, dec is a set {cm_dec_params, spa_dec_params, rna_dec_params}, each element shapes like enc
    :class_num: number of scRNA-seq cell types
    :device: device on which SPAMM model is aranged

    """

    def __init__(self, x_dim, z_dim, enc, dec, class_num, device, tau=0.5):
        super().__init__()
        self.class_num = class_num
        self.device = device
        self.feat_dim = enc[-1][1]
        # tau: temperature of the Spatial Contrastive Loss (InfoNCE) used by train_eval
        self.tau = tau
        self.encoder = Encoder(x_dim, enc).to(device)
        self.decoder = Decoder(z_dim, dec).to(device)
        self.classifier = CLS(z_dim, class_num).to(device)

    # ----------------------------------------------------------------------------------------------
    # losses
    # ----------------------------------------------------------------------------------------------
    def _spa_loss_margin(self, norm_hc, coor_hc, k, sigma, margin, alpha):
        """spatial contrastive-margin loss (used by `train`)"""
        n = norm_hc.size(0)
        feat_sim = torch.matmul(norm_hc, norm_hc.t())  # cosine similarity in [-1, 1]
        coor_dist = pdists(coor_hc, squared=True)
        coor_rbf = torch.exp(-coor_dist / (2 * (sigma**2)))  # soft spatial adjacency
        knn_index = torch.topk(coor_dist, k=min(k, n), largest=False)[1]
        mask = torch.zeros_like(coor_dist).scatter_(1, knn_index, 1.0)
        loss_pos = coor_rbf * (1.0 - feat_sim) ** 2
        loss_neg = (1.0 - coor_rbf) * torch.clamp(feat_sim - margin, min=0.0) ** 2
        total = loss_pos + alpha * loss_neg
        return torch.sum(mask * total) / (mask.sum() + 1e-10)

    def _spa_loss_infonce(self, mu_hc, coor_hc, k):
        """spatial InfoNCE loss (used by `train_eval`)"""
        n = mu_hc.size(0)
        z_sp = F.normalize(mu_hc, dim=1)
        eye = torch.eye(n, dtype=torch.bool, device=self.device)
        coor_dist = pdists(coor_hc, squared=True)
        coor_fill = coor_dist + torch.max(coor_dist) * eye.float()
        pos_idx = torch.topk(coor_fill, k=min(k, n - 1), largest=False)[1]
        pos_mask = torch.zeros(n, n, dtype=torch.bool, device=self.device).scatter_(1, pos_idx, True)
        sim = torch.matmul(z_sp, z_sp.t()) / self.tau
        sim = sim.masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        # torch.where (not "* mask"): -inf * 0 = NaN in the original implementation
        log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        num_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        return (-log_prob.sum(dim=1) / num_pos).mean()

    def _neighbor_loss(self, coor_hc, feat_hc, norm_hc, logit_hc, Q_t, temp):
        n = coor_hc.size(0)
        eye = torch.eye(n, device=self.device)
        dist_mat = pdists(coor_hc)
        dist_mat = dist_mat + torch.max(dist_mat) * eye
        spatial_nb_idx = torch.argmin(dist_mat, 1)

        # adaptive-weight spatial neighbour loss (anti over-smoothing)
        z_hc = F.normalize(feat_hc, dim=1)
        alpha_ij = torch.clamp((z_hc * z_hc[spatial_nb_idx]).sum(dim=1), min=0.0).detach()
        log_p = F.log_softmax(logit_hc, dim=1)
        ce_anchor = -(Q_t[spatial_nb_idx] * log_p).sum(dim=1)
        ce_neighbor = -(Q_t * log_p[spatial_nb_idx]).sum(dim=1)
        spatial_loss = (alpha_ij * (ce_anchor + ce_neighbor)).sum() / (2 * n)

        # feature-space nearest neighbour loss
        feat_mat = torch.matmul(norm_hc, norm_hc.t()) / temp
        feat_mat = feat_mat.masked_fill(eye.bool(), -1 / temp)
        local_nb_idx = torch.argmax(feat_mat, 1)
        local_loss = -torch.sum(Q_t[local_nb_idx] * log_p) - torch.sum(Q_t * log_p[local_nb_idx])
        local_loss = local_loss / (2 * n)

        return (spatial_loss + local_loss) / 2

    # ----------------------------------------------------------------------------------------------
    # evaluation pass (fast: one big loop, banks live on the GPU, one transfer at the end)
    # ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, d, beta, beta_s, resolution, want_source=True, with_src_acc=False):
        dev = self.device
        proto = self.classifier.ProtoCLS.fc.weight
        bs = d.bs
        res = {}

        if want_source:
            src_z = torch.zeros(d.n_src, self.feat_dim, device=dev)
            ok_ot = ok_cls = 0
            perm = torch.randperm(d.n_src, device=dev)
            for s in range(0, d.n_src, bs):
                idx = perm[s : s + bs]
                z, _, _ = self.encoder(d.take(d.src_cm, idx), 0, 1)
                feat, logit = self.classifier(z)
                src_z[idx] = feat
                if with_src_acc:
                    sim = torch.matmul(F.normalize(feat), proto.t())
                    lab_ot, _, _, _ = ubot_CCD(sim, beta_s, dev)
                    ok_ot += int((lab_ot == d.src_lab[idx]).sum())
                    ok_cls += int((logit.argmax(1) == d.src_lab[idx]).sum())
            res["source_z"] = src_z.cpu().numpy()
            if with_src_acc:
                res["acc_source_ot"] = ok_ot / d.n_src
                res["acc_source_cls"] = ok_cls / d.n_src

        n = d.n_tgt
        tgt_z = torch.zeros(n, self.feat_dim, device=dev)
        E = torch.zeros(n, device=dev)
        pred_cls = torch.zeros(n, dtype=torch.long, device=dev)
        pred_ot = torch.zeros(n, dtype=torch.long, device=dev)
        thr_test = None
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            z, _, _ = self.encoder(d.take(d.tgt_cm, idx), 0, 0)
            feat, logit = self.classifier(z)
            tgt_z[idx] = feat
            norm_feat = F.normalize(feat)

            sim_all = torch.matmul(norm_feat, proto.t())
            _, _, _, wt_i = ubot_CCD(sim_all, beta, dev)
            Escore_t, _, _, _ = ensemble_score(logit, _minmax(wt_i))

            if thr_test is None:  # threshold from mixup of the first (random) batch
                f_mix, logit_mix = self.classifier(mixup_target(z))
                sim_mix = torch.matmul(F.normalize(f_mix), proto.t())
                _, _, _, wt_mix = ubot_CCD(sim_mix, beta_s, dev)
                E_mix, _, _, _ = ensemble_score(logit_mix, _minmax(wt_mix))
                thr_test = torch.minimum(E_mix.mean(), torch.quantile(E_mix, 1 - resolution))

            known = Escore_t >= thr_test
            lab_cls = torch.where(known, logit.argmax(1), torch.full_like(idx, -1))
            lab_ot = torch.full_like(idx, -1)
            kid = torch.nonzero(known).view(-1)
            if kid.numel() > 0:  # OT only on known cells (unknown cells would break the beta marginal)
                sim_hc = torch.matmul(norm_feat[kid], proto.t())
                lab_hc, _, _, _ = ubot_CCD(sim_hc, beta, dev)
                lab_ot[kid] = lab_hc
            E[idx] = Escore_t
            pred_cls[idx] = lab_cls
            pred_ot[idx] = lab_ot

        res["target_z"] = tgt_z.cpu().numpy()
        res["Escore"] = E.cpu().numpy()
        res["pred_cls"] = pred_cls.cpu().numpy()
        res["pred_ot"] = pred_ot.cpu().numpy()
        res["threshold"] = float(thr_test)
        return res

    @staticmethod
    def _novel_test(Escore, bc_thr, n_samples):
        """dip test + bimodality coefficient -> resolution (0.5: novel cells likely, 0.8: unlikely) or None"""
        try:
            dip_test = dip(Escore, num_bins=100, n_samples=n_samples)
            dip_p_value, dip_unimodal = dip_test[-1], dip_test[-2]
            BC = calculate_bimodality_coefficient(np.sqrt(Escore))
        except Exception as e:  # never let the statistical test kill a training run
            print("bimodality test failed (%s) - resolution unchanged" % e)
            return None
        print("bimodality of dip test:", dip_p_value, not dip_unimodal)
        print("bimodality coefficient:(>%s indicates bimodality)" % bc_thr, BC, BC > bc_thr)
        bimodality = (not dip_unimodal) or (BC > bc_thr)
        print("ood sample exists:", bimodality)
        return 0.5 if bimodality else 0.8

    def _print_acc(self, iteration, res, d, tl_eval, known_mask):
        unk_mask = ~known_mask
        n_k, n_u = max(int(known_mask.sum()), 1), max(int(unk_mask.sum()), 1)
        out = {}
        for name in ("ot", "cls"):
            correct = res["pred_" + name] == tl_eval
            out[name] = (
                correct.mean(),
                correct[known_mask].sum() / n_k,
                correct[unk_mask].sum() / n_u,
                res["acc_source_" + name],
            )
        for name, tag in (("ot", "OT"), ("cls", "CLS")):
            t, k_, u_, s_ = out[name]
            print(
                "#Iter %d: (%s) Source acc: %f, Target total acc : %f,  Target known acc: %f, Target unknown acc: %f"
                % (iteration, tag, s_, t, k_, u_)
            )

    # ----------------------------------------------------------------------------------------------
    # training core (shared by train / train_eval)
    # ----------------------------------------------------------------------------------------------
    def _fit(
        self,
        mode,
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        spatial_coor,
        source_labels,
        target_labels,
        cell_types,
        common_cell_type,
        lr,
        lambda_recon,
        lambda_kl,
        lambda_spa,
        lambda_cd,
        lambda_nb,
        mu,
        temp,
        k,
        resolution,
        novel_cell_test,
        maxiter,
        miditer1,
        miditer2,
        miditer3,
        test_freq,
        sigma,
        margin,
        alpha,
        time_budget,
        dip_n_samples,
    ):
        dev = self.device
        t0 = time.time()
        bs = int(getattr(source_cm_dl, "batch_size", None) or 256)
        d = _Data(source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, source_labels, dev, bs)
        print(
            "[SPANN] data on %s: rna %d cells, spatial %d cells, batch %d (%.1fs)"
            % (dev, d.n_src, d.n_tgt, bs, time.time() - t0)
        )

        params = (
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
        )
        try:
            optim = torch.optim.Adam(params, lr=lr, weight_decay=5e-4, fused=(dev.type == "cuda"))
        except Exception:
            optim = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)

        coor = torch.as_tensor(np.asarray(spatial_coor, dtype=np.float32), device=dev)
        coor = coor / torch.max(coor, dim=0)[0]

        smooth_ce = CrossEntropyLabelSmooth(self.class_num, device=dev, epsilon=0.1)
        zero = torch.zeros((), device=dev)
        beta = None
        beta_s = np.ones(self.class_num) / self.class_num
        threshold = None
        thr_every = max(int(test_freq / 100), 1)
        print_every = max(int(test_freq / 10), 1)
        bc_thr = 0.555 if mode == "train" else 0.45

        # progress in [0, 1): phases start at miditer/maxiter, run stops at 1 (iterations or wall clock)
        f1, f2, f3 = miditer1 / maxiter, miditer2 / maxiter, miditer3 / maxiter
        t_loop = time.time()

        def get_progress(it):
            p = it / maxiter
            if time_budget:
                p = max(p, (time.time() - t_loop) / time_budget)
            return p

        if mode == "eval":
            tl = np.asarray(target_labels).astype(int)
            known_mask = np.isin(tl, np.unique(np.asarray(source_labels).astype(int)))
            tl_eval = np.where(known_mask, tl, -1)

        iteration = 0
        progress = 0.0
        dip_done = False
        while progress < 1.0:
            id_s = d.sample_source()
            id_t = d.sample_target()
            im_source = d.take(d.src_cm, id_s)
            xs_source = d.take(d.src_sp, id_s)
            label_source = d.src_lab[id_s]
            im_target = d.take(d.tgt_cm, id_t)
            xs_target = d.take(d.tgt_sp, id_t)
            batch_coor = coor[id_t]

            # domain ids as python ints: 1 = scRNA, 0 = spatial (no tensor / no sync)
            z_source, mu_source, var_source = self.encoder(im_source, 0, 1)
            z_target, mu_target, var_target = self.encoder(im_target, 0, 0)

            recon_source_c = self.decoder.dec[0](z_source, 1)
            recon_target_c = self.decoder.dec[0](z_target, 0)
            recon_source_s = self.decoder.dec[2](z_source, 1)
            recon_target_s = self.decoder.dec[1](z_target, 0)

            feat_s, logit_s = self.classifier(z_source)
            feat_t, logit_t = self.classifier(z_target)
            norm_feat_t = F.normalize(feat_t)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_source_c, im_source) + F.binary_cross_entropy(
                recon_target_c, im_target
            )
            recon_loss = recon_loss + 0.5 * (
                F.binary_cross_entropy(recon_source_s, xs_source) + F.binary_cross_entropy(recon_target_s, xs_target)
            )
            kl_loss = kl_div(mu_source, var_source) + kl_div(mu_target, var_target)
            cls_loss = smooth_ce(logit_s, label_source)

            spa_loss = zero
            ccd_loss = zero
            neighbor_loss = zero

            if progress > f1:
                proto = self.classifier.ProtoCLS.fc.weight
                if beta is None:
                    beta = np.ones(self.class_num) / self.class_num
                with torch.no_grad():
                    sim_tar_all = torch.matmul(norm_feat_t, proto.t())
                    _, _, _, wti_all = ubot_CCD(sim_tar_all, beta, dev)
                    Escore_t, _, _, _ = ensemble_score(logit_t, _minmax(wti_all))
                    # threshold from mixup of random target cells, refreshed every thr_every iterations
                    if threshold is None or iteration % thr_every == 0:
                        idx_rand = torch.randperm(z_target.size(0), device=dev)[:100]
                        f_mix, logit_mix = self.classifier(mixup_target(z_target[idx_rand]))
                        sim_mix = torch.matmul(F.normalize(f_mix), proto.t())
                        _, _, _, wti_mix = ubot_CCD(sim_mix, beta_s, dev)
                        E_mix, _, _, _ = ensemble_score(logit_mix, _minmax(wti_mix))
                        threshold = torch.minimum(E_mix.mean(), torch.quantile(E_mix, 1 - resolution))
                    high_conf_idx = Escore_t > threshold
                    n_hc = int(high_conf_idx.sum())

                if n_hc >= 2:
                    coor_hc = batch_coor[high_conf_idx]
                    feat_hc = feat_t[high_conf_idx]
                    norm_hc = norm_feat_t[high_conf_idx]
                    logit_hc = logit_t[high_conf_idx]

                    # spatial alignment loss
                    if progress > f3:
                        if mode == "train":
                            spa_loss = self._spa_loss_margin(norm_hc, coor_hc, k, sigma, margin, alpha)
                        else:
                            spa_loss = self._spa_loss_infonce(mu_target[high_conf_idx], coor_hc, k)

                    # cell type ot alignment (UOT-based CCD)
                    sim = torch.matmul(norm_hc.detach(), proto.detach().t())
                    _, new_beta, Q_t, wt_i = ubot_CCD(sim, beta, dev)
                    soft_hc = Q_t / Q_t.sum(1, keepdim=True)
                    weight_t = _minmax(wt_i)
                    if not np.isnan(new_beta).any():  # adaptive update of the marginal vector
                        beta = mu * beta + (1 - mu) * new_beta
                    soft_ok = torch.isfinite(soft_hc).all().float()  # skip ccd loss if the plan is degenerate
                    soft_hc = torch.nan_to_num(soft_hc, nan=0.0, posinf=0.0, neginf=0.0)
                    log_p = torch.log(torch.softmax(logit_hc, dim=1) + 1e-10)
                    ccd_loss = torch.mean(weight_t * (-soft_hc * log_p).sum(dim=1)) * soft_ok

                    # neighbor loss
                    if progress > f2:
                        neighbor_loss = self._neighbor_loss(coor_hc, feat_hc, norm_hc, logit_hc, Q_t, temp)

            # =====Total Loss=====
            loss = (
                lambda_recon * recon_loss
                + lambda_kl * kl_loss
                + cls_loss
                + lambda_cd * ccd_loss
                + lambda_nb * neighbor_loss
                + lambda_spa * spa_loss
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            iteration += 1
            progress = get_progress(iteration)

            if iteration % print_every == 0:
                print(
                    "#Iter %d (%.0fs, progress %.0f%%): Reconstruction loss: %f, KL loss: %f, CLS loss: %f, "
                    "Spatial loss: %f, CCD loss: %f, Neighbor loss: %f"
                    % (
                        iteration,
                        time.time() - t_loop,
                        100 * min(progress, 1.0),
                        _f(recon_loss),
                        _f(kl_loss),
                        _f(cls_loss),
                        _f(spa_loss),
                        _f(ccd_loss),
                        _f(neighbor_loss),
                    )
                )

            # (a) bimodality test when the UOT phase is about to start: decides the resolution
            if novel_cell_test and not dip_done and progress >= f1:
                dip_done = True
                if beta is None:
                    beta = np.ones(self.class_num) / self.class_num
                res_mid = self._evaluate(d, beta, beta_s, resolution, want_source=False)
                new_res = self._novel_test(res_mid["Escore"], bc_thr, dip_n_samples)
                if new_res is not None:
                    resolution = new_res
                if mode == "eval":
                    res_mid["acc_source_ot"] = res_mid["acc_source_cls"] = float("nan")
                    self._print_acc(iteration, res_mid, d, tl_eval, known_mask)
            # (b) periodic accuracy report only for train_eval without time budget (original behaviour)
            elif mode == "eval" and not time_budget and iteration % test_freq == 0 and progress < 1.0:
                if beta is None:
                    beta = np.ones(self.class_num) / self.class_num
                res_mid = self._evaluate(d, beta, beta_s, resolution, want_source=True, with_src_acc=True)
                self._print_acc(iteration, res_mid, d, tl_eval, known_mask)

        # ------------------------------ final evaluation ------------------------------
        if beta is None:
            beta = np.ones(self.class_num) / self.class_num
        res = self._evaluate(d, beta, beta_s, resolution, want_source=True, with_src_acc=(mode == "eval"))
        if mode == "eval":
            self._print_acc(iteration, res, d, tl_eval, known_mask)
        print(
            "[SPANN] done: %d iterations, loop %.0fs, total %.0fs"
            % (iteration, time.time() - t_loop, time.time() - t0)
        )

        src_lab_np = np.asarray(source_labels).astype(int)
        adata_source = sc.AnnData(sp.csr_matrix(res["source_z"]))
        adata_source.obs["cell_type"] = [cell_types[i] for i in src_lab_np]
        adata_source.obs["source"] = "scRNA"

        adata_target = sc.AnnData(sp.csr_matrix(res["target_z"]))
        names = list(cell_types) if mode == "train" else list(common_cell_type)
        total_cell_types = np.asarray(names + ["Unknown"], dtype=object)
        if mode == "eval":
            adata_target.obs["cell_type"] = [cell_types[i] for i in tl]
        adata_target.obs["pred_cls"] = total_cell_types[res["pred_cls"]]  # -1 -> "Unknown"
        adata_target.obs["pred_ot"] = total_cell_types[res["pred_ot"]]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = res["Escore"]
        adata_target.obs["source"] = "Spatial"

        return adata_source, adata_target, res["threshold"]

    # ----------------------------------------------------------------------------------------------
    # public API (signatures unchanged, new optional: time_budget, dip_n_samples)
    # ----------------------------------------------------------------------------------------------
    def train(
        self,
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        spatial_coor,
        test_source_cm_dl,
        test_target_cm_dl,
        source_labels,
        cell_types,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
        sigma=0.2,
        margin=0.5,
        alpha=0.2,
        time_budget=None,
        dip_n_samples=500,
    ):
        """
        Training function for SPANN on a spatial dataset WITHOUT ground truth labels.

        :source_cm_dl: torch dataloader of the common genes scRNA-seq data (only .dataset / .batch_size are used)
        :target_cm_dl: torch dataloader of the common genes spatial data
        :source_sp_ds: torch dataset of the scRNA-seq specific genes data
        :target_sp_ds: torch dataset of the spatial specific genes data
        :spatial_coor: pandas dataframe of the raw spatial coordinates, column names ['X','Y']
        :test_source_cm_dl, test_target_cm_dl: kept for API compatibility (evaluation uses the same in-memory data)
        :source_labels: integer labels for scRNA-seq data
        :cell_types: list of all cell types exist in scRNA-seq dataset

        :lr: learning rate for VAE and classifier
        :lambda_recon / lambda_kl / lambda_spa / lambda_cd / lambda_nb: loss weights
        :mu: updating speed of beta by moving average
        :resolution: the expected minimum proportion of known cells, 0 means no constraints on novel cell discovery and 1 means no novel cells
        :novel_cell_test: whether to apply dip test and compute BC co-efficient to adaptively select resolution
        :maxiter: maximum iteration
        :miditer1: after which iteration (of maxiter) SPANN starts to conduct UOT alignment
        :miditer2: after which iteration SPANN starts to train with neighbor loss
        :miditer3: after which iteration SPANN starts to train with adjacency loss
        :test_freq: controls the print / threshold refresh frequency
        :sigma, margin, alpha: spatial niche radius, contrastive margin, repulsion weight of the spatial loss
        :time_budget: wall-clock budget (seconds) of the training loop. Phase starts are expressed as
                      fractions of maxiter and are reached on max(iter/maxiter, elapsed/time_budget), so
                      the whole schedule is compressed to fit the budget. None = pure iteration based.
        :dip_n_samples: Monte-Carlo samples for the dip test null distribution

        :return: adata_source, adata_target, threshold
        """
        return self._fit(
            "train",
            source_cm_dl,
            target_cm_dl,
            source_sp_ds,
            target_sp_ds,
            spatial_coor,
            source_labels,
            None,
            cell_types,
            None,
            lr,
            lambda_recon,
            lambda_kl,
            lambda_spa,
            lambda_cd,
            lambda_nb,
            mu,
            temp,
            k,
            resolution,
            novel_cell_test,
            maxiter,
            miditer1,
            miditer2,
            miditer3,
            test_freq,
            sigma,
            margin,
            alpha,
            time_budget,
            dip_n_samples,
        )

    def train_eval(
        self,
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        spatial_coor,
        test_source_cm_dl,
        test_target_cm_dl,
        source_labels,
        target_labels,
        cell_types,
        common_cell_type,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
        time_budget=None,
        dip_n_samples=500,
    ):
        """
        Training + evaluation on a spatial dataset WITH ground truth labels (prints accuracies).
        Same arguments as `train`, plus:

        :target_labels: integer labels for spatial data
        :cell_types: list of all cell types exist in scRNA-seq or spatial datasets
        :common_cell_type: list of cell types exist in both scRNA-seq and spatial datasets
        """
        return self._fit(
            "eval",
            source_cm_dl,
            target_cm_dl,
            source_sp_ds,
            target_sp_ds,
            spatial_coor,
            source_labels,
            target_labels,
            cell_types,
            common_cell_type,
            lr,
            lambda_recon,
            lambda_kl,
            lambda_spa,
            lambda_cd,
            lambda_nb,
            mu,
            temp,
            k,
            resolution,
            novel_cell_test,
            maxiter,
            miditer1,
            miditer2,
            miditer3,
            test_freq,
            0.2,
            0.5,
            0.2,
            time_budget,
            dip_n_samples,
        )
