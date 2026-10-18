# %% [markdown]
# # SPANN fast - whole pipeline in ~5 minutes on Kaggle (GPU T4/P100)
# Upload the repo folder (SPANN_Remake-main) and the two h5ad files as Kaggle datasets,
# turn on **GPU** and **Internet** (only needed once for `pip install`), then Run All.

# %%
import importlib.util, subprocess, sys, time
T_START = time.time()
for pkg in ("anndata", "scanpy"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

# %%
import glob, os, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, torch, anndata as ad
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(no GPU: expect it to be much slower)")

# ---- locate repo + data (edit the patterns if your dataset names differ)
def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) + glob.glob(f"./**/{pattern}", recursive=True)
    assert hits, f"cannot find {pattern} - add it as a Kaggle dataset"
    return hits[0]

REPO = os.path.dirname(os.path.dirname(find("preprocess_fast.py")))
sys.path.insert(0, REPO)
PATH_RNA = find("adata_rna.h5ad")
PATH_SPA = find("adata_seqfish_40.h5ad")

# %%
# ------------------------------ settings ------------------------------
WITH_SPATIAL_LABELS = False   # True: spatial data has 'cell_type' -> print accuracies (tutorial 2 style)
TRAIN_SECONDS = 170           # wall-clock budget of the training loop (whole notebook ~ 4-5 min)
MAXITER = 5000                # upper bound, training stops earlier if the time budget is reached
BATCH = 256
HP = dict(lr=2e-4, resolution=0.5, lambda_spa=0.001, lambda_cd=0.001, lambda_nb=10)

# %%
from spann.preprocess_fast import anndata_preprocess, generate_dataloaders, generate_ae_params
from spann.model_fast import SPANN_model

t = time.time()
adata_rna = ad.read_h5ad(PATH_RNA)
adata_spa = ad.read_h5ad(PATH_SPA)
true_ct = None
if not WITH_SPATIAL_LABELS and "cell_type" in adata_spa.obs:
    true_ct = adata_spa.obs["cell_type"].astype(str).values.copy()   # kept only for the final check
    adata_spa.obs = adata_spa.obs.drop(columns="cell_type")

adata_cm, adata_spa, adata_rna = anndata_preprocess(adata_spa, adata_rna, spatial_labels=WITH_SPATIAL_LABELS)
src_sp, tgt_sp, src_dl, tgt_dl, test_src_dl, test_tgt_dl = generate_dataloaders(adata_cm, adata_spa, adata_rna, batch_size=BATCH)
enc, dec, x_dim, z_dim = generate_ae_params(adata_cm, adata_spa, adata_rna)
print(f"preprocess: {time.time()-t:.0f}s | rna {adata_rna.shape} spatial {adata_spa.shape} common {adata_cm.shape}")

# %%
model = SPANN_model(x_dim, z_dim, enc, dec, class_num=len(adata_rna.obs["cell_type"].unique()), device=device)
rna_labels = np.array(adata_rna.obs["labels"]).astype(int)
rna_types = np.unique(adata_rna.obs["cell_type"])
coords = adata_spa.obs[["X", "Y"]]
common = dict(
    maxiter=MAXITER, miditer1=int(.4 * MAXITER), miditer2=int(.8 * MAXITER), miditer3=int(.8 * MAXITER),
    time_budget=TRAIN_SECONDS, **HP)

if WITH_SPATIAL_LABELS:
    spa_labels = np.array(adata_spa.obs["labels"]).astype(int)
    common_ct = [c for c in rna_types if c in set(adata_spa.obs["cell_type"])]
    novel_ct = [c for c in np.unique(adata_spa.obs["cell_type"]) if c not in set(rna_types)]
    adata_source, adata_target, thr = model.train_eval(
        src_dl, tgt_dl, src_sp, tgt_sp, coords, test_src_dl, test_tgt_dl,
        rna_labels, spa_labels, common_ct + novel_ct, common_ct, **common)
else:
    adata_source, adata_target, thr = model.train(
        src_dl, tgt_dl, src_sp, tgt_sp, coords, test_src_dl, test_tgt_dl, rna_labels, rna_types, **common)
print(f"TOTAL wall-clock so far: {(time.time()-T_START)/60:.1f} min")

# %%
# quick look (matplotlib only: seaborn lmplot / UMAP on 27k cells would cost minutes)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for c in np.unique(adata_target.obs["pred_cls"]):
    m = (adata_target.obs["pred_cls"] == c).values
    ax[0].scatter(adata_target.obs.X[m], adata_target.obs.Y[m], s=1, label=c)
ax[0].set_title("pred_cls"); ax[0].legend(markerscale=8, fontsize=6, ncol=2)
ax[1].hist(adata_target.obs["Escore"], bins=25, color="green"); ax[1].axvline(thr, color="brown")
ax[1].set_title(f"E-score (threshold {thr:.2f})")
plt.show()

if os.path.isdir("/kaggle/working"):
    adata_target.obs.to_csv("/kaggle/working/spann_predictions.csv")

# %%
if true_ct is not None:   # optional check when ground truth was available but hidden from the model
    from spann.evaluation import evaluate_novel_discovery
    pred = adata_target.obs["pred_cls"].astype(str).values
    print(evaluate_novel_discovery(true_ct, pred, list(rna_types.astype(str))))
