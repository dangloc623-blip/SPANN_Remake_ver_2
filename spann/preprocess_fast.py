"""
Fast / modern-stack preprocessing for SPANN (same API and outputs as preprocess.py).

Differences:
  * works with current anndata / pandas / scanpy (no AnnData.concatenate, no chained assignment)
  * MaxAbs scaling is one vectorised numpy op (original: chunked sparse fancy-assignment)
  * HVG selection is skipped when a dataset has <= `highly_variable` genes (it was a no-op anyway)
  * no torch DataLoader / worker processes: `generate_dataloaders` returns light objects that only expose
    `.dataset` and `.batch_size`, which is all `model_fast.SPANN_model` needs.
"""
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def _dense32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _maxabs_(adata, groups=None):
    """column-wise X / max|X| (per group of rows if `groups` is given); result stored dense float32."""
    X = _dense32(adata.X)
    blocks = [np.arange(X.shape[0])] if groups is None else [np.where(groups == g)[0] for g in np.unique(groups)]
    for idx in blocks:
        m = np.abs(X[idx]).max(0)
        m[m == 0] = 1.0
        X[idx] = X[idx] / m
    adata.X = X


def _hvg_(adata, n_top, batch_key=None):
    if adata.n_vars > n_top:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, batch_key=batch_key, subset=True)


def anndata_preprocess(adata_spa, adata_rna, highly_variable=2000, spatial_labels=False):
    """
    :adata_spa: spatial AnnData, .obs has 'X','Y' (and 'cell_type' if spatial_labels)
    :adata_rna: scRNA AnnData, .obs has 'cell_type'
    :return: adata_cm (common genes, both domains; domain_id '0'=spatial, '1'=rna), adata_spa, adata_rna
    """
    adata_cm = ad.concat([adata_spa, adata_rna], join="inner", label="domain_id", keys=["0", "1"], index_unique="-")
    adata_cm.obs_names_make_unique()

    rna_ct = adata_rna.obs["cell_type"].astype(str).values
    if spatial_labels:
        spa_ct = adata_spa.obs["cell_type"].astype(str).values
        common = [c for c in np.unique(rna_ct) if c in set(spa_ct)]
        novel = [c for c in np.unique(spa_ct) if c not in set(rna_ct)]
        if len(set(rna_ct)) > len(common):
            print("WARNING: scRNA has cell types absent from spatial data; they get label 0 (same as original code)")
        mapping = {c: i for i, c in enumerate(common + novel)}
        rna_labels = np.array([mapping.get(c, 0) for c in rna_ct], dtype=int)
        spa_labels = np.array([mapping.get(c, 0) for c in spa_ct], dtype=int)
        adata_rna.obs["labels"] = pd.Categorical(rna_labels, categories=np.unique(rna_labels))
        adata_spa.obs["labels"] = pd.Categorical(spa_labels, categories=np.unique(spa_labels))
        print("rna_labels:", np.unique(rna_labels))
        print("spatial_labels:", np.unique(spa_labels))
    else:
        types = list(np.unique(rna_ct))
        rna_labels = np.array([types.index(c) for c in rna_ct], dtype=int)
        adata_rna.obs["labels"] = pd.Categorical(rna_labels, categories=np.unique(rna_labels))
        adata_spa.obs["labels"] = -1
        print("rna_labels:", np.unique(rna_labels))

    for a in (adata_cm, adata_spa, adata_rna):
        a.X = _dense32(a.X) if a.n_obs * a.n_vars < 3e8 else a.X.astype(np.float32)
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
    _hvg_(adata_cm, highly_variable, batch_key="domain_id")
    _hvg_(adata_spa, highly_variable)
    _hvg_(adata_rna, highly_variable)
    _maxabs_(adata_cm, groups=adata_cm.obs["domain_id"].values)
    _maxabs_(adata_spa)
    _maxabs_(adata_rna)
    return adata_cm, adata_spa, adata_rna


class SingleCellDataset:
    def __init__(self, data, labels):
        self.data = _dense32(data)
        self.labels = np.asarray(labels)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], idx


def generate_dataloaders(adata_cm, adata_spa, adata_rna, batch_size=256, num_workers=0):
    """Same return signature as preprocess.generate_dataloaders (loaders are only descriptors here)."""
    cm_rna = adata_cm[adata_cm.obs["domain_id"] == "1"]
    cm_spa = adata_cm[adata_cm.obs["domain_id"] == "0"]
    rna_labels = np.array(adata_rna.obs["labels"])
    spa_labels = np.array(adata_spa.obs["labels"])
    source_cm_ds = SingleCellDataset(cm_rna.X, rna_labels)
    target_cm_ds = SingleCellDataset(cm_spa.X, spa_labels)
    source_sp_ds = SingleCellDataset(adata_rna.X, rna_labels)
    target_sp_ds = SingleCellDataset(adata_spa.X, spa_labels)

    def dl(ds):
        return SimpleNamespace(dataset=ds, batch_size=batch_size)

    return source_sp_ds, target_sp_ds, dl(source_cm_ds), dl(target_cm_ds), dl(source_cm_ds), dl(target_cm_ds)


def generate_ae_params(adata_cm, adata_spa, adata_rna, feat_dim=16):
    """Same as preprocess.generate_ae_params; x_dim = {0: common genes, 1: spatial genes, 2: rna genes}."""
    n_common, n_spa, n_rna = adata_cm.X.shape[1], adata_spa.X.shape[1], adata_rna.X.shape[1]
    enc = [["fc", 1024, 1, "relu"], ["fc", feat_dim, "", ""]]
    dec = {0: [["fc", n_common, 2, "sigmoid"]], 1: [["fc", n_spa, 1, "sigmoid"]], 2: [["fc", n_rna, 1, "sigmoid"]]}
    x_dim = {key: dec[key][-1][1] for key in dec}
    return enc, dec, x_dim, feat_dim
