# SPANN fast (Kaggle ~5 min)
New files: `spann/model_fast.py`, `spann/preprocess_fast.py`, `spann/diptest.py`, `kaggle/kaggle_spann_5min.ipynb`.
Dependencies: torch, anndata, scanpy, numpy, scipy (no POT, no easydl).
Drop-in use: replace `from model import *` by `from spann.model_fast import SPANN_model` and
`from preprocess import *` by `from spann.preprocess_fast import *`; `train()/train_eval()` accept the same arguments
plus `time_budget=<seconds>`, `batch_size`, `log_every`.
