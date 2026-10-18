import json
import os

with open("seq-fish.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Markdown
nb["cells"][0]["source"] = (
    "# SPANN fast - whole pipeline in ~5 minutes on Kaggle (GPU T4/P100)\n"
    "Tối ưu hóa chạy trên Kaggle GPU (T4 / P100) với **model_spatial.py** lightweight.\n"
    "Bật **GPU** và **Internet** trong Kaggle notebook settings, sau đó bấm **Run All**."
)

# Cell 1: Repo / Environment setup
nb["cells"][1]["source"] = """import importlib.util, os, subprocess, sys, time
T_START = time.time()

# 1. Tìm hoặc clone mã nguồn SPANN
REPO_URL = "https://github.com/dangloc623-blip/SPANN_Remake_ver_2.git"
BRANCH = "main"
CODE_DIR = "/kaggle/working/spann_repo"

# Nếu đã có thư mục spann ở thư mục hiện tại hoặc lân cận thì dùng trực tiếp
found_repo = None
for candidate in [".", "..", CODE_DIR]:
    if os.path.isdir(os.path.join(candidate, "spann")):
        found_repo = os.path.abspath(candidate)
        break

if not found_repo:
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        REPO_URL = REPO_URL.replace("https://", f"https://{_tok}@")
    except Exception:
        pass
    if os.path.isdir(CODE_DIR):
        try:
            subprocess.run(["git", "-C", CODE_DIR, "pull", "-q"], check=False)
        except Exception:
            pass
    else:
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, "--depth", "1", REPO_URL, CODE_DIR], check=True)
    found_repo = CODE_DIR

if found_repo not in sys.path:
    sys.path.insert(0, found_repo)
print(f"Using SPANN package from: {found_repo}")

# 2. Cài đặt các thư viện cần thiết nếu thiếu
for pkg in ("anndata", "scanpy", "pot"):
    spec_name = "ot" if pkg == "pot" else pkg
    if importlib.util.find_spec(spec_name) is None:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
"""

# Cell 2: Data paths
nb["cells"][2]["source"] = """import glob, os, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, torch, anndata as ad
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "(CPU - expect slower runtime)")

# ---- DATA PATHS ----
def find_file(pattern, candidate_dir="/kaggle/input"):
    # Tìm kiếm linh hoạt trong /kaggle/input hoặc thư mục hiện tại
    hits = glob.glob(f"{candidate_dir}/**/{pattern}", recursive=True)
    if not hits:
        hits = glob.glob(f"./**/{pattern}", recursive=True) or glob.glob(f"../**/{pattern}", recursive=True)
    return hits[0] if hits else None

PATH_RNA = None
PATH_SPA = None

# Thử các đường dẫn mặc định phổ biến
default_rna = "/kaggle/input/datasets/leebaam/seq-fish-dataset/adata_rna.h5ad"
default_spa = "/kaggle/input/datasets/leebaam/seq-fish-dataset/adata_seqfish_40.h5ad"

if os.path.isfile(default_rna):
    PATH_RNA = default_rna
else:
    PATH_RNA = find_file("adata_rna*.h5ad") or find_file("*rna*.h5ad")

if os.path.isfile(default_spa):
    PATH_SPA = default_spa
else:
    PATH_SPA = find_file("adata_seqfish*.h5ad") or find_file("*seqfish*.h5ad") or find_file("*spatial*.h5ad")

assert PATH_RNA and os.path.exists(PATH_RNA), f"Không tìm thấy file RNA h5ad. Hãy add dataset vào Kaggle hoặc chỉ định PATH_RNA."
assert PATH_SPA and os.path.exists(PATH_SPA), f"Không tìm thấy file Spatial h5ad. Hãy add dataset vào Kaggle hoặc chỉ định PATH_SPA."

print("Data paths:\n  RNA:", PATH_RNA, "\n  SPA:", PATH_SPA)
"""

# Cell 3: Settings
nb["cells"][3]["source"] = """# ------------------------------ Settings ------------------------------
WITH_SPATIAL_LABELS = False   # True: spatial data có sẵn 'cell_type' -> in độ chính xác
TRAIN_SECONDS = 180           # Khung thời gian tối đa cho training: 180s (3 phút). Toàn bộ notebook < 5 phút!
MAXITER = 5000                # Giới hạn số vòng lặp tối đa
BATCH = 256                   # Kích thước batch tối ưu cho GPU T4/P100
LIGHTWEIGHT = True            # Bật cấu hình lightweight (hidden_mlp=512) giúp tính toán nhanh hơn

HP = dict(
    lr=2e-4, 
    resolution=0.5, 
    lambda_spa=0.001, 
    lambda_cd=0.001, 
    lambda_nb=10
)
"""

# Cell 4: Preprocessing
nb["cells"][4]["source"] = """from spann.preprocess import anndata_preprocess, generate_dataloaders, generate_ae_params
from spann.model_spatial import SPANN_model

t_pre = time.time()
adata_rna = ad.read_h5ad(PATH_RNA)
adata_spa = ad.read_h5ad(PATH_SPA)

true_ct = None
if not WITH_SPATIAL_LABELS and "cell_type" in adata_spa.obs:
    true_ct = adata_spa.obs["cell_type"].astype(str).values.copy()   # Giữ lại để đánh giá cuối cùng
    adata_spa.obs = adata_spa.obs.drop(columns="cell_type")

adata_cm, adata_spa, adata_rna = anndata_preprocess(adata_spa, adata_rna, spatial_labels=WITH_SPATIAL_LABELS)
src_sp, tgt_sp, src_dl, tgt_dl, test_src_dl, test_tgt_dl = generate_dataloaders(adata_cm, adata_spa, adata_rna, batch_size=BATCH)
enc, dec, x_dim, z_dim = generate_ae_params(adata_cm, adata_spa, adata_rna)

print(f"Preprocess hoàn tất trong: {time.time()-t_pre:.1f}s | RNA: {adata_rna.shape}, Spatial: {adata_spa.shape}, Common: {adata_cm.shape}")
"""

# Cell 5: Model training
nb["cells"][5]["source"] = """# Khởi tạo SPANN model từ model_spatial.py với tùy chọn lightweight
hidden_mlp = 512 if LIGHTWEIGHT else 1024
model = SPANN_model(
    x_dim, z_dim, enc, dec, 
    class_num=len(adata_rna.obs["cell_type"].unique()), 
    device=device,
    hidden_mlp=hidden_mlp
)

rna_labels = np.array(adata_rna.obs["labels"]).astype(int)
rna_types = np.unique(adata_rna.obs["cell_type"])
coords = adata_spa.obs[["X", "Y"]]

common = dict(
    maxiter=MAXITER, 
    miditer1=int(0.4 * MAXITER), 
    miditer2=int(0.8 * MAXITER), 
    miditer3=int(0.8 * MAXITER),
    time_budget=TRAIN_SECONDS, 
    **HP
)

print(f"Bắt đầu huấn luyện mô hình (giới hạn thời gian budget = {TRAIN_SECONDS}s)...")
if WITH_SPATIAL_LABELS:
    spa_labels = np.array(adata_spa.obs["labels"]).astype(int)
    common_ct = [c for c in rna_types if c in set(adata_spa.obs["cell_type"])]
    novel_ct = [c for c in np.unique(adata_spa.obs["cell_type"]) if c not in set(rna_types)]
    adata_source, adata_target, thr = model.train_eval(
        src_dl, tgt_dl, src_sp, tgt_sp, coords, test_src_dl, test_tgt_dl,
        rna_labels, spa_labels, common_ct + novel_ct, common_ct, **common
    )
else:
    adata_source, adata_target, thr = model.train(
        src_dl, tgt_dl, src_sp, tgt_sp, coords, test_src_dl, test_tgt_dl, 
        rna_labels, rna_types, **common
    )

total_elapsed = time.time() - T_START
print(f"===> TOÀN BỘ PIPELINE ĐÃ CHẠY XONG TRONG: {total_elapsed/60:.2f} phút ({total_elapsed:.1f}s) <===")
"""

# Cell 6: Visualization
nb["cells"][6]["source"] = """# Trực quan hóa kết quả nhanh chóng với matplotlib
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
unique_preds = np.unique(adata_target.obs["pred_cls"])
for c in unique_preds:
    m = (adata_target.obs["pred_cls"] == c).values
    ax[0].scatter(adata_target.obs.X[m], adata_target.obs.Y[m], s=2, label=c, alpha=0.8)
ax[0].set_title("Spatial Predicted Cell Types (pred_cls)")
ax[0].legend(markerscale=6, fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")

ax[1].hist(adata_target.obs["Escore"], bins=30, color="teal", edgecolor="black", alpha=0.7)
if thr is not None:
    ax[1].axvline(float(thr), color="red", linestyle="--", linewidth=2, label=f"Threshold ({float(thr):.2f})")
    ax[1].legend()
ax[1].set_title("E-score Distribution (Novel Discovery)")

plt.tight_layout()
plt.show()

# Lưu kết quả dự đoán ra file CSV
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
csv_path = os.path.join(out_dir, "spann_predictions.csv")
adata_target.obs.to_csv(csv_path)
print(f"Đã lưu kết quả dự đoán tại: {csv_path}")
"""

# Cell 7: Novel evaluation
nb["cells"][7]["source"] = """if true_ct is not None:
    try:
        from spann.evaluation import evaluate_novel_discovery
        pred = adata_target.obs["pred_cls"].astype(str).values
        metrics = evaluate_novel_discovery(true_ct, pred, list(rna_types.astype(str)))
        print("Đánh giá độ chính xác (Ground Truth vs Prediction):")
        print(metrics)
    except Exception as e:
        print(f"Bỏ qua bước đánh giá chi tiết: {e}")
"""

# Lưu file seq-fish.ipynb
with open("seq-fish.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Đồng thời tạo file seq_fish.ipynb (với dấu gạch dưới) để tiện cho người dùng
with open("seq_fish.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cập nhật thành công cả seq-fish.ipynb và seq_fish.ipynb!")
