import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ==== project pieces (sesuai kode Anda) ====
from dataset import BoneTumorDataset, BoneTumorDatasetCenter
from util import CLAHE
from utils_xray import WaveletDenoise, UnsharpMask

from models_yolo import (ClassificationModel)
from models import (
    YOLOv8ClsFromYAML,
    ConvNeXtBTXRD,
    EfficientNetBTXRD,
    EfficientNetB4BTXRD,
    ResNet50,
)

# ===== sklearn untuk metrik & CV =====
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import pandas as pd

# ------------- utilities -------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(name: str, num_classes: int = 3):
    name = name.lower()
    if name == "resnet50":
        return ResNet50(num_classes=num_classes)
    elif name == "efficientnetb0":
        return EfficientNetBTXRD(num_classes=num_classes)
    elif name == "efficientnetb4":
        return EfficientNetB4BTXRD(num_classes=num_classes)
    elif name == "convnext":
        return ConvNeXtBTXRD(num_classes=num_classes)
    elif name == "yolov8":
        cfg = os.path.join('yolo/cfg','models','v8',f'yolov8n-cls.yaml')
        model = ClassificationModel(cfg, nc=3, ch=3)
        return model
    else:
        raise ValueError(f"Unknown model_name: {name}")

def load_ckpt_if_any(model, ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}\n  missing={len(missing)} unexpected={len(unexpected)}")
    return model

def last_linear_layer(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear found; please adapt hook to your model.")
    return last

class PenultimateHook:
    """Tangkap input ke layer Linear terakhir (embedding penultimate)."""
    def __init__(self, layer: nn.Linear):
        self.buffer = []
        self.h = layer.register_forward_pre_hook(self._hook)
    def _hook(self, module, inputs):
        x = inputs[0].detach()
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        self.buffer.append(x.cpu())
    def pop(self):
        out = torch.cat(self.buffer, dim=0) if self.buffer else torch.empty(0)
        self.buffer.clear()
        return out
    def close(self):
        self.h.remove()

# ----- preprocessing pipelines (deterministic) -----
def make_pipelines(img_size, clahe_clip=2.0, clahe_grid=(8,8),
                   wavelet="db2", level=2,
                   un_amount=0.7, un_radius=1.0, un_thresh=2):
    resize = transforms.Resize((img_size, img_size))
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    P0 = transforms.Compose([resize, transforms.ToTensor(), norm])  # Original
    P1 = transforms.Compose([CLAHE(clahe_clip, clahe_grid, p=1.0), resize, transforms.ToTensor(), norm])
    P2 = transforms.Compose([CLAHE(clahe_clip, clahe_grid, p=1.0),
                             WaveletDenoise(wavelet=wavelet, level=level, p=1.0),
                             resize, transforms.ToTensor(), norm])
    P3 = transforms.Compose([CLAHE(clahe_clip, clahe_grid, p=1.0),
                             WaveletDenoise(wavelet=wavelet, level=level, p=1.0),
                             UnsharpMask(amount=un_amount, radius=un_radius, threshold=un_thresh, p=1.0),
                             resize, transforms.ToTensor(), norm])
    return {
        "Original": P0,
        "CLAHE": P1,
        "CLAHE+Wavelet": P2,
        "CLAHE+Wavelet+Unsharp": P3
    }

def make_loader(split: str, metadata_xlsx: str, images_dir: str, transform, batch=32, workers=4,
                center_split=False, center_id_for_test=3):
    if center_split:
        ds = BoneTumorDatasetCenter(metadata_xlsx_path=metadata_xlsx, image_dir=images_dir,
                                    center_id=center_id_for_test, transform=transform)
    else:
        ds = BoneTumorDataset(split_xlsx_path=split, metadata_xlsx_path=metadata_xlsx,
                              image_dir=images_dir, transform=transform)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

# ----- embedding helpers -----
def embed_2d(features: np.ndarray, method="umap", seed=42):
    Z = None
    if method.lower() == "umap":
        try:
            import umap.umap_ as umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                                random_state=seed)
            Z = reducer.fit_transform(features)
        except Exception as e:
            print(f"UMAP not available ({e}); falling back to t-SNE.")
            method = "tsne"
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto",
                 random_state=seed).fit_transform(features)
    return Z

def collect_features(model, loader, device):
    model.eval()
    layer = last_linear_layer(model)
    hook = PenultimateHook(layer)
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)        # features captured by hook
            ys.append(y.numpy())
    F = hook.pop().numpy()
    hook.close()
    y = np.concatenate(ys, axis=0)
    return F, y

def plot_grid(embeds_dict, labels_dict, class_names, out_path=None, title_suffix=""):
    cmap = {0:"#4C78A8", 1:"#F58518", 2:"#54A24B"}  # Normal, B, M
    variants = list(embeds_dict.keys())
    cols = 2; rows = int(np.ceil(len(variants)/cols))
    plt.figure(figsize=(6*cols, 5*rows))
    for i, v in enumerate(variants, 1):
        Z = embeds_dict[v]; y = labels_dict[v]
        ax = plt.subplot(rows, cols, i)
        for c in np.unique(y):
            m = (y == c)
            ax.scatter(Z[m,0], Z[m,1], s=12, alpha=0.75, label=class_names[c], c=cmap.get(int(c), None))
        ax.set_title(f"{v} {title_suffix}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

# ====== metrik separabilitas & CV klasifikasi ======
def fisher_ratio(X, y):
    Xs = StandardScaler().fit_transform(X)
    classes = np.unique(y)
    mu = Xs.mean(axis=0)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = Xs[y==c]
        mu_c = Xc.mean(axis=0)
        nc = Xc.shape[0]
        Sb += nc * np.sum((mu_c - mu)**2)
        Sw += np.sum((Xc - mu_c)**2)
    return float(Sb / (Sw + 1e-12))

def cluster_metrics(F, y):
    X = StandardScaler().fit_transform(F)
    return {
        "Silhouette":          float(silhouette_score(X, y, metric="euclidean")),
        "CalinskiHarabasz":    float(calinski_harabasz_score(X, y)),
        "DaviesBouldin":       float(davies_bouldin_score(X, y)),
        "FisherRatio":         float(fisher_ratio(F, y)),
    }

def cv_classify_metrics(F, y, class_names, seed=42, n_splits=5):
    X = StandardScaler().fit_transform(F)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = {
        "LogReg": LogisticRegression(max_iter=4000, n_jobs=1),
        "kNN-5":  KNeighborsClassifier(n_neighbors=5),
        "NearestCentroid": NearestCentroid()
    }

    summary_rows = []
    per_class_rows = []

    for mname, clf in models.items():
        y_true_all, y_pred_all = [], []
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            yp = clf.predict(X[te])
            y_true_all.append(y[te]); y_pred_all.append(yp)
        y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
        )
        macroF1 = float(np.mean(f1))
        summary_rows.append({"Model": mname, "Accuracy": acc, "MacroF1": macroF1})

        for i, cname in enumerate(class_names):
            per_class_rows.append({
                "Model": mname, "Class": cname,
                "Precision": float(prec[i]),
                "Recall": float(rec[i]),
                "F1": float(f1[i]),
                "Support": int(sup[i]),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--dataset_dir", default="data/BTXRD")
    ap.add_argument("--use_center_split", action="store_true",
                    help="If set, uses Center 3 as test; otherwise uses val.xlsx")
    ap.add_argument("--img_size", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    # model
    ap.add_argument("--model_name", default="resnet50", choices=[
        "resnet50","efficientnetb0","efficientnetb4","convnext","yolov8"
    ])
    ap.add_argument("--ckpt", default=None, help="Path to model checkpoint (.pt/.pth)")
    # viz/emb
    ap.add_argument("--method", default="umap", choices=["umap","tsne"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_plot", default=None, help="If set, saves UMAP/TSNE grid")
    # op params
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, nargs=2, default=(8,8))
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--wavelet_level", type=int, default=2)
    ap.add_argument("--unsharp_amount", type=float, default=0.7)
    ap.add_argument("--unsharp_radius", type=float, default=1.0)
    ap.add_argument("--unsharp_threshold", type=int, default=2)
    # output CSV
    ap.add_argument("--save_csv_prefix", default=None, help="e.g., out/metrics ; will create *_cluster.csv & *_perclass.csv")
    args = ap.parse_args()

    set_seed(args.seed)

    DATASET_DIR = args.dataset_dir
    meta_xlsx = os.path.join(DATASET_DIR, "dataset.xlsx")
    val_xlsx  = os.path.join(DATASET_DIR, "val.xlsx")
    images_dir = os.path.join(DATASET_DIR, "images")

    # build model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, num_classes=3)
    model = load_ckpt_if_any(model, args.ckpt).to(device)

    # pipelines
    pipes = make_pipelines(
        img_size=args.img_size,
        clahe_clip=args.clahe_clip, clahe_grid=tuple(args.clahe_grid),
        wavelet=args.wavelet, level=args.wavelet_level,
        un_amount=args.unsharp_amount, un_radius=args.unsharp_radius, un_thresh=args.unsharp_threshold
    )

    # loaders (sample order sama; hanya transform yang beda)
    loaders = {}
    for name, tfm in pipes.items():
        loaders[name] = make_loader(
            split=val_xlsx, metadata_xlsx=meta_xlsx, images_dir=images_dir, transform=tfm,
            batch=args.batch_size, workers=args.num_workers, center_split=args.use_center_split
        )

    class_names = ["Normal", "B-tumor", "M-tumor"]
    embeds2d, labels_map = {}, {}

    # kumpulkan fitur + metrik untuk tiap pipeline
    cluster_rows = []
    perclass_all = []

    for name, loader in loaders.items():
        F, y = collect_features(model, loader, device)
        print(f"{name:>26s}: features {F.shape}")

        # separabilitas klaster
        cm = cluster_metrics(F, y)
        cluster_rows.append({"Pipeline": name, **cm})

        # 5-fold CV di fitur penultimate
        df_sum, df_percls = cv_classify_metrics(F, y, class_names, seed=args.seed)
        print("\nCV classification (5-fold) on penultimate features —", name)
        print(df_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nPer-class:")
        print(df_percls.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        df_percls.insert(0, "Pipeline", name)
        perclass_all.append(df_percls)

        # opsional: embedding 2D utk visual
        if args.save_plot:
            Z = embed_2d(F, method=args.method, seed=args.seed)
            embeds2d[name] = Z
            labels_map[name] = y

    # tampilkan ringkasan klaster
    cluster_df = pd.DataFrame(cluster_rows)
    print("\n=== Cluster separability summary ===")
    print(cluster_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # simpan CSV bila diminta
    if
