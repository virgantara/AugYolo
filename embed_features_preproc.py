import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ==== your project pieces ====
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
        # allow partial load (different key prefixes)
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
    """Captures input to the LAST Linear layer (penultimate embedding)."""
    def __init__(self, layer: nn.Linear):
        self.buffer = []
        self.h = layer.register_forward_pre_hook(self._hook)
    def _hook(self, module, inputs):
        # inputs is a tuple; take first tensor
        x = inputs[0].detach()
        if x.ndim > 2:  # flatten if needed
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
    # embeds_dict: {pipeline: (N,2)}
    # labels_dict: {pipeline: (N,)}
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
    ap.add_argument("--save_plot", default="out/feature_space.png")
    # op params
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, nargs=2, default=(8,8))
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--wavelet_level", type=int, default=2)
    ap.add_argument("--unsharp_amount", type=float, default=0.7)
    ap.add_argument("--unsharp_radius", type=float, default=1.0)
    ap.add_argument("--unsharp_threshold", type=int, default=2)
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

    # loaders (same sample order; only transform differs)
    loaders = {}
    for name, tfm in pipes.items():
        loaders[name] = make_loader(
            split=val_xlsx, metadata_xlsx=meta_xlsx, images_dir=images_dir, transform=tfm,
            batch=args.batch_size, workers=args.num_workers, center_split=args.use_center_split
        )

    # collect features and embed
    embeds2d, labels_map = {}, {}
    class_names = ["Normal", "B-tumor", "M-tumor"]
    for name, loader in loaders.items():
        F, y = collect_features(model, loader, device)
        Z = embed_2d(F, method=args.method, seed=args.seed)
        embeds2d[name] = Z
        labels_map[name] = y
        print(f"{name:>26s}: features {F.shape} -> embed {Z.shape}")

    plot_grid(embeds2d, labels_map, class_names, out_path=args.save_plot,
              title_suffix=f"({args.method.upper()})")

if __name__ == "__main__":
    main()
