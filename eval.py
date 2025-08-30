import torch
from torch.utils.data import random_split, DataLoader
from dataset import BoneTumorDataset
import os
from torchvision import transforms
import argparse
from util import top_k_accuracy, append_row_to_excel, FocalCE, CLAHE
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from models_yolo import (ClassificationModel)

from models import (
    YOLOv8ClsFromYAML, 
    ConvNeXtBTXRD, 
    EfficientNetBTXRD,
    EfficientNetB4BTXRD,
    ResNet50
)
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import matplotlib
matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt
from van import VAN, load_model_weights
from timm.models.vision_transformer import _cfg
from functools import partial
from transforms_factory import build_transforms

def compute_imbalanced_metrics(y_true_np, y_pred_np, probs_np, num_classes=3):
    # Per-class metrics
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(num_classes)), zero_division=0
    )

    # Aggregates
    metrics = {
        "acc": (y_true_np == y_pred_np).mean(),
        "macro_f1": f1_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true_np, y_pred_np, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true_np, y_pred_np),
        "kappa": cohen_kappa_score(y_true_np, y_pred_np),
        "mcc": matthews_corrcoef(y_true_np, y_pred_np),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": supp.tolist(),
    }

    # Prob-based metrics (gracefully handle missing classes)
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true_np, probs_np, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_ovr_macro"] = None

    try:
        metrics["pr_auc_macro"] = average_precision_score(y_true_np, probs_np, average="macro")
    except Exception:
        metrics["pr_auc_macro"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    return metrics, cm

def _get_logits(outputs):
    # Normalize various model return types to a single Tensor of logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    if isinstance(outputs, dict):
        # common keys to try; fall back to first value
        for k in ('logits', 'out', 'pred', 'cls'):
            if k in outputs:
                return outputs[k]
        return next(iter(outputs.values()))
    return outputs  # already a Tensor

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return fig


def main(args):
    set_seed(args)
    print("Hyper-parameters: {}".format(args.__str__()))

    
    
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    
    _, test_transform = build_transforms(args)
    
    test_dataset = BoneTumorDataset(
        split_xlsx_path=test_path,
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        transform=test_transform
    )

    CLASS_NAMES = ['normal', 'benign', 'malignant']  # <-- adjust if needed

    print("Val label distribution:")
    print(test_dataset.df['label'].value_counts().sort_index())

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                          num_workers=args.num_worker, pin_memory=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    y_labels = test_dataset.df['label'].tolist()

    
    
    model_map = {
        # 'yolov8': lambda: YOLOv8ClsFromYAML(
        #     yaml_path='yolov8-cls.yaml',
        #     scale='n',
        #     num_classes=3,
        #     pretrained=args.pretrain_path
        # ),
        'convnext': lambda: ConvNeXtBTXRD(num_classes=3),
        'efficientnetb0': lambda: EfficientNetBTXRD(num_classes=3, dropout_p=args.dropout),
        'efficientnetb4': lambda: EfficientNetB4BTXRD(num_classes=3, dropout_p=args.dropout),
        'resnet50': lambda: ResNet50(num_classes=3, dropout_p=args.dropout)
    }
    
    if args.model_name == 'van':
        model = VAN(
            img_size=args.img_size,
            num_classes=3,
            embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3])
        
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    elif args.model_name == 'yolov11':
        cfg = os.path.join('yolo/cfg','models','11',f'yolo11{args.yolo_scale}-cls.yaml')
        model = ClassificationModel(cfg, nc=3, ch=3)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    elif args.model_name == 'yolov8':
        cfg = os.path.join('yolo/cfg','models','v8',f'yolov8{args.yolo_scale}-cls.yaml')
        model = ClassificationModel(cfg, nc=3, ch=3)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    else:
        model = model_map[args.model_name]()
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    if args.use_balanced_weight:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=[0, 1, 2],
            y=y_labels  # or collect all labels manually
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    criterion = FocalCE(weight=class_weights if args.use_balanced_weight else None, gamma=2.0)

    val_loss, top1_acc, top5_acc, extra_metrics, cm_image = validate(model, test_loader, criterion, device, class_names=CLASS_NAMES)
    
    print(f"Val  Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-2 Acc: {top5_acc:.4f}")
    print(f"Balanced Acc: {extra_metrics['balanced_acc']:.4f} | Macro F1: {extra_metrics['macro_f1']:.4f} | Weighted F1: {extra_metrics['weighted_f1']:.4f}")
    print(f"Kappa: {extra_metrics['kappa']:.4f} | MCC: {extra_metrics['mcc']:.4f}")
    print(f"Per-class F1: {extra_metrics['per_class_f1']} (support={extra_metrics['per_class_support']})")

    row = {
        "val_loss": float(val_loss),
        "top1_acc": float(top1_acc),
        "top2_acc": float(top5_acc),
        "balanced_acc": float(extra_metrics["balanced_acc"]),
        "macro_f1": float(extra_metrics["macro_f1"]),
        "micro_f1": float(extra_metrics["micro_f1"]),
        "weighted_f1": float(extra_metrics["weighted_f1"]),
        "kappa": float(extra_metrics["kappa"]),
        "mcc": float(extra_metrics["mcc"]),
        "roc_auc_ovr_macro": (None if extra_metrics["roc_auc_ovr_macro"] is None else float(extra_metrics["roc_auc_ovr_macro"])),
        "pr_auc_macro": (None if extra_metrics["pr_auc_macro"] is None else float(extra_metrics["pr_auc_macro"])),
    }

    # Expand per-class metrics into columns using CLASS_NAMES (keep order aligned)
    for i, cname in enumerate(CLASS_NAMES):
        row[f"precision_{cname}"] = float(extra_metrics["per_class_precision"][i])
        row[f"recall_{cname}"]    = float(extra_metrics["per_class_recall"][i])
        row[f"f1_{cname}"]        = float(extra_metrics["per_class_f1"][i])
        row[f"support_{cname}"]   = int(extra_metrics["per_class_support"][i])

    save_dir = os.path.join("checkpoints", args.exp_name)
    excel_path = os.path.join(save_dir, "eval_metrics.xlsx")
    append_row_to_excel(excel_path, row, sheet_name="Eval")
    if cm_image is not None:
        cm_path = os.path.join("checkpoints", args.exp_name, f"cm.png")
        cm_image.image.save(cm_path)
        print(f"Confusion matrix saved at {cm_path}")
def validate(model, dataloader, criterion, device, class_names=[]):
    model.eval()
    running_loss = 0.0
    top1_total = 0
    top5_total = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # logits
            outputs = _get_logits(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            top1 = top_k_accuracy(outputs, labels, k=1)
            top5 = top_k_accuracy(outputs, labels, k=2)  # for 3 classes, k=2 is fine as "top-2"
            top1_total += top1
            top5_total += top5

            all_logits.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = probs.argmax(axis=1)

    # Compute imbalance-aware metrics
    metrics, cm = compute_imbalanced_metrics(y_true, y_pred, probs, num_classes=probs.shape[1])

    # Confusion matrix figure
    try:
        fig = plot_confusion_matrix(cm, class_names if class_names is not None else [str(i) for i in range(probs.shape[1])])
        cm_image = wandb.Image(fig)
        plt.close(fig)
    except Exception:
        cm_image = None

    epoch_loss = running_loss / len(dataloader.dataset)
    top1_acc = top1_total / len(dataloader.dataset)
    top5_acc = top5_total / len(dataloader.dataset)

    # Return everything
    return epoch_loss, top1_acc, top5_acc, metrics, cm_image


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='BTXRD Classification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--scenario', default='A', type=str,help='A=no clahe, B=clahe as weak aug, C=clahe as preprocessing')
    
    parser.add_argument('--img_size', type=int, default=608, metavar='img_size',
                        help='Size of input image)')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp/best_model.pth', metavar='N',
                        help='model_path')
    parser.add_argument('--model_name', type=str, default='convnext', metavar='N',
                        help='Name of the model')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--yolo_scale', default='n', choices=['n','s','m','l','x'])
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='LR',
                        help='Dropout')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_worker', type=int, default=4, metavar='S',
                        help='Num of Worker')
    parser.add_argument('--use_balanced_weight', action='store_true', default=False, help='Use Weight Balancing')

    # Wavelet toggles
    parser.add_argument('--use_wavelet', action='store_true')
    parser.add_argument('--wavelet_name', type=str, default='db2')
    parser.add_argument('--wavelet_level', type=int, default=2)
    parser.add_argument('--wavelet_p', type=float, default=1.0)  # 1.0 => deterministic preprocessing

    # Unsharp toggles
    parser.add_argument('--use_unsharp', action='store_true')
    parser.add_argument('--unsharp_amount', type=float, default=0.7)
    parser.add_argument('--unsharp_radius', type=float, default=1.0)
    parser.add_argument('--unsharp_threshold', type=int, default=2)
    parser.add_argument('--unsharp_p', type=float, default=1.0)

    # Structure map (optional)
    parser.add_argument('--use_structuremap', action='store_true')

    args = parser.parse_args()

    main(args)
