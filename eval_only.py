import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import BoneTumorDataset, BoneTumorDatasetCenter
import os
from torchvision import transforms
import argparse
from util import top_k_accuracy, FocalCE
from tqdm import tqdm
import wandb
from models import (
    YOLOv8ClsFromYAML, 
    ConvNeXtBTXRD, 
    EfficientNetBTXRD,
    EfficientNetB4BTXRD,
    ResNet50
)
from models_yolo import ClassificationModel
from model_zoo.medvit.MedViT import MedViT
from model_zoo.swin.model import SwinTransformer
from model_zoo.swin.modelv2 import SwinTransformerV2
from model_zoo.van.van import VAN
from timm.models.vision_transformer import _cfg
from functools import partial
from transforms_factory import build_transforms
import numpy as np
import random
import sys
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    auc,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main(args):
    set_seed(args)
    print("Running evaluation only...")
    print(f"Model: {args.model_name}, Experiment: {args.exp_name}")

    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')
    IMG_DIR = os.path.join(DATASET_DIR, 'images')

    _, test_transform = build_transforms(args)

    # --- Dataset ---
    if args.use_center_dataset_split:
        test_dataset = BoneTumorDatasetCenter(
            metadata_xlsx_path=metadata_xlsx_path,
            image_dir=IMG_DIR,
            center_id=3,  # evaluation only on Center 3
            transform=test_transform
        )
    else:
        test_dataset = BoneTumorDataset(
            split_xlsx_path=test_path,
            metadata_xlsx_path=metadata_xlsx_path,
            image_dir=IMG_DIR,
            transform=test_transform
        )

    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.num_worker, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model ---
    model = build_model(args)
    model = model.to(device)

    ckpt_path = os.path.join("checkpoints", args.exp_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = FocalCE(weight=None, gamma=2.0)

    # --- Evaluate ---
    metrics = validate(model, test_loader, criterion, device)

    print("\nEvaluation results:")
    for k, v in metrics.items():
        print(f"{k.capitalize():<15}: {v:.4f}")

    print(f"\nEvaluation results:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Top-1 Acc: {top1_acc:.4f}")
    print(f"Top-5 Acc: {top5_acc:.4f}")


def build_model(args):
    model_map = {
        'swin2': lambda: SwinTransformerV2(img_size=args.img_size, num_classes=3, patch_size=args.patch_size, window_size=args.window_size),
        'swin': lambda: SwinTransformer(img_size=args.img_size, num_classes=3, patch_size=args.patch_size, window_size=args.window_size),
        'convnext': lambda: ConvNeXtBTXRD(num_classes=3),
        'efficientnetb0': lambda: EfficientNetBTXRD(num_classes=3, dropout_p=args.dropout),
        'efficientnetb4': lambda: EfficientNetB4BTXRD(num_classes=3, dropout_p=args.dropout),
        'resnet50': lambda: ResNet50(num_classes=3, dropout_p=args.dropout),
        'medvit': lambda: MedViT(num_classes=3, stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1),
    }

    if args.model_name == 'van':
        model = VAN(
            img_size=args.img_size,
            num_classes=3,
            embed_dims=[64, 128, 320, 512],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 27, 3]
        )
        model.default_cfg = _cfg()
        print("Loaded VAN model (pretrained weights skipped for eval-only)")
    elif args.model_name == 'yolo':
        model = ClassificationModel(args.path_yolo_yaml, nc=3, ch=3)
        weights = torch.load(args.pretrain_path, map_location="cpu", weights_only=False)
        if weights:
            model.load_state_dict(weights, strict=False)
    else:
        model = model_map[args.model_name]()
    return model


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_total = 0
    top5_total = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = _get_logits(outputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            top1_total += top_k_accuracy(outputs, labels, k=1)
            top5_total += top_k_accuracy(outputs, labels, k=2)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Compute loss/accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    top1_acc = top1_total / len(dataloader.dataset)
    top5_acc = top5_total / len(dataloader.dataset)

    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nPer-class metrics:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    print(f"\nMacro Precision: {macro_prec:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

    # --- Confusion Matrix (optional) ---
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    n_classes = all_probs.shape[1]
    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    # per-class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro- and macro-average AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro average (average of AUCs)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # --- Plot ROC Curve ---
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Normal', 'Benign', 'Malignant']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=2,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', linewidth=2,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right", frameon=False)
    os.makedirs("results", exist_ok=True)
    roc_path = os.path.join("results", f"roc_curve_{args.exp_name}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    return {
        "loss": epoch_loss,
        "top1": top1_acc,
        "top5": top5_acc,
        "precision": macro_prec,
        "recall": macro_recall,
        "f1": macro_f1,
        "roc_auc_macro": roc_auc["macro"],
        "roc_auc_micro": roc_auc["micro"]
    }


def _get_logits(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    if isinstance(outputs, dict):
        for k in ('logits', 'out', 'pred', 'cls'):
            if k in outputs:
                return outputs[k]
        return next(iter(outputs.values()))
    return outputs


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTXRD Evaluation Only')
    parser.add_argument('--exp_name', type=str, default='exp_eval', help='Experiment name')
    parser.add_argument('--pretrain_path', type=str, default='pretrain/yolov8n-cls.pt', help='Path to pretrained weights')
    parser.add_argument('--path_yolo_yaml', type=str, default='yolo/cfg/models/11/yolo11-cls-lka.yaml')
    parser.add_argument('--model_name', type=str, default='convnext')
    parser.add_argument('--scenario', type=str, default='G')
    parser.add_argument('--img_size', type=int, default=608)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_clahe', action='store_true')

    parser.add_argument('--clahe_p', type=float, default=0.25)

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

    parser.add_argument('--use_center_dataset_split', action='store_true', default=False, help='Use Center 1,2 as train, 3 as test')

    # Structure map (optional)
    parser.add_argument('--use_structuremap', action='store_true')
    args = parser.parse_args()

    main(args)
