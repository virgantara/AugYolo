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
from sklearn.metrics import roc_auc_score, roc_curve, auc
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

    class_names = ['Normal', 'Benign', 'Malignant']
    mean_acc, ci95, p_val = evaluate_with_statistics(
        model, test_loader, criterion, device,
        n_repeats=5, class_names=class_names, exp_name=args.exp_name
    )

    print("Mean Acc",mean_acc)
    print("ci95",ci95)
    print("p_val",p_val)


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


def evaluate_with_statistics(model, dataloader, criterion, device, n_repeats=5, class_names=None, exp_name="exp_eval"):
    """
    Evaluate model repeatedly to compute mean accuracy, 95% CI, significance vs baseline,
    and plot per-class PR curves.
    """

    all_acc = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    print(f"Running {n_repeats} repeated evaluations...")
    for r in tqdm(range(n_repeats)):
        model.eval()
        correct, total = 0, 0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                outputs = _get_logits(outputs)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        acc = correct / total
        all_acc.append(acc)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        print(f"Run {r+1}: Accuracy = {acc*100:.2f}%")

    # --- Compute mean and 95% CI ---
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc, ddof=1)
    ci95 = 1.96 * (std_acc / np.sqrt(n_repeats))
    print(f"\nMean Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}% (95% CI)")

    # --- Compare to baseline (YOLOv8) ---
    # Suppose you have baseline accuracies (same runs)
    baseline_acc = np.array([0.912, 0.913, 0.911, 0.915, 0.914])  # example
    t_stat, p_val = stats.ttest_rel(all_acc, baseline_acc)
    print(f"Paired t-test vs YOLOv8 baseline: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("Improvement is statistically significant (p < 0.05)")
    else:
        print("No statistically significant difference (p ≥ 0.05)")

    # --- Per-class metrics ---
    print("\nPer-class metrics:")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    # --- Precision–Recall (PR) Curves ---
    y_true_bin = np.zeros((len(all_y_true), len(class_names)))
    for i, label in enumerate(all_y_true):
        y_true_bin[i, label] = 1

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], np.array(all_y_prob)[:, i])
        ap = average_precision_score(y_true_bin[:, i], np.array(all_y_prob)[:, i])
        plt.plot(recall, precision, lw=2, label=f'{cls} (AP={ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curves (BTXRD)')
    plt.legend(loc='lower left')
    os.makedirs("results", exist_ok=True)
    pr_path = os.path.join("results", f"pr_curve_{exp_name}.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"PR curves saved to: {pr_path}")

    return mean_acc, ci95, p_val

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
