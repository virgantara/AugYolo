import torch
from torch.utils.data import random_split, DataLoader
from dataset import BoneTumorDataset
import os
from torchvision import transforms
import argparse
from util import top_k_accuracy
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import (
    YOLOv8ClsFromYAML, 
    ConvNeXtBTXRD, 
    EfficientNetBTXRD,
    EfficientNetB4BTXRD
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

    num_epochs = args.epochs

    
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    
    test_transform = transforms.Compose([
        transforms.Resize((608, 608)),  # or (384, 384)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

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
    
    y_labels = train_dataset.df['label'].tolist()

    
    
    if args.model_name == 'yolov8':
        model = YOLOv8ClsFromYAML(
            yaml_path='yolov8-cls.yaml',
            scale='n',
            num_classes=3,
            pretrained=args.pretrain_path
        )

    elif args.model_name == 'convnext':
        model = ConvNeXtBTXRD(num_classes=3)

    elif args.model_name == 'efficientnetb0':
        model = EfficientNetBTXRD(num_classes=3, dropout_p=args.dropout)
    elif args.model_name == 'efficientnetb4':
        model = EfficientNetB4BTXRD(num_classes=3, dropout_p=args.dropout)
    
    model.load_state_dict(torch.load(args.model_path), weights_only=True)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    val_loss, top1_acc, top5_acc, extra_metrics, cm_image = validate(model, test_loader, criterion, device, class_names=CLASS_NAMES)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val  Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-2 Acc: {top5_acc:.4f}")
    print(f"Balanced Acc: {extra_metrics['balanced_acc']:.4f} | Macro F1: {extra_metrics['macro_f1']:.4f} | Weighted F1: {extra_metrics['weighted_f1']:.4f}")
    print(f"Kappa: {extra_metrics['kappa']:.4f} | MCC: {extra_metrics['mcc']:.4f}")
    print(f"Per-class F1: {extra_metrics['per_class_f1']} (support={extra_metrics['per_class_support']})")


def validate(model, dataloader, criterion, device, class_names=CLASS_NAMES):
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
    parser.add_argument('--model_path', type=str, default='checkpoints/exp/best_model.pth', metavar='N',
                        help='model_path')
    parser.add_argument('--model_name', type=str, default='convnext', metavar='N',
                        help='Name of the model')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_worker', type=int, default=4, metavar='S',
                        help='Num of Worker')
    args = parser.parse_args()

    main(args)
