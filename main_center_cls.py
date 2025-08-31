import torch
from torch.utils.data import random_split, DataLoader
from dataset import BoneTumorDatasetCenter
import os
from torchvision import transforms
import argparse
from util import top_k_accuracy, CLAHE, FocalCE
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
    EfficientNetB4BTXRD,
    ResNet50
)

from models_yolo import (ClassificationModel)
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from van import VAN, load_model_weights
from timm.models.vision_transformer import _cfg
from functools import partial
from transforms_factory import build_transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

def main(args):
    set_seed(args)
    print("Hyper-parameters: {}".format(args.__str__()))

    num_epochs = args.epochs

    
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    train_path = os.path.join(DATASET_DIR, 'train.xlsx')
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    
    train_transform, test_transform = build_transforms(args)

    
    full_dataset = BoneTumorDatasetCenter(
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        center_id=args.center_id,
        transform=train_transform
    )
    val_view = BoneTumorDatasetCenter(
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        center_id=args.center_id,
        transform=test_transform
    )
    assert len(full_dataset) == len(val_view)

    labels = full_dataset.df["label"].to_numpy()
    classes_all = np.unique(labels)

    # Make folds
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lr = args.lr if not args.use_sgd else args.lr  # Don't multiply

    all_fold_metrics = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"Fold {fold_id}: train={len(tr_idx)}, val={len(va_idx)}")

        # subsets
        train_subset = Subset(full_dataset, tr_idx)
        val_subset   = Subset(val_view, va_idx)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_worker, pin_memory=True
        )
        val_loader   = DataLoader(
            val_subset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.num_worker, pin_memory=True
        )

        # fresh model/opt/scheduler per fold
        model = get_model(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr*1e-2, 1e-6))

        if getattr(args, "use_balanced_weight", False):
            y_train = labels[tr_idx]
            cw = compute_class_weight(class_weight="balanced", classes=classes_all, y=y_train)
            class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
        else:
            class_weights = None
        criterion = FocalCE(weight=class_weights, gamma=2.0)

        best_top1_acc = -float('inf')
        best_model_state = None

        # wandb run per fold (optional: set reinit=True if using multiple runs)
        run = wandb.init(
            project=getattr(args, "project_name", "BoneTumor-CV"),
            name=f"{args.exp_name}_center{args.center_id}_fold{fold_id}",
            config={
                "exp_name": args.exp_name,
                "center": args.center_id,
                "fold": fold_id,
                "epochs": num_epochs,
                "batch_size": args.batch_size,
                "img_size": getattr(args, "img_size", None),
                "seed": getattr(args, "seed", None),
                "lr": lr,
                "use_balanced_weight": getattr(args, "use_balanced_weight", False),
            },
            reinit=True
        )

        print(f"\n=== Center {center_id} | Fold {fold_id} ({len(tr_idx)} train / {len(va_idx)} val) ===")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, top1_acc, top5_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

            wandb.log({
                "fold": fold_id,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"],
            })

            # save best for this fold
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                save_dir = os.path.join("checkpoints", args.exp_name, f"center{center_id}", f"fold{fold_id}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_model_state, os.path.join(save_dir, "best_model.pth"))
                print(f" [Center {center_id} Fold {fold_id}] Best model saved @ epoch {epoch+1} (Top-1: {top1_acc:.4f})")

        # end of fold — log summary & store metrics
        wandb.summary["best_top1_accuracy"] = float(best_top1_acc)
        all_fold_metrics.append(best_top1_acc)
        run.finish()

    # After all folds — aggregate
    mean_top1 = float(np.mean(all_fold_metrics)) if all_fold_metrics else 0.0
    std_top1  = float(np.std(all_fold_metrics)) if all_fold_metrics else 0.0
    print(f"\n=== Center {center_id} K-Fold Summary ===")
    print(f"Top-1 Accuracy: {mean_top1:.4f} ± {std_top1:.4f}")

    # (Optional) log an overall summary run or to the last run's summary
    wandb.run = wandb.init(
        project="BoneTumor-CV",
        name=f"{args.exp_name}_center{center_id}_summary",
        reinit=True
    )
    wandb.summary["center"] = center_id
    wandb.summary["kfold_mean_top1"] = mean_top1
    wandb.summary["kfold_std_top1"] = std_top1
    wandb.finish()


def get_model(args):
    cfg = os.path.join('yolo/cfg','models','v8',f'yolov8{args.yolo_scale}-cls.yaml')
    model = ClassificationModel(cfg, nc=3, ch=3)
    
    pretrain_path = os.path.join('pretrain','yolov8n-cls.pt')
    weights = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if weights:
        model.load(weights)

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and args.dropout:
            m.p = args.dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # for training

    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = _get_logits(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    top1_total = 0
    top5_total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = _get_logits(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            top1 = top_k_accuracy(outputs, labels, k=1)
            top5 = top_k_accuracy(outputs, labels, k=2)

            top1_total += top1
            top5_total += top5

    epoch_loss = running_loss / len(dataloader.dataset)
    top1_acc = top1_total / len(dataloader.dataset)
    top5_acc = top5_total / len(dataloader.dataset)

    return epoch_loss, top1_acc, top5_acc

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

def _init_():


    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)

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
    parser.add_argument('--pretrain_path', type=str, default='pretrain/yolov8n-cls.pt', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--path_yolo_yaml', type=str, default='yolo/cfg/models/11/yolo11-cls-lka.yaml', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default='convnext', metavar='N',
                        help='Name of the model')
    parser.add_argument('--yolo_scale', default='n', choices=['n','s','m','l','x'])
    parser.add_argument('--img_size', type=int, default=608, metavar='img_size',
                        help='Size of input image)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
    parser.add_argument('--scenario', default='A', type=str,help='A=no clahe, B=clahe as weak aug, C=clahe as preprocessing')
    parser.add_argument('--use_balanced_weight', action='store_true', default=False, help='Use Weight Balancing')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='LR',
                        help='Dropout')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_worker', type=int, default=4, metavar='S',
                        help='Num of Worker')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--project_name', type=str, default='BTXRD KFold', metavar='N',
                        help='Name of the Project WANDB')
    parser.add_argument('--center_id', type=int, default=1,choices=[1,2,3],
                        help='random seed (default: 1)')

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

    # Structure map (optional)
    parser.add_argument('--use_structuremap', action='store_true')
    args = parser.parse_args()

    _init_()
    main(args)
