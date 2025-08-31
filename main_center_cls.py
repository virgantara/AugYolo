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
from torch.utils.data import ConcatDataset

def main(args):
    set_seed(args)
    print("Hyper-parameters: {}".format(args.__str__()))

    num_epochs = args.epochs

    center_id = args.center_id
    
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    train_path = os.path.join(DATASET_DIR, 'train.xlsx')
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    
    train_transform, test_transform = build_transforms(args)

    
    train_dataset_c1 = BoneTumorDatasetCenter(
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        center_id=1,   # Center 1
        transform=train_transform
    )
    train_dataset_c2 = BoneTumorDatasetCenter(
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        center_id=2,   # Center 2
        transform=train_transform
    )
    
    train_dataset = ConcatDataset([train_dataset_c1, train_dataset_c2])

    # Test dataset = Center 3
    test_dataset = BoneTumorDatasetCenter(
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        center_id=3,   # Center 3
        transform=test_transform   # use test transform
    )

    if getattr(args, "use_balanced_weight", False):
        y_train = np.concatenate([
            train_dataset_c1.df["label"].to_numpy(),
            train_dataset_c2.df["label"].to_numpy()
        ])
        classes_all = np.unique(y_train)
        cw = compute_class_weight(class_weight="balanced", classes=classes_all, y=y_train)
        class_weights_np = cw
    else:
        class_weights_np = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_worker, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = args.lr if not args.use_sgd else args.lr

    # ---------- MULTI-SEED TRAIN + EVAL ----------
    seed_list = args.seeds if isinstance(args.seeds, list) else [args.seeds]
    per_seed_best_top1 = []

    for seed in seed_list:
        set_seed(argparse.Namespace(seed=seed))  # reuse your set_seed

        # fresh model / opt / scheduler per seed
        model = get_model(args).to(device)

        # (Paper policy) freeze backbone, finetune head ONLY — for your YOLOv8Cls model
        freeze_all_but_head(model)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                      eta_min=max(args.lr * 1e-2, 1e-6))

        class_weights = None
        if class_weights_np is not None:
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

        criterion = FocalCE(weight=class_weights, gamma=2.0)

        run = wandb.init(
            project=getattr(args, "project_name", "BTXRD Seeds"),
            name=f"{args.exp_name}_Seeds_{seed}",
            config={
                "exp_name": args.exp_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": getattr(args, "img_size", None),
                "seed": seed,
                "lr": lr,
                "use_balanced_weight": getattr(args, "use_balanced_weight", False),
                "train_centers": "1+2",
                "test_center": "3",
            },
            reinit=True
        )

        best_top1 = -1e9
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, top1_acc, top5_acc = validate(model, test_loader, criterion, device)
            scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"],
            })

            if top1_acc > best_top1:
                best_top1 = top1_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                save_dir = os.path.join("checkpoints", args.exp_name, f"seed{seed}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
                print(f"Best model saved at Seed: {seed}, epoch: {epoch+1} with Top-1 Acc: {top1_acc:.4f}")

        wandb.summary["best_top1_accuracy"] = float(best_top1)
        per_seed_best_top1.append(float(best_top1))
        run.finish()

    # ---------- SUMMARY & (optional) T-TEST ----------
    mean_top1 = float(np.mean(per_seed_best_top1))
    std_top1  = float(np.std(per_seed_best_top1))
    print(f"\n=== Seeds Summary (Center1+2 → Center3) ===")
    print(f"Top-1 Accuracy: {mean_top1:.4f} ± {std_top1:.4f}")
    print(f"Per-seed best top1: {per_seed_best_top1}")

    wandb_run = wandb.init(project=getattr(args, "project_name", "BTXRD Seeds"),
                           name=f"{args.exp_name}_summary", reinit=True)
    wandb.summary["seeds"] = seed_list
    wandb.summary["mean_top1"] = mean_top1
    wandb.summary["std_top1"]  = std_top1
    wandb.summary["per_seed"]  = per_seed_best_top1

    # Optional: compare to a baseline list and report Welch’s t-test
    if args.compare_scores_file:
        try:
            baseline = np.loadtxt(args.compare_scores_file, dtype=float, ndmin=1)
            t_stat, p_val = welch_ttest(per_seed_best_top1, baseline.tolist())
            print(f"Welch t-test vs baseline: t={t_stat:.3f}, p={p_val:.4g}")
            wandb.summary["ttest_t"] = float(t_stat)
            wandb.summary["ttest_p"] = float(p_val)
        except Exception as e:
            print(f"[WARN] Could not load baseline scores from {args.compare_scores_file}: {e}")

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

def freeze_all_but_head(model):
    """
    Freeze all params, then unfreeze the classification head.
    For YOLOv8-Cls wrappers, try common head names; if not found,
    unfreeze the last nn.Linear layer found.
    """
    for p in model.parameters():
        p.requires_grad = False

    head_names = ['classifier', 'fc', 'head', 'cls', 'heads']
    unfroze = 0
    for name in head_names:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
            unfroze += 1

    if unfroze == 0:
        # fallback: last Linear module
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True

def welch_ttest(group_a, group_b):
    """Independent two-sample Welch’s t-test (no equal variances assumption)."""
    import math
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    t = (ma - mb) / math.sqrt(va/na + vb/nb)
    df = (va/na + vb/nb)**2 / ((va*va)/((na**2)*(na-1)) + (vb*vb)/((nb**2)*(nb-1)))
    # two-sided p-value from Student-t CDF (scipy-free); fall back to scipy if available
    try:
        from mpmath import quad, beta
        # Not strict; keep simple if mpmath unavailable
        raise ImportError
    except Exception:
        from scipy.stats import t as student_t
        p = 2 * (1 - student_t.cdf(abs(t), df))
    return float(t), float(p)


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
    parser.add_argument('--project_name', type=str, default='BTXRD Seeds', metavar='N',
                        help='Name of the Project WANDB')
    parser.add_argument('--center_id', type=int, default=1,choices=[1,2,3],
                        help='random seed (default: 1)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])  # for t-test
    parser.add_argument('--compare_scores_file', type=str, default='',
                    help='Optional: path to a text file containing baseline per-seed scores (one number per line) for Welch’s t-test')


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
