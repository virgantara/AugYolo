import torch
from torch.utils.data import random_split, DataLoader
from dataset import BoneTumorDataset, BoneTumorDatasetCenter
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

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models_yolo import (ClassificationModel)
from model_zoo.medvit.MedViT import MedViT
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from van import VAN, load_model_weights
from timm.models.vision_transformer import _cfg
from functools import partial
from transforms_factory import build_transforms
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset

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

    if args.use_center_dataset_split:
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
    else:
        train_dataset = BoneTumorDataset(
            split_xlsx_path=train_path,
            metadata_xlsx_path=metadata_xlsx_path,
            image_dir=IMG_DIR,  # make sure this exists
            transform=train_transform
        )

        test_dataset = BoneTumorDataset(
            split_xlsx_path=test_path,
            metadata_xlsx_path=metadata_xlsx_path,
            image_dir=IMG_DIR,
            transform=test_transform
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_worker, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                          num_workers=args.num_worker, pin_memory=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # y_labels = train_dataset.df['label'].tolist()

    

    wandb.init(
        project=args.project_name, 
        name=f"{args.exp_name}",
        config={
            "epochs": num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "architecture": args.model_name,
            "optimizer": "adam"
        }
    )

    wandb_log = {}  

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
        
        
        model.default_cfg = _cfg()
        url = 'https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )

        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    elif args.model_name == 'yolov11':
        cfg = os.path.join('yolo/cfg','models','11',f'yolo11{args.yolo_scale}-cls.yaml')
        model = ClassificationModel(cfg, nc=3, ch=3)
        
        pretrain_path = os.path.join('pretrain','yolo11n-cls.pt')
        weights = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if weights:
            model.load(weights)

        for m in model.modules():
            if isinstance(m, torch.nn.Dropout) and args.dropout:
                m.p = args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
    elif args.model_name == 'yolov11lka':
        cfg = os.path.join(args.path_yolo_yaml)
        model = ClassificationModel(cfg, nc=3, ch=3)
        
        pretrain_path = os.path.join('pretrain','yolo11n-cls.pt')
        weights = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if weights:
            model.load(weights)
    elif args.model_name == 'yolov8':
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
    elif args.model_name == 'yolov8doconv':
        cfg = os.path.join('yolo/cfg','models','v8',f'yolov8{args.yolo_scale}-cls-doconv.yaml')
        model = ClassificationModel(cfg, nc=3, ch=3)
        
        pretrain_path = os.path.join('pretrain','yolov8n-cls.pt')
        weights = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if weights:
            model.load(weights)

        for m in model.modules():
            if isinstance(m, torch.nn.Dropout) and args.dropout:
                m.p = args.dropout  # set dropout
        # for p in model.parameters():
        #     p.requires_grad = True  # for training
    elif args.model_name == 'medvit':
        model = MedViT(
            num_classes=3,
            stem_chs=[64, 32, 64], 
            depths=[3, 4, 10, 3], 
            path_dropout=0.1
        )
    else:
        model = model_map[args.model_name]()
        
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    wandb.watch(model)

    # if args.use_balanced_weight:
    #     class_weights = compute_class_weight(
    #         class_weight='balanced',
    #         classes=[0, 1, 2],
    #         y=y_labels  # or collect all labels manually
    #     )
    #     class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    # else:
    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    criterion = FocalCE(weight=class_weights if args.use_balanced_weight else None, gamma=2.0)

    lr = args.lr if not args.use_sgd else args.lr  # Don't multiply

    if args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr*1e-2, 1e-6))

    best_top1_acc = 0.0
    best_model_state = None

    patience = args.early_stop_patience
    epochs_since_improve = 0
    best_epoch = -1
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, top1_acc, top5_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

        
        improved = top1_acc > best_top1_acc

        if improved:
            best_top1_acc = top1_acc
            best_model_state = model.state_dict()
            save_dir = os.path.join("checkpoints",args.exp_name)
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with Top-1 Acc: {top1_acc:.4f}")

            best_epoch = epoch + 1
            epochs_since_improve = 0
        # else:
        #     epochs_since_improve += 1
        #     print(f"No improvement for {epochs_since_improve}/{patience} evals.")

        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "epochs_since_improve": epochs_since_improve,
            "best_top1_accuracy_so_far": best_top1_acc
        }
        wandb.log(wandb_log)

        # if epochs_since_improve >= patience:
        #     print(f"\nEarly stopping at epoch {epoch+1}. "
        #           f"Best Top-1 Acc: {best_top1_acc:.4f} (epoch {best_epoch}).")
        #     break
    wandb.finish()



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
    parser.add_argument('--project_name', type=str, default='BTXRD', metavar='N',
                        help='Name of the Project WANDB')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='Stop if no Top-1 improvement for this many evals')

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

    _init_()
    main(args)
