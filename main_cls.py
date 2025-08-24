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
    EfficientNetB4BTXRD,
    ResNet50
)
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from van import VAN, load_model_weights
from timm.models.vision_transformer import _cfg

def main(args):
    set_seed(args)
    print("Hyper-parameters: {}".format(args.__str__()))

    num_epochs = args.epochs

    
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
    train_path = os.path.join(DATASET_DIR, 'train.xlsx')
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    
    train_transform = transforms.Compose([
        transforms.Resize((608, 608)),  # or (384, 384)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((608, 608)),  # or (384, 384)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

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
    
    y_labels = train_dataset.df['label'].tolist()

    

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
        'yolov8': lambda: YOLOv8ClsFromYAML(
            yaml_path='yolov8-cls.yaml',
            scale='n',
            num_classes=3,
            pretrained=args.pretrain_path
        ),
        'convnext': lambda: ConvNeXtBTXRD(num_classes=3),
        'efficientnetb0': lambda: EfficientNetBTXRD(num_classes=3, dropout_p=args.dropout),
        'efficientnetb4': lambda: EfficientNetB4BTXRD(num_classes=3, dropout_p=args.dropout),
        'resnet50': lambda: ResNet50(num_classes=3, dropout_p=args.dropout)
    }

    if args.model_name == 'van':
        model = VAN(
            img_size=608,
            num_classes=3
        )
        
        model.default_cfg = _cfg()
        url = 'https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )

        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        
    else:
        model = model_map[args.model_name]()
        
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    wandb.watch(model)

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
        
    lr = args.lr if not args.use_sgd else args.lr  # Don't multiply
    optimizer = (optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=1e-4)
             if args.use_sgd else
             optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4))

    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)

    best_top1_acc = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, top1_acc, top5_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
        }
        wandb.log(wandb_log)

        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            best_model_state = model.state_dict()
            save_dir = os.path.join("checkpoints",args.exp_name)
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with Top-1 Acc: {top1_acc:.4f}")

    wandb.finish()

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
    parser.add_argument('--model_name', type=str, default='convnext', metavar='N',
                        help='Name of the model')
    parser.add_argument('--img_size', type=int, default=608, metavar='img_size',
                        help='Size of input image)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
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
    args = parser.parse_args()

    _init_()
    main(args)
