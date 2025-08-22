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
from sklearn.model_selection import StratifiedKFold
from models import YOLOv8nCls

def main(args):
    print("Hyper-parameters: {}".format(args.__str__()))

    num_epochs = args.epochs

    k_folds = 5
    
    
    BATCH_SIZE = 32

    DATASET_DIR = 'data/BTXRD'
    file_path = os.path.join(DATASET_DIR, 'dataset.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')
    CSV_FILE = file_path

    transform = transforms.Compose([
        transforms.Resize((600, 600)),  # or (384, 384)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = BoneTumorDataset(csv_path=CSV_FILE, image_dir=IMG_DIR, transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    labels = full_dataset.data['label'].values

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(full_dataset.data, labels)):
        wandb.init(
            project=args.project_name, 
            name=f"{args.exp_name}_fold{fold+1}",
            config={
                "epochs": num_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "architecture": args.model_name,
                "optimizer": "adam",
                "fold": fold + 1
            }
        )

        wandb_log = {}  

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.test_batch_size, shuffle=False)

        # Model definition
        model = YOLOv8nCls(num_classes=3)
        
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

        wandb.watch(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = (optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
                     if args.use_sgd else
                     optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4))
        
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)

        best_top1_acc = 0.0
        best_model_state = None
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, top1_acc, top5_acc = validate(model, test_loader, criterion, device)
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
                torch.save(best_model_state, f'best_model_fold{fold+1}.pth')
                print(f"Best model saved at epoch {epoch+1} with Top-1 Acc: {top1_acc:.4f}")

        wandb.finish()


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler):
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

    scheduler.step()

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
            top5 = top_k_accuracy(outputs, labels, k=5)

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

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='BTXRD Classification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default='resnet18', metavar='N',
                        help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--project_name', type=str, default='BTXRD', metavar='N',
                        help='Name of the Project WANDB')
    parser.add_argument('--model_path', type=str, default='pretrained/GDANet_ModelNet40_93.4.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()
    main(args)
