import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary

from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from CustomDataset.datasets import BoneTumorDataset
import torchattacks
from torchattacks import PGD, FGSM
from MedViT import MedViT_small, MedViT_base, MedViT_large
import wandb
from torchvision.transforms.transforms import Resize
from torchvision.transforms import InterpolationMode

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
print("Medmnist", medmnist.__version__)

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

parser.add_argument('--use_center_dataset_split', action='store_true', default=False, help='Use Center 1,2 as train, 3 as test')

# Structure map (optional)
parser.add_argument('--use_structuremap', action='store_true')
args = parser.parse_args()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)

_init_()

# data_flag = 'retinamnist'
# [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
# pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
# download = True

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
lr = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project=args.project_name, 
    name=f"{args.exp_name}",
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": args.model_name,
        "optimizer": "adam"
    }
)

wandb_log = {} 
# info = INFO[data_flag]
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])

n_classes = 3
# DataClass = getattr(medmnist, info['python_class'])

# print("number of channels : ", n_channels)
# print("number of classes : ", n_classes)

# preprocessing
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.Lambda(lambda image: image.convert('RGB')),
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

DATASET_DIR = '../btxrd/data/BTXRD'
metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
train_path = os.path.join(DATASET_DIR, 'train.xlsx')
test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
IMG_DIR = os.path.join(DATASET_DIR, 'images')

# load the data
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

# pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)


model = MedViT_small(num_classes = n_classes).to(device)
#model = MedViT_base(num_classes = n_classes).cuda()
#model = MedViT_large(num_classes = n_classes).cuda()
wandb.watch(model)

task = ''
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.squeeze().long().to(device) if task != "multi-label, binary-class" else y.to(device).to(torch.float32)
        logits = model(x)
        if criterion is not None:
            running_loss += criterion(logits, y).item() * x.size(0)
        if task == "multi-label, binary-class":
            # Convert logits->probs->preds for multi-label if needed; here we keep simple single-label case
            preds = (logits.sigmoid() > 0.5).long()
            # Adjust accuracy computation for multi-label if you actually use it
            correct += (preds == y.long()).all(dim=1).sum().item()
            total += x.size(0)
        else:
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = running_loss / max(total, 1) if criterion is not None else None
    acc = correct / max(total, 1)
    return acc, avg_loss

best_acc = -1.0
best_epoch = -1

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    print('Epoch [%d/%d]'% (epoch+1, NUM_EPOCHS))
    model.train()

    running_loss = 0.0
    seen = 0
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device)
        if task == 'multi-label, binary-class':
            targets = targets.to(device).to(torch.float32)
        else:
            targets = targets.to(device).squeeze().long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    train_loss = running_loss / max(seen, 1)

    val_acc, val_loss = evaluate(model, test_loader, criterion)
    print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

    wandb_log = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "top1_accuracy": val_acc,
        # "top5_accuracy": 0,
        # "epochs_since_improve": epochs_since_improve,
        # "best_top1_accuracy_so_far": best_top1_acc
    }
    wandb.log(wandb_log)
    # ---- SAVE BEST CHECKPOINT ----
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        best_model_state = model.state_dict()
        save_dir = os.path.join("checkpoints",args.exp_name)
        torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
        # torch.save({
        #     "epoch": best_epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "best_acc": best_acc,
        #     "val_loss": val_loss,
        #     "num_classes": n_classes,
        #     "arch": "MedViT_small"
        # }, ckpt_path)
        print(f"Saved new best model to {save_dir} (val_acc={best_acc*100:.2f}%)")

wandb.finish()
last_model_state = model.state_dict()
save_dir = os.path.join("checkpoints",args.exp_name)
torch.save(last_model_state, os.path.join(save_dir, 'last_model.pth'))
print(f"Saved final model to {final_path} | best@epoch {best_epoch} = {best_acc*100:.2f}%")


split = 'test'

model.eval()
y_true = torch.tensor([])
y_score = torch.tensor([])

data_loader = train_loader_at_eval if split == 'train' else test_loader

with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = outputs.softmax(dim=-1)
        y_score = torch.cat((y_score, outputs.cpu()), 0)

    y_score = y_score.detach().numpy()

    evaluator = Evaluator(data_flag, split, size=224)
    metrics = evaluator.evaluate(y_score)

    print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))

BATCH_SIZE = 5
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)    
model.eval()

correct = 0
total = 0

atk = FGSM(model, eps=0.01)

for images, labels in test_loader:
    labels = labels.squeeze(1)
    images = atk(images, labels).cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('FGSM Robust accuracy: %.2f %%' % (100 * float(correct) / total))


model.eval()

correct = 0
total = 0

atk = PGD(model, eps=8/255, alpha=4/255, steps=10, random_start=True)

for images, labels in test_loader:
    labels = labels.squeeze(1)
    images = atk(images, labels).cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('PGD Robust accuracy: %.2f %%' % (100 * float(correct) / total))