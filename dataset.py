import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

def encode_label(row):
    if row['tumor'] == 0:
        return 0  # normal
    elif row['benign'] == 1:
        return 1  # benign
    elif row['malignant'] == 1:
        return 2  # malignant
    else:
        return -1  # unknown


class BoneTumorDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_excel(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Generate label column
        self.data['label'] = self.data.apply(encode_label, axis=1)

        # Filter unknown labels
        self.data = self.data[self.data['label'] != -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
