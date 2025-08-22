import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

def encode_label(row):
    """
    Encode labels based on 'tumor', 'benign', and 'malignant' values.
    - normal: tumor == 0
    - benign: benign == 1
    - malignant: malignant == 1
    """
    if row['tumor'] == 0:
        return 0  # normal
    elif row['benign'] == 1:
        return 1  # benign
    elif row['malignant'] == 1:
        return 2  # malignant
    else:
        return -1  # unknown or corrupt

class BoneTumorDataset(Dataset):
    def __init__(self, split_xlsx_path, metadata_xlsx_path, image_dir, transform=None):
        """
        split_xlsx_path: path to train.xlsx or val.xlsx
        metadata_xlsx_path: path to dataset.xlsx (contains tumor/benign/malignant)
        image_dir: path to images folder
        """
        # Load image list
        split_df = pd.read_excel(split_xlsx_path)
        meta_df = pd.read_excel(metadata_xlsx_path)

        # Merge on image_id
        self.df = pd.merge(split_df, meta_df, on="image_id", how="left")

        # Generate labels
        self.df['label'] = self.df.apply(encode_label, axis=1)

        # Drop invalid labels
        self.df = self.df[self.df['label'] != -1].reset_index(drop=True)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, row['label']
