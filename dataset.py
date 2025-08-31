import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms import ToTensor

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

class BoneTumorDatasetWithCaption(Dataset):
    """
    New BTXRD dataset loader (old + caption).

    Parameters
    ----------
    split_xlsx_path : str
        Path to train.xlsx / val.xlsx (must contain 'image_id' column).
    metadata_xlsx_path : str
        Path to dataset.xlsx (contains tumor/benign/malignant and 'caption').
    image_dir : str
        Directory where images live.
    transform : callable | None
        Torchvision transform for images.
    text_tokenizer : callable | None
        Optional callable that maps a string caption -> tokenized outputs (e.g., tensors or dict).
        Examples:
          - CLIP: lambda s: clip_tokenizer(s, truncate=True)
          - HF:   lambda s: hf_tokenizer(s, padding='max_length', truncation=True, return_tensors='pt')
          - DIY:  lambda s: my_tf_idf_vectorizer.transform([s]).toarray()[0]
    return_dict : bool
        If True, returns a dict with keys ['image','label','caption','text','image_id'].
        If False, returns (image_tensor, label) to remain backward compatible.
    caption_col : str
        Name of caption column in metadata. Defaults to 'caption'.
    """
    def __init__(
        self,
        split_xlsx_path,
        metadata_xlsx_path,
        image_dir,
        transform=None,
        text_tokenizer=None,
        return_dict=True,
        caption_col="caption"
    ):
        # Load sheets
        split_df = pd.read_excel(split_xlsx_path)
        meta_df = pd.read_excel(metadata_xlsx_path)

        if "image_id" not in split_df.columns:
            raise ValueError("split_xlsx must have an 'image_id' column.")
        if "image_id" not in meta_df.columns:
            raise ValueError("metadata_xlsx must have an 'image_id' column.")
        if caption_col not in meta_df.columns:
            # Allow silently continuing with empty captions if column missing (but warn)
            # You can also raise here if you prefer strictness.
            meta_df[caption_col] = ""

        # Merge on image_id
        df = pd.merge(split_df, meta_df, on="image_id", how="left")

        # Generate labels
        df["label"] = df.apply(encode_label, axis=1)

        # Drop invalid labels
        df = df[df["label"] != -1].reset_index(drop=True)

        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.text_tokenizer = text_tokenizer
        self.return_dict = return_dict
        self.caption_col = caption_col

        # Optional: keep a small mapping (useful for logging)
        self.id2label = {0: "normal", 1: "benign", 2: "malignant"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_id):
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row["image_id"])
        label = int(row["label"])

        # Robustly get caption (ensure string)
        caption = row.get(self.caption_col, "")
        if pd.isna(caption):
            caption = ""
        caption = str(caption)

        # Optionally tokenize caption
        text = None
        if self.text_tokenizer is not None:
            text = self.text_tokenizer(caption)
            # If tokenizer returns dict of tensors with batch dim=1, squeeze it
            if isinstance(text, dict):
                text = {k: (v.squeeze(0) if hasattr(v, "shape") and v.shape[:1] == (1,) else v) for k, v in text.items()}

        if self.return_dict:
            return {
                "image": image,
                "label": label,
                "caption": caption,   # raw text
                "text": text,         # tokenized output or None
                "image_id": row["image_id"],
            }
        else:
            # Backward compatible: (image, label)
            return image, label

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
        else:
            image = ToTensor()(image)  # ensure tensor even if no transform

        return image, row['label']


class BoneTumorDatasetCenter(Dataset):
    def __init__(self, metadata_xlsx_path, image_dir, center_id, transform=None):
        """
        split_xlsx_path: path to train.xlsx or val.xlsx
        metadata_xlsx_path: path to dataset.xlsx (contains tumor/benign/malignant)
        image_dir: path to images folder
        """
        # Load image list
        meta_df = pd.read_excel(metadata_xlsx_path)

        # Merge on image_id
        self.df = meta_df[meta_df["center"] == center_id].copy() 

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
        else:
            image = ToTensor()(image)  # ensure tensor even if no transform

        return image, row['label']
