# clip_btxrd.py
import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- import your dataset class here ----
from dataset import BoneTumorDatasetWithCaption
from torchvision import models, transforms

try:
    from transformers import AutoTokenizer, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------
# Projection + Normalization
# ---------------------------
class ProjectionMLP(nn.Module):
    """Linear projection to common embedding dim with optional layernorm."""
    def __init__(self, in_dim: int, out_dim: int, use_ln: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)  # cosine space
        return x

# ---------------------------
# Image Encoder (torchvision)
# ---------------------------
class ImageEncoder(nn.Module):
    """Wrap a torchvision backbone and expose a pooled feature vector."""
    def __init__(self, name="efficientnet_b0", pretrained=True, out_dim=512):
        super().__init__()
        if name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
        elif name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(f"Unsupported image encoder: {name}")

        self.proj = ProjectionMLP(in_dim=feat_dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)          # (B, feat_dim)
        return self.proj(feats)           # (B, out_dim) L2-normalized

# ---------------------------
# Text Encoder (HuggingFace)
# ---------------------------
class TextEncoderHF(nn.Module):
    """HuggingFace text encoder wrapper with CLS pooling + projection."""
    def __init__(self, model_name: str, out_dim: int = 512, freeze_base: bool = True):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Please `pip install transformers` or switch to a simple encoder.")
        self.model = AutoModel.from_pretrained(model_name)
        hidden = self.model.config.hidden_size
        if freeze_base:
            for p in self.model.parameters():
                p.requires_grad = False
        self.proj = ProjectionMLP(in_dim=hidden, out_dim=out_dim)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # CLS pool (B, hidden_size)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)             # (B, out_dim) L2-normalized

# ---------------------------
# CLIP-style container
# ---------------------------
class CLIPLike(nn.Module):
    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, logit_scale_init: float = math.log(1/0.07)):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # Learnable logit scale (temperature^-1). Start near CLIP default (≈1/0.07).
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init, dtype=torch.float32))

    def forward(self, batch: Dict[str, Any]):
        img = batch["image"]    # (B, C, H, W)
        txt = batch["text"]     # dict from tokenizer: input_ids, attention_mask, ...
        img_emb = self.image_encoder(img)                                # (B, D)
        txt_emb = self.text_encoder(**txt)                               # (B, D)
        # cosine similarity matrix (B, B)
        logits = torch.matmul(img_emb, txt_emb.t())
        # apply temperature
        logit_scale = self.logit_scale.exp().clamp(1e-2, 100.0)
        logits = logits * logit_scale
        return logits, img_emb, txt_emb

# ---------------------------
# Contrastive Loss (Symmetric)
# ---------------------------
def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE: CE(img->text) + CE(text->img) / 2
    logits: (B, B) where logits[i, j] = sim(image_i, text_j) * scale
    """
    B = logits.size(0)
    target = torch.arange(B, device=logits.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.t(), target)
    return (loss_i2t + loss_t2i) / 2

# ---------------------------
# Collate (works with your dataset dict)
# ---------------------------
def collate_dict(batch):
    """
    Expects each item to be a dict with:
      'image' : Tensor (C,H,W)
      'label' : int
      'caption': str
      'text'  : dict of token tensors (input_ids, attention_mask, ...)
      'image_id': str
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    captions = [b["caption"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    # Merge tokenizer dicts: take keys from first, stack/squeeze
    text_keys = batch[0]["text"].keys()
    text = {}
    for k in text_keys:
        vals = [b["text"][k] for b in batch]
        # Ensure tensors and add batch dim if needed
        vals = [v if torch.is_tensor(v) else torch.tensor(v) for v in vals]
        # Many tokenizers return shape (seq,) per sample; stack to (B, seq)
        text[k] = torch.stack(vals, dim=0)

    return {"image": images, "label": labels, "caption": captions, "text": text, "image_id": image_ids}

# ---------------------------
# Training
# ---------------------------
@dataclass
class TrainConfig:
    image_encoder: str = "efficientnet_b0"   # or 'resnet50'
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    freeze_text: bool = True
    img_lr: float = 1e-4
    txt_lr: float = 1e-5
    wd: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    embed_dim: int = 512
    num_workers: int = 4
    amp: bool = True

def build_transforms(img_size=608):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--img_backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_img", type=float, default=1e-4)
    parser.add_argument("--lr_txt", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_text", action="store_true", default=True)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--img_size", type=int, default=608)
    args = parser.parse_args()

    cfg = TrainConfig(
        image_encoder=args.img_backbone,
        text_model_name=args.text_model,
        freeze_text=args.freeze_text,
        img_lr=args.lr_img,
        txt_lr=args.lr_txt,
        wd=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
    )

    # ---- tokenizer for your dataset ----
    if not HF_AVAILABLE:
        raise RuntimeError("Install `transformers` to use the HF text encoder.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    def hf_tokenize(s: str):
        # returns dict of tensors (seq,) per sample; collate stacks to (B, seq)
        return tokenizer(s, padding="max_length", truncation=True, max_length=32, return_tensors=None)

    # ---- dataset ----
    DATASET_DIR = 'data/BTXRD'
    metadata_xlsx_path = os.path.join('dataset_with_captions.xlsx')
    train_path = os.path.join(DATASET_DIR, 'train.xlsx')
    test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
    IMG_DIR = os.path.join(DATASET_DIR, 'images')

    ds = BoneTumorDatasetWithCaption(
        split_xlsx_path=train_path,
        metadata_xlsx_path=metadata_xlsx_path,
        image_dir=IMG_DIR,
        transform=build_transforms(args.img_size),
        text_tokenizer=lambda s: {k: torch.tensor(v.squeeze(0)) for k, v in tokenizer(s, padding='max_length', truncation=True, max_length=32, return_tensors='pt').items()},
        return_dict=True
    )

    # Example (uncomment after you import):
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_dict, pin_memory=True)

    # ---- model ----
    img_enc = ImageEncoder(name=cfg.image_encoder, pretrained=True, out_dim=cfg.embed_dim)
    txt_enc = TextEncoderHF(model_name=cfg.text_model_name, out_dim=cfg.embed_dim, freeze_base=cfg.freeze_text)
    model = CLIPLike(img_enc, txt_enc).cuda()

    # ---- optimizer (separate groups) ----
    params = [
        {"params": [p for p in model.image_encoder.parameters() if p.requires_grad], "lr": cfg.img_lr},
        {"params": [p for p in model.text_encoder.parameters() if p.requires_grad], "lr": cfg.txt_lr},
        {"params": [model.logit_scale], "lr": cfg.img_lr},
    ]
    optim = torch.optim.AdamW(params, weight_decay=cfg.wd)

    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    # ---- training ----
    model.train()
    for epoch in tqdm(range(cfg.epochs)):
        running = 0.0
        for batch in loader:
            images = batch["image"].cuda(non_blocking=True)
            text = {k: v.cuda(non_blocking=True) for k, v in batch["text"].items()}

            with torch.amp.autocast(enabled=cfg.amp):
                logits, _, _ = model({"image": images, "text": text})
                loss = clip_contrastive_loss(logits)

            scaler.scale(loss).zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()

        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {running / len(loader):.4f}")

if __name__ == "__main__":
    main()
