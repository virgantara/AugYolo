import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights

# from torchvision.models import vit_b_32, ViT_B_32_Weights
# from torchvision.models import vit_l_16, ViT_L_16_Weights
# from torchvision.models import vit_l_32, ViT_L_32_Weights
# from torchvision.models import vit_h_14, ViT_H_14_Weights

class VisionTransformerBTXRD(nn.Module):
    """
    Wrapper for torchvision Vision Transformer that replaces the classification head
    to match `num_classes`, similar to your EfficientNetBTXRD.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_p: float = 0.0):
        super().__init__()
        # Pick a ViT backbone (change variant/weights as needed)
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)

        # Figure out in_features of the last linear layer robustly across torchvision versions
        # Common layouts:
        # - self.backbone.heads = nn.Sequential(nn.Linear(embed_dim, num_classes))
        # - or contains a named `head` module inside `heads`
        if hasattr(self.backbone.heads, "head") and isinstance(self.backbone.heads.head, nn.Linear):
            in_features = self.backbone.heads.head.in_features
        else:
            # Fallback: assume the last module inside `heads` is Linear
            last = list(self.backbone.heads.modules())[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
            else:
                # Last resort: use the embedding dimension attribute name used by torchvision
                in_features = getattr(self.backbone, "hidden_dim", None) \
                              or getattr(self.backbone, "embed_dim", None)
                if in_features is None:
                    raise RuntimeError("Could not determine ViT classifier in_features.")

        # Replace classification head (keep it simple like your EfficientNet wrapper)
        head = []
        if dropout_p and dropout_p > 0:
            head.append(nn.Dropout(p=dropout_p, inplace=False))
        head.append(nn.Linear(in_features, num_classes))
        self.backbone.heads = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
