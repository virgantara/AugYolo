import yaml
import torch
import torch.nn as nn
from pathlib import Path

from models_lib import ConvBNAct as Conv, C2f, ClassifyHead as Classify

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_base, ConvNeXt_Tiny_Weights

class ConvNeXtBTXRD(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtBTXRD, self).__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)  # You can change to convnext_small/base/large

        # Replace the classifier (head)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


module_map = {
    'Conv': Conv,
    'C2f': C2f,
    'Classify': Classify,
}

def make_divisible(x, divisor):
    return int((x + divisor / 2) // divisor * divisor)

class YOLOv8ClsFromYAML(nn.Module):
    def __init__(self, yaml_path='yolov8n-cls.yaml', scale='n', num_classes=None, pretrained=None):
        super().__init__()
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Scaling
        depth, width, max_channels = cfg['scales'][scale]
        backbone, head = cfg['backbone'], cfg['head']
        nc = num_classes if num_classes is not None else cfg['nc']

        self.model = self.build_model(backbone + head, depth, width, max_channels, nc)

        if pretrained:
            self._load_pretrained_weights(pretrained)

    def build_model(self, layers_cfg, depth_mul, width_mul, max_ch, nc):
        layers = []
        ch = [3]  # track channels of outputs
        for i, (from_idx, repeat, module_name, args) in enumerate(layers_cfg):
            module = module_map[module_name]

            # Adjust width and depth
            if module_name == 'C2f':
                c1 = ch[from_idx]
                c2 = args[0]
                shortcut = args[1] if len(args) > 1 else True
                c2 = make_divisible(c2 * width_mul, 8)
                repeat = max(round(repeat * depth_mul), 1)
                m = module(c1, c2, n=repeat, shortcut=shortcut)
            elif module_name == 'Conv':
                c2 = make_divisible(args[0] * width_mul, 8)
                k = args[1]
                s = args[2]
                c1 = ch[from_idx]
                m = module(c1, c2, k=k, s=s)
            elif module_name == 'Classify':
                in_c = ch[from_idx]
                m = module(in_c, nc)
            else:
                raise NotImplementedError(f"Module {module_name} not implemented")

            layers.append(m)
            ch.append(m.out_channels if hasattr(m, 'out_channels') else c2 if module_name != 'Classify' else nc)

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _load_pretrained_weights(self, checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        if 'model' in ckpt and hasattr(ckpt['model'], 'model'):
            pretrained_dict = ckpt['model'].model.state_dict()
        else:
            raise ValueError("Unsupported checkpoint format")

        # Strip 'model.' prefix
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("model."):
                new_key = k[len("model."):]
            else:
                new_key = k
            new_pretrained_dict[new_key] = v

        missing, unexpected = self.model.load_state_dict(new_pretrained_dict, strict=False)
        print(f"Loaded {len(new_pretrained_dict)} layers.")
        print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")
