import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c2, k=1, s=1, p=0)
        self.cv2 = ConvBNAct(c2, c2, k=3, s=1, p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """
    Ultralytics-style C2f block.
    Concats: [y1, y2 (pre-block), y2_1, ..., y2_n] → fuse
    So final input to cv2 has (n+2)*hidden channels.
    """
    def __init__(self, in_channels, out_channels, n=1, e=0.5, shortcut=True):
        super().__init__()
        hidden = int(out_channels * e)
        self.cv1 = ConvBNAct(in_channels, hidden * 2, k=1, s=1, p=0)
        self.blocks = nn.ModuleList([Bottleneck(hidden, hidden, shortcut) for _ in range(n)])
        self.cv2 = ConvBNAct(hidden * (n + 2), out_channels, k=1, s=1, p=0)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, dim=1)
        outputs = [y1, y2]
        for m in self.blocks:
            y2 = m(y2)
            outputs.append(y2)
        return self.cv2(torch.cat(outputs, dim=1))


class ClassifyHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 1280)
        self.bn = nn.BatchNorm1d(1280)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.0)
        self.fc2 = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class YOLOv8nCls(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, checkpoint_path=None):
        super().__init__()
        self.stem   = ConvBNAct(3, 16, k=3, s=2)
        self.down0  = ConvBNAct(16, 32, k=3, s=2)
        
        self.stage1 = C2f(32, 32, n=1, e=1.0)
        self.stage2 = C2f(64, 64, n=2, e=1.0)
        self.stage3 = C2f(128, 128, n=2, e=1.0)
        self.stage4 = C2f(256, 256, n=1, e=1.0)

        self.down1  = ConvBNAct(32, 64, k=3, s=2)
        self.down2  = ConvBNAct(64, 128, k=3, s=2)
        self.down3  = ConvBNAct(128, 256, k=3, s=2)

        self.head = ClassifyHead(256, num_classes)

        if pretrained and checkpoint_path:
            self._load_pretrained_weights(checkpoint_path)

    def forward(self, x):
        x = self.stem(x)
        x = self.down0(x)
        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        x = self.down3(x)
        x = self.stage4(x)

        return self.head(x)

    def _load_pretrained_weights(self, checkpoint_path):
	    print(f"Loading weights from {checkpoint_path}")
	    ckpt = torch.load(checkpoint_path, map_location='cpu')

	    if 'model' in ckpt and hasattr(ckpt['model'], 'model'):
	        pretrained_dict = ckpt['model'].model.state_dict()
	    else:
	        raise ValueError("Unsupported checkpoint format")

	    model_dict = self.state_dict()

	    # Mapping from Ultralytics module index to your named modules
	    prefix_map = {
	        'model.0': 'stem',
	        'model.1': 'down0',
	        'model.2': 'stage1',
	        'model.3': 'down1',
	        'model.4': 'stage2',
	        'model.5': 'down2',
	        'model.6': 'stage3',
	        'model.7': 'down3',
	        'model.8': 'stage4',
	        # model.9 is head → we skip it
	    }

	    mapped_pretrained = {}
	    for full_key, weight in pretrained_dict.items():
	        parts = full_key.split('.', 2)
	        if len(parts) < 3:
	            continue
	        prefix = f"{parts[0]}.{parts[1]}"
	        if prefix in prefix_map:
	            new_key = f"{prefix_map[prefix]}.{parts[2]}"
	            if new_key in model_dict and model_dict[new_key].shape == weight.shape:
	                mapped_pretrained[new_key] = weight

	    missing_keys, unexpected_keys = self.load_state_dict(mapped_pretrained, strict=False)

	    print(f"✅ Loaded {len(mapped_pretrained)} layers.")
	    print(f"❌ Missing keys: {len(missing_keys)}")
	    for k in missing_keys[:10]:
	        print(" -", k)



