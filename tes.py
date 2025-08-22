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
    def __init__(self, num_classes=3):
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


if __name__ == "__main__":
    model = YOLOv8nCls(num_classes=3)
    dummy_input = torch.randn(2, 3, 608, 608)
    out = model(dummy_input)
    print(model)
    print("Output shape:", out.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
