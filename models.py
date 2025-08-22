import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden * 2, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*[
            ConvBNAct(hidden, hidden, k=3, s=1, p=1) for _ in range(n)
        ])
        self.cv2 = ConvBNAct(hidden * 2, out_channels, k=1, s=1, p=0)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((y1, self.blocks(y2)), dim=1))

class YOLOv8nCls(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = ConvBNAct(3, 16, k=3, s=2, p=1)       # 3 -> 16
        self.stage1 = C2f(16, 32, n=1)
        self.down1 = ConvBNAct(32, 64, k=3, s=2, p=1)

        self.stage2 = C2f(64, 64, n=2)
        self.down2 = ConvBNAct(64, 128, k=3, s=2, p=1)

        self.stage3 = C2f(128, 128, n=2)
        self.down3 = ConvBNAct(128, 256, k=3, s=2, p=1)

        self.stage4 = C2f(256, 256, n=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)