"""
ChannelNet = SRCNN cascaded with DnCNN
Architectures follow the paper (Section III-B) exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    Super-resolution CNN (3 layers):
      64 × 9×9  → ReLU
      32 × 1×1  → ReLU
       1 × 5×5
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 9, padding=4)   # 2 channels = real + imag
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 2, 5, padding=2)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


class DnCNN(nn.Module):
    """
    Denoising residual network (20 layers) as in Zhang et al. / the paper.
    """
    def __init__(self, depth=20, channels=64):
        super().__init__()
        layers = [nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels, eps=1e-4, momentum=0.05),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.Conv2d(channels, 2, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x - self.net(x)          # residual learning


class ChannelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.srcnn = SRCNN()
        self.dncnn = DnCNN()

    def forward(self, x, stage="full"):
        z = self.srcnn(x)
        if stage == "sr":
            return z
        return self.dncnn(z)
