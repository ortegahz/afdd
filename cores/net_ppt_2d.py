# =============================================================
#  Simple-SE-2D-Net  (input size = 1 × 224 × 224)
# =============================================================
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ---------------- 1)  SE 注意力 ----------------
class SE2d(nn.Module):
    """Squeeze-and-Excitation for 2-D features"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B,C,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x : (B,C,H,W)
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)  # (B,C)
        y = self.fc(y).view(b, c, 1, 1)  # (B,C,1,1)
        return x * y  # broadcast 乘


# ---------------- 2)  Conv-BN-ReLU-SE Block ----------------
class CBRSE2d(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 k: int = 3,
                 s: int = 1,
                 p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k,
                              stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SE2d(out_c)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.se(x)
        return x


# ---------------- 3)  整体网络 ----------------
class SimpleSE2DNet224(nn.Module):
    """
    输入 :  (B, 1, 224, 224)
    输出 :  out  (B, num_classes)
            feat (B, 256) —— 全局特征
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        # 224 → 112
        self.stem = CBRSE2d(1, 32, k=7, s=2, p=3)

        # 112 → 56
        self.stage1 = CBRSE2d(32, 64, k=5, s=2, p=2)

        # 56  → 28
        self.stage2 = CBRSE2d(64, 128, k=3, s=2, p=1)

        # 28  → 14
        self.stage3 = CBRSE2d(128, 256, k=3, s=2, p=1)

        # 14  → 7
        self.stage4 = CBRSE2d(256, 256, k=3, s=2, p=1)

        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,256,1,1)
        self.drop = nn.Dropout(0.3)
        self.head1 = nn.Linear(256, 128)
        self.head2 = nn.Linear(128, num_classes)

    def forward(self, x):  # x : (B,1,224,224)
        x = self.stem(x)  # (B, 32,112,112)
        x = self.stage1(x)  # (B, 64, 56, 56)
        x = self.stage2(x)  # (B,128, 28, 28)
        x = self.stage3(x)  # (B,256, 14, 14)
        x = self.stage4(x)  # (B,256,  7,  7)

        feat = self.gap(x).squeeze(-1).squeeze(-1)  # (B,256)
        x = F.relu(self.head1(self.drop(feat)))  # (B,128)
        out = self.head2(x)  # (B,num_classes)
        return out, feat


# ---------------- 4)  Demo ----------------
if __name__ == '__main__':
    model = SimpleSE2DNet224(num_classes=1)
    dummy = torch.randn(1, 1, 224, 224)

    macs, params = profile(model, inputs=(dummy,), verbose=False)
    print(f'MACs   : {macs / 1e6:.1f} M')
    print(f'Params : {params / 1e6:.3f} M')

    with torch.no_grad():
        t0 = time.time()
        out, feat = model(dummy)
        t1 = time.time()
    print(f'CPU Inference Time : {(t1 - t0) * 1e3:.2f} ms')
