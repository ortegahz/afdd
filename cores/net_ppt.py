# =============================================================
#  Simple-SE-1D-Net  (input length = 1024)
# =============================================================
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ---------------- 1)  SE 注意力 ----------------
class SE1d(nn.Module):
    """Squeeze-and-Excitation for 1-D features"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B,C,L)
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)  # (B,C)
        y = self.fc(y).view(b, c, 1)  # (B,C,1)
        return x * y  # broadcast 乘


# ---------------- 2)  Conv-BN-ReLU-SE Block ----------------
class CBRSE(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SE1d(out_c)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.se(x)
        return x


# ---------------- 3)  整体网络 ----------------
class SimpleSE1DNet1024(nn.Module):
    """
    输入 : (B, 1, 1024)
    输出 :  out  (B, 1)
            feat (B, 256) —— 全局特征
    """

    def __init__(self, num_classes=1):
        super().__init__()
        # 1024 → 512
        self.stem = CBRSE(1, 32, k=7, s=2, p=3)

        # 512 → 256
        self.stage1 = CBRSE(32, 64, k=5, s=2, p=2)

        # 256 → 128
        self.stage2 = CBRSE(64, 128, k=3, s=2, p=1)

        # 128 →  64
        self.stage3 = CBRSE(128, 256, k=3, s=2, p=1)

        #  64 →  32
        self.stage4 = CBRSE(256, 256, k=3, s=2, p=1)

        self.gap = nn.AdaptiveAvgPool1d(1)  # (B,256,1)
        self.drop = nn.Dropout(0.3)
        self.head1 = nn.Linear(256, 128)
        self.head2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)  # (B, 32, 512)
        x = self.stage1(x)  # (B, 64, 256)
        x = self.stage2(x)  # (B,128, 128)
        x = self.stage3(x)  # (B,256,  64)
        x = self.stage4(x)  # (B,256,  32)

        feat = self.gap(x).squeeze(-1)  # (B,256)
        x = F.relu(self.head1(self.drop(feat)))
        out = self.head2(x)  # (B,1)
        return out, feat


# ---------------- 4)  Demo ----------------
if __name__ == "__main__":
    model = SimpleSE1DNet1024()
    dummy = torch.randn(1, 1, 1024)

    macs, params = profile(model, inputs=(dummy,), verbose=False)
    print(f'MACs   : {macs / 1e6:.1f} M')
    print(f'Params : {params / 1e6:.3f} M')

    with torch.no_grad():
        t0 = time.time()
        out, feat = model(dummy)
        t1 = time.time()
    print(f'CPU Inference Time : {(t1 - t0) * 1e3:.2f} ms')
