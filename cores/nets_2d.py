import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ---------- Depth-wise-Separable 2-D Conv ----------
class DSConv2d(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, in_c, k, stride=s, padding=p,
                      groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


# ---------- 带残差的 DW-Sep Block ----------
class DSResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.body = DSConv2d(in_c, out_c, k=3, s=1, p=1)

        self.skip = nn.Identity()
        if in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


# ---------- 主网络 ----------
class NetAFD(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 1) conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        # 2) 四个 DW-Sep 残差块
        self.stage = nn.Sequential(
            DSResBlock(8, 8),
            DSResBlock(8, 16),
            DSResBlock(16, 16),
            DSResBlock(16, 32)
        )
        # 3) GAP + FC
        self.gap = nn.AdaptiveAvgPool2d(1)  # → (B,32,1,1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (B,8 ,11,11)
        x = self.stage(x)  # (B,32,11,11)
        feat = self.gap(x).flatten(1)  # (B,32)
        logit = self.fc(feat)  # (B,1)
        return logit, feat  # 返回 logit 和特征


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 22, 22)

    model = NetAFD()
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"MACs      : {macs:,}")
    print(f"Parameters: {params:,}")

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)  # warm-up
        start_time = time.time()
        logit, feat = model(input_tensor)
        end_time = time.time()

    print(f"Inference Time (CPU): {(end_time - start_time) * 1000:.3f} ms")
    print(f"logit shape : {logit.shape}")
    print(f"feat  shape : {feat.shape}")  # (B,32)
