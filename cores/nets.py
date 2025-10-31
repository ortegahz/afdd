# FILE: nets.py

import time
from typing import Tuple

import torch
import torch.nn as nn
from thop import profile

from utils.macros import SAMPLE_RATE


class NetAFDV0(nn.Module):
    def __init__(self):
        super(NetAFDV0, self).__init__()
        # self.channel_in = int(SAMPLE_RATE / 50) * 2
        _channel_in = int(SAMPLE_RATE / 50)
        self.channel_in = ((_channel_in // 32) + 1) * 32
        self.channel_out = 128
        self.channels = [32, 64]
        self.dropout_rate = 0.5

        self.conv1 = nn.Conv1d(1, self.channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(int(self.channel_in / (2 ** len(self.channels)) * self.channels[1]), self.channel_out)
        self.fc2 = nn.Linear(self.channel_out, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        feat = self.fc1(x)
        x = self.dropout(F.relu(feat))
        x = self.fc2(x)

        return x, feat


class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积模块，并行使用不同大小的卷积核来捕捉不同范围的特征。
    - 使用 1x1 卷积作为瓶颈层来降低计算成本。
    - 拼接所有分支的输出并通过一个 1x1 卷积进行特征融合。
    """

    def __init__(self, in_c, out_c, bottleneck_ratio=0.25):
        super().__init__()
        bottleneck_c = max(1, int(in_c * bottleneck_ratio))
        k_sizes = [3, 5, 7]

        # 瓶颈层，用于降低后续多尺度卷积的计算量
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_c, bottleneck_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_c),
            nn.ReLU(inplace=True)
        )

        # 并行的多尺度卷积分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bottleneck_c, bottleneck_c, kernel_size=k, padding=(k - 1) // 2, bias=False),
                nn.BatchNorm1d(bottleneck_c),
                nn.ReLU(inplace=True)
            ) for k in k_sizes
        ])

        # 特征融合层
        self.merge = nn.Sequential(
            nn.Conv1d(bottleneck_c * len(k_sizes), out_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_c)
        )

        # 残差连接的捷径，确保输入输出通道数和维度一致
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        shortcut_x = self.shortcut(x)
        x = self.bottleneck(x)
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.merge(x)
        return F.relu(x + shortcut_x)


class ResidualBlock(nn.Module):
    """
    标准的残差块，包含两个卷积层和一个捷径连接。
    """

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class UpBlock(nn.Module):
    """
    解码器中的上采样模块 (U-Net 风格)。
    - 使用 Upsample + Conv 代替转置卷积，减少伪影。
    - 接收来自解码器下一层和编码器对应层的 skip connection 输入。
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # 上采样，并将通道数减半
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # 拼接 skip connection 后进行卷积
        self.conv = ResidualBlock(in_c + skip_c, out_c)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class NetAFDAE_UNet(nn.Module):
    """
    一个增强版的自编码器，集成了以下特性以提升重构精度：
    1.  **多尺度卷积 (Multi-scale Convolutions)**: 在编码器初期捕捉不同尺度的时序特征。
    2.  **残差连接 (Residual Connections)**: 在编码器和解码器中使用残差块，缓解梯度消失，保留信息。
    3.  **U-Net 风格的跳转连接 (Skip Connections)**: 将编码器的浅层特征直接传递给解码器，帮助恢复高频细节。
    4.  **上采样+卷积 (Upsample + Conv)**: 在解码器中使用，以减少转置卷积可能带来的棋盘格效应。
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        _channel_in = int(SAMPLE_RATE / 50)
        in_len = ((_channel_in // 32) + 1) * 32
        assert in_len == 448

        # --- 编码器 (Encoder) ---
        # 初始多尺度卷积层
        self.in_conv = MultiScaleConvBlock(1, 16)  # 448 -> 448
        # 下采样块
        self.down1 = self._make_encoder_stage(16, 32)  # 448 -> 224
        self.down2 = self._make_encoder_stage(32, 64)  # 224 -> 112
        self.down3 = self._make_encoder_stage(64, 128)  # 112 -> 56
        self.down4 = self._make_encoder_stage(128, 256)  # 56  -> 28

        # --- 瓶颈层 (Bottleneck) ---
        self.bottleneck_conv = self._make_encoder_stage(256, 512)  # 28 -> 14
        final_seq_len = in_len // 32  # 448 / 32 = 14
        self.encoder_fc = nn.Linear(512 * final_seq_len, latent_dim)

        # --- 解码器 (Decoder) ---
        self.decoder_fc = nn.Linear(latent_dim, 512 * final_seq_len)
        self.unflatten = lambda x: x.view(-1, 512, final_seq_len)

        # 上采样块
        self.up1 = UpBlock(512, 256, 256)  # 14 -> 28
        self.up2 = UpBlock(256, 128, 128)  # 28 -> 56
        self.up3 = UpBlock(128, 64, 64)  # 56 -> 112
        self.up4 = UpBlock(64, 32, 32)  # 112 -> 224
        self.up5 = UpBlock(32, 16, 16)  # 224 -> 448

        # 输出层，将特征图映射回单通道信号
        self.out_conv = nn.Conv1d(16, 1, kernel_size=1)

    def _make_encoder_stage(self, in_c, out_c):
        return nn.Sequential(
            nn.MaxPool1d(2),
            ResidualBlock(in_c, out_c)
        )

    def encode(self, x):
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        bottleneck = self.bottleneck_conv(s5)

        z = self.encoder_fc(bottleneck.flatten(1))
        return z, [s1, s2, s3, s4, s5]

    def decode(self, z, skips):
        s1, s2, s3, s4, s5 = skips
        x = self.decoder_fc(z)
        x = self.unflatten(x)

        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)

        return self.out_conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, skips = self.encode(x)
        reconstructed_x = self.decode(z, skips)
        return reconstructed_x, z


import torch.nn as nn


# ------------- 基本积木 -------------
def CBR(in_c, out_c, k=3, s=1, p=1):  # Conv-BN-ReLU
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True)
    )


# ------------- 主网络 -------------
class NetAFD(nn.Module):
    """
    下采样 448→224→112（AvgPool）→56→28→14→7（stride=2 Conv）
    """

    def __init__(self, in_len=448):
        super().__init__()
        _channel_in = int(SAMPLE_RATE / 50)
        in_len = ((_channel_in // 32) + 1) * 32
        assert in_len == 448

        # 1) 前两级：stride=1 Conv + AvgPool(2)
        self.stage1 = nn.Sequential(
            CBR(1, 8, k=3, s=1, p=1),  # 448
            nn.AvgPool1d(kernel_size=2)  # 224
        )
        self.stage2 = nn.Sequential(
            CBR(8, 16, k=3, s=1, p=1),  # 224
            nn.AvgPool1d(kernel_size=2)  # 112
        )

        # 2) 后四级：全部 stride=2 Conv
        self.stage3 = CBR(16, 32, k=3, s=2, p=1)  # 112→56
        self.stage4 = CBR(32, 32, k=3, s=2, p=1)  # 56 →28
        self.stage5 = CBR(32, 64, k=3, s=2, p=1)  # 28 →14
        self.stage6 = CBR(64, 64, k=3, s=2, p=1)  # 14 → 7

        # 3) 全连接部分
        self.fc1 = nn.Linear(64 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):  # x : (B,1,448)
        x = self.stage1(x)  # (B,  8,224)
        x = self.stage2(x)  # (B, 16,112)
        x = self.stage3(x)  # (B, 32, 56)
        x = self.stage4(x)  # (B, 32, 28)
        x = self.stage5(x)  # (B, 64, 14)
        x = self.stage6(x)  # (B, 64,  7)

        x = x.flatten(1)  # (B, 448)
        feat = self.fc1(x)  # (B, 128)
        x = self.drop(F.relu(feat))
        x = self.fc2(x)  # (B, 1)
        return x, feat


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple  # <--- 1. 导入 Tuple


# ------------- 基本积木（编码器和解码器通用） -------------
def CBR(in_c, out_c, k=3, s=1, p=1):  # Conv-BN-ReLU
    """
    一个标准的卷积块，用于编码器。
    """
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True)
    )


# ------------- 自编码器解码器积木 -------------
def DeCBR(in_c, out_c, k=3, s=2, p=1, op=1):  # DeConv-BN-ReLU
    """
    转置卷积块，用于解码器中通过学习来上采样。
    参数 (k=3, s=2, p=1, op=1) 配置可以使序列长度精确地加倍。
    """
    return nn.Sequential(
        nn.ConvTranspose1d(in_c, out_c, k, stride=s, padding=p, output_padding=op, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True)
    )


# ------------- AutoEncoder Network -------------
class NetAFDAE(nn.Module):
    """
    基于NetAFD编码器结构的自编码器（AutoEncoder）。
    它由一个编码器（Encoder）将输入信号压缩成潜在向量，
    以及一个解码器（Decoder）将潜在向量重建为原始信号构成。
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        # 确认输入长度与原始模型一致
        _channel_in = int(SAMPLE_RATE / 50)
        in_len = ((_channel_in // 32) + 1) * 32
        assert in_len == 448, "Input length must be 448 for this architecture"

        # ----------------- 编码器 (Encoder) -----------------
        # 下采样路径: 448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.encoder_stage1 = nn.Sequential(
            CBR(1, 8, k=3, s=1, p=1),  # 448
            nn.AvgPool1d(kernel_size=2)  # 224
        )
        self.encoder_stage2 = nn.Sequential(
            CBR(8, 16, k=3, s=1, p=1),  # 224
            nn.AvgPool1d(kernel_size=2)  # 112
        )
        self.encoder_stage3 = CBR(16, 32, k=3, s=2, p=1)  # 112 -> 56
        self.encoder_stage4 = CBR(32, 32, k=3, s=2, p=1)  # 56 -> 28
        self.encoder_stage5 = CBR(32, 64, k=3, s=2, p=1)  # 28 -> 14
        self.encoder_stage6 = CBR(64, 64, k=3, s=2, p=1)  # 14 -> 7

        # 全连接层，用于从特征图生成潜在向量
        self.encoder_fc = nn.Linear(64 * 7, latent_dim)

        # ----------------- 解码器 (Decoder) -----------------
        # 上采样路径: 7 -> 14 -> 28 -> 56 -> 112 -> 224 -> 448 (与编码器对称)
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7)

        # 将展平的向量恢复为 (B, 64, 7) 的特征图
        self.decoder_unflatten = lambda x: x.view(-1, 64, 7)

        # 使用转置卷积镜像编码器的步进卷积层
        self.decoder_stage6 = DeCBR(64, 64, k=3, s=2, p=1, op=1)  # 7 -> 14
        self.decoder_stage5 = DeCBR(64, 32, k=3, s=2, p=1, op=1)  # 14 -> 28
        self.decoder_stage4 = DeCBR(32, 32, k=3, s=2, p=1, op=1)  # 28 -> 56
        self.decoder_stage3 = DeCBR(32, 16, k=3, s=2, p=1, op=1)  # 56 -> 112

        # 使用上采样层+卷积层镜像编码器的平均池化层
        self.decoder_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 112 -> 224
            CBR(16, 8, k=3, s=1, p=1)
        )
        self.decoder_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 224 -> 448
            # 最后一层仅用卷积层输出重建信号，不使用BN和ReLU
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码过程，将输入信号压缩为潜在向量"""
        x = self.encoder_stage1(x)
        x = self.encoder_stage2(x)
        x = self.encoder_stage3(x)
        x = self.encoder_stage4(x)
        x = self.encoder_stage5(x)
        x = self.encoder_stage6(x)
        x = x.flatten(1)  # (B, C, L) -> (B, C*L)
        z = self.encoder_fc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码过程，从潜在向量重建信号"""
        x = F.relu(self.decoder_fc(z))  # 激活后进入卷积部分
        x = self.decoder_unflatten(x)
        x = self.decoder_stage6(x)
        x = self.decoder_stage5(x)
        x = self.decoder_stage4(x)
        x = self.decoder_stage3(x)
        x = self.decoder_stage2(x)
        reconstructed_x = self.decoder_stage1(x)
        return reconstructed_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # <--- 2. 修改此处
        """
        完整的前向传播。
        返回: (重建的信号, 潜在向量)
        """
        z = self.encode(x)
        reconstructed_x = self.decode(z)
        return reconstructed_x, z


import torch.nn as nn


# def CBR(in_c, out_c, k=3, s=1, p=1):  # 普通 1-D Conv
#     return nn.Sequential(
#         nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=False),
#         nn.BatchNorm1d(out_c),
#         nn.ReLU(inplace=True)
#     )


class NetAFDV2(nn.Module):
    """
    ↓ 448→224→112(A)→56→28→14→7（与原网络的降采样点一致）
    A = AvgPool1d(2)
    通道数改为 [4, 8, 16, 16, 32, 32]，最终用 GAP 去掉 64*7→128 的大 FC。
    总参数量 ≈ 6.4 k（原版 ≈ 81 k），已小于 1/10。
    """

    def __init__(self):
        super().__init__()

        # ---------- 1) 前两级：stride=1 Conv + AvgPool ----------
        self.stage1 = nn.Sequential(
            CBR(1, 4, k=3, s=1, p=1),  # 448
            nn.AvgPool1d(kernel_size=2)  # 224
        )
        self.stage2 = nn.Sequential(
            CBR(4, 8, k=3, s=1, p=1),  # 224
            nn.AvgPool1d(kernel_size=2)  # 112
        )

        # ---------- 2) 后四级：全部 stride=2 Conv ----------
        self.stage3 = CBR(8, 16, k=3, s=2, p=1)  # 112→56
        self.stage4 = CBR(16, 16, k=3, s=2, p=1)  # 56 →28
        self.stage5 = CBR(16, 32, k=3, s=2, p=1)  # 28 →14
        self.stage6 = CBR(32, 32, k=3, s=2, p=1)  # 14 → 7

        # ---------- 3) GAP + 极小的 FC ----------
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, 32, 1)
        self.fc = nn.Linear(32, 1)  # 32 → 1
        self.drop = nn.Dropout(0.5)

    def forward(self, x):  # x: (B,1,448)
        x = self.stage1(x)  # (B, 4 ,224)
        x = self.stage2(x)  # (B, 8 ,112)
        x = self.stage3(x)  # (B, 16, 56)
        x = self.stage4(x)  # (B, 16, 28)
        x = self.stage5(x)  # (B, 32, 14)
        x = self.stage6(x)  # (B, 32,  7)

        feat = self.gap(x).squeeze(-1)  # (B, 32)
        out = self.fc(self.drop(feat))  # (B, 1)
        return out, feat


class DSConv1d(nn.Sequential):
    """
    Depthwise-Separable Conv1d = depthwise(k) + pointwise(1)
    """

    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            # depthwise：groups = in_c
            nn.Conv1d(in_c, in_c, k, stride=s, padding=p,
                      groups=in_c, bias=False),
            nn.BatchNorm1d(in_c),
            nn.ReLU(inplace=True),
            # pointwise：1×1
            nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )


class NetAFDV3(nn.Module):
    """
    总参数量 ≈ 3 091  (原始 NetAFD ≈ 80 921 → 缩到 3.8 %)
    采样点与原网络一致：448→224→112→56→28→14→7
    通道表： [3, 6,  6, 12, 12, 12]
    """

    def __init__(self):
        super().__init__()

        # 1) 前两级：stride=1 DW-Sep + AvgPool(2)
        self.stage1 = nn.Sequential(
            DSConv1d(1, 3, k=3, s=1, p=1),  # 448
            nn.AvgPool1d(2)  # 224
        )
        self.stage2 = nn.Sequential(
            DSConv1d(3, 6, k=3, s=1, p=1),  # 224
            nn.AvgPool1d(2)  # 112
        )

        # 2) 后四级：全部 stride=2 DW-Sep
        self.stage3 = DSConv1d(6, 6, k=3, s=2, p=1)  # 112→56
        self.stage4 = DSConv1d(6, 12, k=3, s=2, p=1)  # 56 →28
        self.stage5 = DSConv1d(12, 12, k=3, s=2, p=1)  # 28 →14
        self.stage6 = DSConv1d(12, 12, k=3, s=2, p=1)  # 14 → 7

        # 3) GAP + 极小 FC
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B,12,1)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(12, 1)  # 12 → 1

    def forward(self, x):  # x: (B,1,448)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)

        feat = self.gap(x).squeeze(-1)  # (B, 12)
        out = self.fc(self.drop(feat))  # (B, 1)
        return out, feat


if __name__ == '__main__':
    _channel_in = int(SAMPLE_RATE / 50)
    _channel_in = ((_channel_in // 32) + 1) * 32

    input_tensor = torch.randn(1, 1, _channel_in)
    macs, params = profile(NetAFD(), inputs=(input_tensor,))
    print(f"MACs: {macs}, Parameters: {params}")
    model = NetAFD()
    input_tensor = torch.randn(1, 1, _channel_in)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        output, feat = model(input_tensor)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference Time (CPU): {inference_time_ms:.3f} ms")
