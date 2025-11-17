# FILE: nets.py

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SkipBottleneck(nn.Module):
    """
    在 Skip Connection 上施加一个瓶颈，以限制信息流。
    比喻：在U-Net的“八车道高速公路”上设置一个“收费站/安检站”。
    这强制解码器更多地依赖于来自记忆模块的潜在向量 z，而不是简单地“复制”编码器的特征，
    从而增强模型对异常的敏感度。
    结构: Conv 1x1 (压缩) -> ReLU -> Conv 1x1 (恢复)
    """

    def __init__(self, in_c, bottleneck_ratio=0.25):
        super().__init__()
        bottleneck_c = max(4, int(in_c * bottleneck_ratio))
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_c, bottleneck_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_c, in_c, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.bottleneck(x)


class MemoryModule(nn.Module):
    """
    记忆模块，用于存储正常模式的原型。
    它接受一个查询向量，通过注意力机制从记忆库中检索信息。
    """

    def __init__(self, mem_dim, fea_dim):
        super().__init__()
        self.mem_dim = mem_dim  # 记忆单元数量
        self.fea_dim = fea_dim  # 每个记忆单元的维度 (等于 latent_dim)

        # 可学习的记忆库
        self.memory = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        # 使用 Xavier 初始化保证初始权重分布合理
        nn.init.xavier_uniform_(self.memory)

    def forward(self, z_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_query (torch.Tensor): 编码器输出的查询向量, shape: [B, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z_retrieved: 从记忆库中检索并加权合成的向量, shape: [B, D]
                - attention: 注意力权重, shape: [B, M]
        """
        # 1. 计算注意力权重 (寻址)
        #    - 使用余弦相似度作为度量, F.linear 是高效的矩阵乘法 z_query @ memory.T
        #    - 对输入和记忆库都进行 L2 归一化
        query_norm = F.normalize(z_query, p=2, dim=1)
        memory_norm = F.normalize(self.memory, p=2, dim=1)
        # attention shape: [B, M]
        attention = F.softmax(F.linear(query_norm, memory_norm), dim=1)

        # 2. 从记忆库中检索信息 (读取)
        #    - 用注意力权重对记忆库中的原型进行加权求和
        #    - 使用 torch.matmul(attention, self.memory) 来实现加权求和
        z_retrieved = torch.matmul(attention, self.memory)

        return z_retrieved, attention


def create_flow_model(latent_dim, num_layers=4, hidden_features=64):
    """辅助函数，用于创建 Normalizing Flow 模型 (RealNVP)"""
    try:
        from nflows.flows.base import Flow
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        from nflows.transforms.permutations import RandomPermutation
    except ImportError:
        raise ImportError("Please install nflows: pip install nflows")

    base_dist = StandardNormal(shape=[latent_dim])
    transforms = []
    for _ in range(num_layers):
        transforms.append(RandomPermutation(features=latent_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=latent_dim,
            hidden_features=hidden_features
        ))
    transform = CompositeTransform(transforms)
    return Flow(transform, base_dist)


class NetAFDAE_UNet(nn.Module):
    """
    一个增强版的自编码器，集成了以下特性以提升重构精度：
    1.  **多尺度卷积 (Multi-scale Convolutions)**: 在编码器初期捕捉不同尺度的时序特征。
    2.  **残差连接 (Residual Connections)**: 在编码器和解码器中使用残差块，缓解梯度消失，保留信息。
    3.  **U-Net 风格的跳转连接 (Skip Connections)**: 将编码器的浅层特征直接传递给解码器，帮助恢复高频细节。
    4.  **上采样+卷积 (Upsample + Conv)**: 在解码器中使用，以减少转置卷积可能带来的棋盘格效应。
    """

    def __init__(self, latent_dim=128, bottleneck_ratio=0.25):
        super().__init__()
        self.latent_dim = latent_dim
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

        # --- Skip Connection Bottlenecks (新增) ---
        # 对跳跃连接施加“瓶颈”，限制信息直接流向解码器，迫使其更多地依赖 z 向量。
        # 这可以防止模型简单地“复印”输入，从而增强其对异常的敏感度。
        skip_channels = [16, 32, 64, 128, 256]
        self.skip_bottlenecks = nn.ModuleList([
            SkipBottleneck(c, bottleneck_ratio=bottleneck_ratio) for c in skip_channels
        ])

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

        # 将原始的 skip connection 存储在一个列表中
        skips_raw = [s1, s2, s3, s4, s5]
        # 对每个 skip connection 应用瓶颈层
        skips_bottlenecked = [self.skip_bottlenecks[i](s) for i, s in enumerate(skips_raw)]
        return z, skips_bottlenecked

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor or None]:
        z, skips = self.encode(x)
        reconstructed_x = self.decode(z, skips)
        # 返回一个额外的 None 以统一接口
        return reconstructed_x, z, None

    def get_latent_dim(self):
        return self.latent_dim


class NetAFDAE_UNet_Mem(nn.Module):
    """
    结合了 U-Net 结构和记忆模块的终极版自编码器。
    - 使用 U-Net 的编码器和解码器以获得高质量的重构。
    - 插入记忆模块以强制通过“正常模式”原型进行重构，增强对异常的敏感度。
    """

    def __init__(self, latent_dim=128, mem_dim=2048, bottleneck_ratio=0.25):
        super().__init__()
        # 实例化 U-Net 作为基础，但不直接作为子模块调用，而是复用其组件
        self.base_unet = NetAFDAE_UNet(latent_dim, bottleneck_ratio=bottleneck_ratio)

        # 插入记忆模块
        self.memory_module = MemoryModule(mem_dim=mem_dim, fea_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        # 使用 U-Net 的编码器，返回 z 和 skip connections
        return self.base_unet.encode(x)

    def decode(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        # 使用 U-Net 的解码器
        return self.base_unet.decode(z, skips)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的前向传播。
        返回: (重建的信号, 原始潜在向量, 注意力权重)
        """
        # 1. 编码得到查询向量 z_query 和 skip connections
        z_query, skips = self.encode(x)

        # 2. 通过记忆模块检索得到 z_retrieved 和注意力权重
        z_retrieved, attention = self.memory_module(z_query)

        # 3. 使用检索到的 z_retrieved 和原始的 skips进行解码
        reconstructed_x = self.decode(z_retrieved, skips)

        return reconstructed_x, z_query, attention


class NetAFDAE_Mem(nn.Module):
    """
    基于 NetAFDAE 结构的记忆增强自编码器 (Memory-Augmented Autoencoder)。
    它在编码器和解码器之间插入一个记忆模块，强制模型通过固定的“正常模式”
    原型来重构输入，从而增强对异常的敏感度。
    """

    def __init__(self, latent_dim=128, mem_dim=2048):
        super().__init__()
        # 复用原始的 NetAFDAE 结构作为编码器和解码器的基础
        self.base_ae = NetAFDAE(latent_dim)

        # 插入记忆模块
        self.memory_module = MemoryModule(mem_dim=mem_dim, fea_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_ae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_ae.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的前向传播。
        返回: (重建的信号, 原始潜在向量, 注意力权重)
        """
        # 1. 编码得到查询向量 z_query
        z_query = self.encode(x)
        # 2. 通过记忆模块检索得到 z_retrieved 和注意力权重
        z_retrieved, attention = self.memory_module(z_query)
        # 3. 使用检索到的 z_retrieved 进行解码
        reconstructed_x = self.decode(z_retrieved)

        return reconstructed_x, z_query, attention


class NetAFDAE_Mem_Flow(nn.Module):
    """
    在 MemAE 的基础上，增加了 Normalizing Flow 模型来对潜空间进行概率密度建模。
    这允许模型不仅通过重构误差，还通过潜向量的概率来检测异常。为了解决高频信号重构
    不佳的问题，已将基础架构替换为U-Net，通过跳跃连接保留高频细节。
    """

    def __init__(self, latent_dim=128, mem_dim=2048, bottleneck_ratio=0.25):
        super().__init__()
        # 将基础架构从 NetAFDAE 更换为 NetAFDAE_UNet，以引入跳跃连接，增强高频重构能力
        self.base_ae = NetAFDAE_UNet(latent_dim, bottleneck_ratio=bottleneck_ratio)

        # 插入记忆模块
        self.memory_module = MemoryModule(mem_dim=mem_dim, fea_dim=latent_dim)

        # 插入 Flow 模型
        self.flow_model = create_flow_model(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """编码器现在返回潜向量 z 和跳跃连接列表 skips"""
        return self.base_ae.encode(x)

    def decode(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        """解码器现在需要潜向量 z 和跳跃连接列表 skips"""
        return self.base_ae.decode(z, skips)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的前向传播。
        返回: (重建信号, 原始潜向量, 注意力权重, 对数概率)
        """
        # 1. 编码以获取查询向量 z_query 和跳跃连接
        z_query, skips = self.encode(x)
        # 2. 通过记忆模块检索得到 z_retrieved 和注意力权重
        z_retrieved, attention = self.memory_module(z_query)
        # 3. 使用检索到的 z_retrieved 和跳跃连接进行解码
        reconstructed_x = self.decode(z_retrieved, skips)
        # 4. 计算原始潜向量的对数概率
        log_prob = self.flow_model.log_prob(z_query)
        return reconstructed_x, z_query, attention, log_prob


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
        self.latent_dim = latent_dim
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        完整的前向传播。
        返回: (重建的信号, 潜在向量, None)
        增加一个 None 输出以统一接口。
        """
        z = self.encode(x)
        reconstructed_x = self.decode(z)
        return reconstructed_x, z, None

    def get_latent_dim(self):
        return self.latent_dim


def CBR2D(in_c, out_c, k=3, s=1, p=1):
    """2D Conv-BatchNorm-ReLU Block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


def DeCBR2D(in_c, out_c, k=3, s=2, p=1, op=1):
    """2D Transposed Conv-BatchNorm-ReLU Block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, output_padding=op, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class NetAFDAE_2D_MTF(nn.Module):
    """
    2D Convolutional AutoEncoder for reconstructing Markov Transition Field (MTF) images.
    Input: (B, 1, 64, 64) MTF image
    Output: (B, 1, 64, 64) reconstructed MTF image
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder (64 -> 32 -> 16 -> 8 -> 4) ---
        self.encoder_stage1 = CBR2D(1, 16, s=2, p=1)  # 64x64 -> 32x32
        self.encoder_stage2 = CBR2D(16, 32, s=2, p=1)  # 32x32 -> 16x16
        self.encoder_stage3 = CBR2D(32, 64, s=2, p=1)  # 16x16 -> 8x8
        self.encoder_stage4 = CBR2D(64, 128, s=2, p=1)  # 8x8 -> 4x4

        self.encoder_fc = nn.Linear(128 * 4 * 4, latent_dim)

        # --- Decoder (4 -> 8 -> 16 -> 32 -> 64) ---
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.unflatten = lambda x: x.view(-1, 128, 4, 4)

        self.decoder_stage4 = DeCBR2D(128, 64, s=2, p=1, op=1)  # 4x4 -> 8x8
        self.decoder_stage3 = DeCBR2D(64, 32, s=2, p=1, op=1)  # 8x8 -> 16x16
        self.decoder_stage2 = DeCBR2D(32, 16, s=2, p=1, op=1)  # 16x16 -> 32x32
        self.decoder_stage1 = DeCBR2D(16, 16, s=2, p=1, op=1)  # 32x32 -> 64x64

        self.output_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.encoder_stage1(x)
        x = self.encoder_stage2(x)
        x = self.encoder_stage3(x)
        x = self.encoder_stage4(x)
        x = x.flatten(1)
        return self.encoder_fc(x)

    def decode(self, z):
        x = F.relu(self.decoder_fc(z))
        x = self.unflatten(x)
        x = self.decoder_stage4(x)
        x = self.decoder_stage3(x)
        x = self.decoder_stage2(x)
        x = self.decoder_stage1(x)
        reconstructed_x = self.output_conv(x)
        return torch.sigmoid(reconstructed_x)  # MTF values are probabilities [0, 1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        z = self.encode(x)
        reconstructed_x = self.decode(z)
        return reconstructed_x, z, None

    def get_latent_dim(self):
        return self.latent_dim


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
