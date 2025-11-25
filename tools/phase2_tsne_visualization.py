# FILE: phase2_tsne_visualization.py

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.macros import MIN_VAL_TH

# -- 路径修正方案 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset
# 假设 MemoryHead 定义在 cores.classifier 中，如果不是请修改此处引用
from cores.classifier import MemoryHead

# -- 检查依赖 --
try:
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 可视化：提取经过MLP和归一化后的特征，并展示Memory Slots关系。")

    # 模型路径参数
    parser.add_argument(
        '--path_ckpt_base',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_base.pt",
        help="Phase 2 训练后的 Base AutoEncoder 模型路径。")
    parser.add_argument(
        '--path_ckpt_head',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_head.pt",
        help="Phase 2 训练后的 Memory Head 模型路径。")

    # 常用参数
    parser.add_argument(
        '--data_path',
        type=str,
        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3",
        help="HDF5数据文件路径")
    parser.add_argument(
        '--ae_model_type',
        type=str,
        default='ae',
        choices=['ae', 'unet'],
        help="Base AE的模型架构类型")
    parser.add_argument(
        '--output_file',
        type=str,
        default="/home/manu/tmp/afdd_ae_results/phase2_tsne_result.html",
        help="输出HTML文件路径")
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="计算设备")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help="批大小")
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help="t-SNE perplexity")
    parser.add_argument(
        '--max_samples',
        type=int,
        default=-1,
        help="最大采样数，-1为全部")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="直接打开浏览器显示")

    return parser.parse_args()


def plot_phase2_tsne(tsne_data, num_memory_slots, labels, recon_errors,
                     output_path, title='Phase 2 Feature Space (After Residual MLP)', show_plot=False):
    """
    绘制 Phase 2 特征。
    tsne_data: 包含了 [样本特征; Memory Slots] 的t-SNE结果
    """
    fig = go.Figure()

    # 拆分数据：样本 vs 记忆槽
    # tsne_data 的最后 num_memory_slots 行是 Memory Slots
    tsne_samples = tsne_data[:-num_memory_slots]
    tsne_memory = tsne_data[-num_memory_slots:]

    # --- 1. 绘制 Memory Slots (作为背景参考点) ---
    fig.add_trace(go.Scatter3d(
        x=tsne_memory[:, 0],
        y=tsne_memory[:, 1],
        z=tsne_memory[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='black',
            symbol='diamond',  # 菱形表示 Memory Slots
            opacity=0.8,
        ),
        name='Memory Slots (Centers)'
    ))

    # --- 2. 绘制样本 ---
    # 分离正常(0)和异常(1)
    neg_indices = np.where(labels == 0)[0]
    pos_indices = np.where(labels == 1)[0]

    # 根据重构误差给正常样本上色 (可选：这里简单用深蓝表示正常)
    # 或者可以使用颜色条表示离最近Memory Slot的距离，这里为了清晰使用离散颜色

    # 正常样本
    fig.add_trace(go.Scatter3d(
        x=tsne_samples[neg_indices, 0],
        y=tsne_samples[neg_indices, 1],
        z=tsne_samples[neg_indices, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='dodgerblue',
            opacity=0.5,
        ),
        name='Normal Samples'
    ))

    # 异常样本
    fig.add_trace(go.Scatter3d(
        x=tsne_samples[pos_indices, 0],
        y=tsne_samples[pos_indices, 1],
        z=tsne_samples[pos_indices, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.8,
            symbol='circle-open'
        ),
        name='Abnormal Samples'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant')
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path)
    logging.info(f"图表已保存: {output_path}")

    if show_plot:
        fig.show()


def main():
    setup_logging()
    args = parse_args()

    if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
        logging.error("缺少 Plotly 或 Scikit-learn。")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on {device}")

    # --- 1. 加载 Base 模型 ---
    model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
    if args.ae_model_type not in model_map:
        raise ValueError(f"Unknown ae_type: {args.ae_model_type}")

    base_model = model_map[args.ae_model_type]()

    # 处理 DDP 保存的 keys ('module.' 前缀)
    state_dict_base = torch.load(args.path_ckpt_base, map_location=device)
    new_state_dict = {}
    for k, v in state_dict_base.items():
        new_state_dict[k.replace('module.', '')] = v
    base_model.load_state_dict(new_state_dict)

    base_model.to(device)
    base_model.eval()

    # 动态获取 latent_dim，用于初始化 MemoryHead
    # 构造一个 dummy input forward 一次
    dummy_input = torch.randn(1, 1, 448).to(device)  # 假设长度448，或者从 FeaturesGeneratorCNN 拿
    with torch.no_grad():
        _, dummy_latent, _ = base_model(dummy_input)
        latent_dim = dummy_latent.shape[1]
    logging.info(f"Detected Latent Dim: {latent_dim}")

    # --- 2. 加载 Memory Head ---
    # 必须确保这里的参数与训练时使用的参数完全一致
    memory_head = MemoryHead(
        latent_dim=latent_dim,
        mem_dim=1024,
        hidden_dim=latent_dim  # <--- 修正：与训练脚本保持一致
    )

    state_dict_head = torch.load(args.path_ckpt_head, map_location=device)
    # 处理 DDP keys
    new_state_dict_head = {}
    for k, v in state_dict_head.items():
        new_state_dict_head[k.replace('module.', '')] = v
    memory_head.load_state_dict(new_state_dict_head)

    memory_head.to(device)
    memory_head.eval()
    logging.info("此模型 Phase 2 加载完成 (Base + Head)。")

    # --- 3. 数据集 ---
    gen = FeaturesGeneratorCNN()
    dataset = HDF5PeakAlignedDataset(
        hdf5_file_path=args.data_path,
        transform=gen.transform_sample_ae,
        seq_len=gen.seq_len,
        min_delta=MIN_VAL_TH
    )
    # Phase 2 训练时通常只用了 Normal，但可视化我们希望看 Abnormal 分得开不开
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- 4. 提取特征 (Apply MLP & Normalize) ---
    z_features_list = []
    labels_list = []
    recon_errors_list = []

    total_samples = 0
    logging.info("开始提取特征 (Base -> Latent -> Residual MLP)...")

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            if args.max_samples > 0 and total_samples >= args.max_samples:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 1. Base Encoder: x -> latent
            recons, latents, _ = base_model(inputs)

            # 计算一下重构误差备用
            errs = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

            # 2. Apply the trained Memory Head (Residual MLP) to get the final features
            z = memory_head(latents)
            z_features_list.append(z.cpu())
            labels_list.append(labels.cpu())
            recon_errors_list.append(errs.cpu())

            total_samples += inputs.size(0)

    z_features = torch.cat(z_features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0).numpy().flatten()
    all_errors = torch.cat(recon_errors_list, dim=0).numpy()

    # --- 过滤: 只保留重构误差 > 0.001 的样本 ---
    filter_th = 0.001
    mask = all_errors > filter_th
    logging.info(f"Filtering samples with error > {filter_th}: keeping {np.sum(mask)} out of {len(all_labels)}")

    z_features = z_features[torch.from_numpy(mask)]
    all_labels = all_labels[mask]
    all_errors = all_errors[mask]

    if len(all_labels) == 0:
        logging.error("No samples found matching the error threshold criteria.")
        return

    # 截断到 max_samples
    if args.max_samples > 0 and len(z_features) > args.max_samples:
        z_features = z_features[:args.max_samples]
        all_labels = all_labels[:args.max_samples]
        all_errors = all_errors[:args.max_samples]

    z_features_np = z_features.numpy()

    # --- 5. 获取 Memory Slots (K-Means 中心点, 不归一化) ---
    mem_matrix = memory_head.memory.detach().cpu()  # Shape: [mem_dim, latent_dim]
    mem_matrix_np = mem_matrix.numpy()

    num_slots = mem_matrix_np.shape[0]
    logging.info(f"Memory Slots 数量: {num_slots}, 样本特征数量: {len(z_features_np)}")

    # --- 6. 拼接映射后的样本特征和 K-Means 中心点进行 t-SNE ---
    tsne_input = np.vstack([z_features_np, mem_matrix_np])

    logging.info(f"开始 t-SNE (Total points: {tsne_input.shape[0]})...")
    tsne = TSNE(
        n_components=3,
        perplexity=args.perplexity,
        n_iter=1000,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    tsne_result = tsne.fit_transform(tsne_input)

    # --- 7. 绘图 ---
    # plot_phase2_tsne(tsne_result, num_slots, all_labels, all_errors, args.output_file,
    #                  args.output_file, show_plot=args.show_plot)
    plot_phase2_tsne(tsne_result, num_slots, all_labels, all_errors, args.output_file, show_plot=args.show_plot)


if __name__ == '__main__':
    main()
