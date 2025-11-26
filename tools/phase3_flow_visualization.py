# FILE: phase3_flow_visualization.py

import argparse
import logging
import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- 路径修正 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from cores.nets import NetAFDAE, NetAFDAE_UNet, create_flow_model
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset
from cores.classifier import MemoryHead
from utils.macros import MIN_VAL_TH

# -- 检查依赖 --
try:
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 可视化：评估 Normalizing Flow 的 Latent 变换和概率密度分布。")

    # 模型路径参数 (需要 Base, Head, Flow 三个)
    parser.add_argument(
        '--path_ckpt_base',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp_v11_s2/phase2_best_base.pt",
        help="Base AutoEncoder 模型路径")
    parser.add_argument(
        '--path_ckpt_head',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp_v11_s2/phase2_best_head.pt",
        help="Memory Head 模型路径")
    parser.add_argument(
        '--path_ckpt_flow',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase3_best_flow.pt",
        help="Flow Model 模型路径")
    parser.add_argument(
        '--path_ckpt_proj',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase3_best_proj.pt",
        help="Projection Layer (Linear) 模型路径")

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
        default="/home/manu/tmp/afdd_ae_results/phase3_flow_result.html",
        help="输出HTML文件路径")
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="计算设备")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256)
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help="t-SNE perplexity")
    parser.add_argument(
        '--max_samples',
        type=int,
        default=-1,
        help="为了t-SNE速度，建议限制采样数")
    parser.add_argument(
        '--show_plot',
        default=True,
        help="直接打开浏览器显示")

    return parser.parse_args()


def plot_phase3_analysis(u_embedding, log_probs, labels, dists_min, output_path, show_plot=False):
    """
    绘制 Phase 3 的综合分析图：
    1. Log-Likelihood 直方图 (核心指标)
    2. Transformed Latent Space (u) 的 t-SNE 散点图
    """

    # 分离正常和异常
    neg_mask = (labels == 0)
    pos_mask = (labels == 1)

    # --- 图1: Log-Likelihood 分布直方图 ---
    fig_hist = go.Figure()
    # 使用 -LogProb (NLL) 更直观，值越大越异常
    nll = -log_probs

    fig_hist.add_trace(go.Histogram(
        x=nll[neg_mask],
        name='Normal (Easy+Hard)',
        marker_color='dodgerblue',
        opacity=0.6,
        nbinsx=100
    ))
    fig_hist.add_trace(go.Histogram(
        x=log_probs[pos_mask],
        name='Abnormal',
        marker_color='red',
        opacity=0.6,
        nbinsx=100
    ))
    fig_hist.update_layout(
        title="1. Log-Probability Distribution (Higher is better for Normal)",
        xaxis_title="Log Probability",
        yaxis_title="Count",
        barmode='overlay'
    )

    # --- 图2: Transformed Latent Space (u) 3D t-SNE ---
    # 理论上 Flow 应该把 Normal 变成一个标准的球体 (Standard Gaussian)
    # Abnormal 应该被甩到球体外面
    fig_tsne = go.Figure()

    # 绘制正常样本
    fig_tsne.add_trace(go.Scatter3d(
        x=u_embedding[neg_mask, 0],
        y=u_embedding[neg_mask, 1],
        z=u_embedding[neg_mask, 2],
        mode='markers',
        marker=dict(size=3, color=log_probs[neg_mask], colorscale='Viridis', opacity=0.5,
                    colorbar=dict(title="LogProb")),
        name='Normal'
    ))

    # 绘制异常样本
    fig_tsne.add_trace(go.Scatter3d(
        x=u_embedding[pos_mask, 0],
        y=u_embedding[pos_mask, 1],
        z=u_embedding[pos_mask, 2],
        mode='markers',
        marker=dict(size=4, color='red', symbol='circle-open', opacity=0.8),
        name='Abnormal'
    ))

    fig_tsne.update_layout(
        title="2. Flow Transformed Space (Target: Gaussian Sphere)",
        scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # --- 保存 ---
    # 为了方便，把两个图保存到一个 HTML 文件里的不同 div，或者分开保存
    # 这里我们使用 subplots 可能会比较挤，所以直接写出两个独立的 div 到 HTML

    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("<html><head><title>Phase 3 Flow Analysis</title></head><body>")
        f.write("<h1>Phase 3 Results: Flow Model Analysis</h1>")
        f.write(f"<p>Filtering: Only samples with Recon Error > 0.001 (Hard Examples + Anomalies)</p>")
        f.write(fig_hist.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr>")
        f.write(fig_tsne.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")

    logging.info(f"Analysis saved to: {output_path}")
    if show_plot:
        fig_hist.show()
        fig_tsne.show()


def main():
    setup_logging()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 1. Load Base Model ---
    logging.info("Loading Base AE...")
    model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
    base_model = model_map[args.ae_model_type]()

    state_dict_base = torch.load(args.path_ckpt_base, map_location=device)
    new_state = {k.replace('module.', ''): v for k, v in state_dict_base.items()}
    base_model.load_state_dict(new_state)
    base_model.to(device).eval()

    # Get dimensions
    with torch.no_grad():
        dummy_in = torch.randn(1, 1, 448).to(device)
        _, dummy_z, _ = base_model(dummy_in)
        latent_dim = dummy_z.shape[1]
    logging.info(f"Latent Dim: {latent_dim}")

    # --- 2. Load Memory Head (As MLP Feature Extractor) ---
    logging.info("Loading Memory Head (MLP)...")
    memory_head = MemoryHead(latent_dim=latent_dim, mem_dim=1024, hidden_dim=latent_dim)
    state_dict_head = torch.load(args.path_ckpt_head, map_location=device)
    new_state_head = {k.replace('module.', ''): v for k, v in state_dict_head.items()}
    memory_head.load_state_dict(new_state_head)
    memory_head.to(device).eval()

    # --- 2.5 Load Projection Layer (Linear) ---
    reduced_dim = 16
    logging.info(f"Loading Projection Layer ({latent_dim} -> {reduced_dim})...")
    dim_reduction = torch.nn.Linear(latent_dim, reduced_dim)
    state_dict_proj = torch.load(args.path_ckpt_proj, map_location=device)
    new_state_proj = {k.replace('module.', ''): v for k, v in state_dict_proj.items()}
    dim_reduction.load_state_dict(new_state_proj)
    dim_reduction.to(device).eval()

    # --- 3. Load Flow Model ---
    logging.info("Loading Conditional Flow Model (MoF)...")
    # 注意：输入维度现在是 reduced_dim (16)
    flow_model = create_flow_model(
        latent_dim=reduced_dim,
        hidden_features=reduced_dim * 2,
        context_features=reduced_dim  # <--- Condition 维度也为 16
    )
    state_dict_flow = torch.load(args.path_ckpt_flow, map_location=device)
    new_state_flow = {k.replace('module.', ''): v for k, v in state_dict_flow.items()}
    flow_model.load_state_dict(new_state_flow)
    flow_model.to(device).eval()

    logging.info("All models loaded.")

    # --- 4. Data Loading ---
    gen = FeaturesGeneratorCNN()
    dataset = HDF5PeakAlignedDataset(
        hdf5_file_path=args.data_path,
        transform=gen.transform_sample_ae,
        seq_len=gen.seq_len,
        min_delta=MIN_VAL_TH
    )
    # Loader with shuffle=False for reproducibility
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- [New Pre-calculation] Prepare Global Context (Projected Memory Bank) ---
    logging.info("Projecting Memory Bank to create Global Context...")
    with torch.no_grad():
        # memory_head.memory is [1024, 128] -> Project to [1024, 16]
        projected_memory = dim_reduction(memory_head.memory)

    # --- 5. Inference Loop ---
    u_list, log_prob_list, label_list, error_list, dist_list = [], [], [], [], []
    total_samples = 0

    logging.info("Processing samples...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            if args.max_samples > 0 and total_samples >= args.max_samples:
                break

            inputs = inputs.to(device)

            # A. Base AE
            recons, latents, _ = base_model(inputs)
            recon_err = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

            # B. Memory Head (MLP transformation)
            # We treat the MemoryHead as a fixed feature transform now
            z_mlp = memory_head(latents)

            # B.5 Projection (Linear Reduction)
            z_reduced = dim_reduction(z_mlp)

            # --- [New Logic] Find Nearest Context ---
            # Calculate distance between current batch samples [B, 16] and all memory slots [1024, 16]
            dists = torch.cdist(z_reduced, projected_memory) # Output: [B, 1024]

            # Find index of nearest slot
            min_indices = torch.argmin(dists, dim=1) # [B]
            min_dists_batch = torch.min(dists, dim=1)[0] # [B] Distance value

            # Select the corresponding projected memory vectors as context
            context = projected_memory[min_indices] # [B, 16]

            # C. Flow Model (Conditional Inference)
            # Calculate Log Probability P(z | c)
            log_prob = flow_model.log_prob(inputs=z_reduced, context=context)

            # Calculate Transformed Latent (u) -> Expecting Gaussian Sphere
            try:
                # nflows transform usually takes context as keyword arg
                # transform_to_noise returns (noise, logabsdet)
                u_features, _ = flow_model._transform(inputs=z_reduced, context=context)
            except (AttributeError, TypeError):
                # Fallback: 如果拿不到 u，就画 z_reduced，但颜色用 log_prob
                u_features = z_reduced

            u_list.append(u_features.cpu())
            log_prob_list.append(log_prob.cpu())
            label_list.append(labels.cpu())
            error_list.append(recon_err.cpu())
            dist_list.append(min_dists_batch.cpu())

            total_samples += inputs.size(0)

    # Concat
    u_features = torch.cat(u_list, dim=0).numpy()
    log_probs = torch.cat(log_prob_list, dim=0).numpy()
    labels = torch.cat(label_list, dim=0).numpy().flatten()
    errors = torch.cat(error_list, dim=0).numpy()
    dists_min = torch.cat(dist_list, dim=0).numpy()

    # --- 6. 难例过滤逻辑 (保持与训练一致) ---
    # 我们只关心那些“重构还可以，但在潜空间可能异常”的样本
    # 或者重构很差的异常样本
    filter_th = 0.001
    mask = errors > filter_th
    logging.info(f"Filtering: Keeping {np.sum(mask)}/{len(labels)} samples with Recon Error > {filter_th}")

    u_features = u_features[mask]
    log_probs = log_probs[mask]
    labels = labels[mask]
    # errors = errors[mask]

    if len(labels) == 0:
        logging.error("No samples left after filtering!")
        return

    # --- 7. t-SNE 降维 (对 u_features) ---
    # 如果样本量太大，t-SNE 会很慢，随机采样一下
    if len(u_features) > 5000:
        indices = np.random.choice(len(u_features), 5000, replace=False)
        u_features_sample = u_features[indices]
        log_probs_sample = log_probs[indices]
        labels_sample = labels[indices]
        dists_sample = dists_min[indices]
        logging.info("Subsampling to 5000 points for t-SNE...")
    else:
        u_features_sample = u_features
        log_probs_sample = log_probs
        labels_sample = labels
        dists_sample = dists_min

    if SKLEARN_AVAILABLE:
        logging.info("Running t-SNE on Flow Transformed Features (u)...")
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, verbose=1, random_state=42)
        u_embedding = tsne.fit_transform(u_features_sample)
    else:
        logging.warning("sklearn not found, skipping t-SNE, only plotting histogram.")
        u_embedding = np.zeros((len(u_features_sample), 3))

    # --- 8. 绘图 ---
    plot_phase3_analysis(u_embedding, log_probs_sample, labels_sample, None, args.output_file, args.show_plot)


if __name__ == '__main__':
    main()
