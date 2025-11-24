# FILE: tsne_visualization.py

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.macros import MIN_VAL_TH

# -- 健壮的路径修正方案 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from cores.nets import NetAFDAE, NetAFDAE_Mem_Flow, NetAFDAE_UNet, NetAFDAE_Mem, NetAFDAE_UNet_Mem
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset

# -- 提早检查依赖库 --
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用预训练的AutoEncoder模型提取特征，并通过t-SNE进行交互式3D可视化。")
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/ae_best.pt",
        help="预训练的AutoEncoder模型 (.pt 文件) 的路径。")
    parser.add_argument(
        '--data_path',
        type=str,
        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3",
        help="HDF5数据文件 (例如, test_data.h5) 的路径。")
    parser.add_argument(
        '--ae_model_type',
        type=str,
        default='ae',
        choices=['ae', 'unet', 'mem-ae', 'unet-mem', 'mem-flow-ae'],
        help="要加载的自编码器模型架构类型。")
    parser.add_argument(
        '--output_file',
        type=str,
        default="/home/manu/tmp/afdd_ae_results/tsne_visualization.html",
        help="用于保存t-SNE交互式3D散点图的输出HTML文件路径。")
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="运行评估的设备，例如 'cpu' or 'cuda:0'。")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help="处理数据时使用的批大小。")
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help="t-SNE算法的perplexity参数。")
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=24,
        help="对Hard Normal样本进行K-Means聚类的簇数量。")
    parser.add_argument(
        '--max_samples',
        type=int,
        default=-1,
        help="用于t-SNE可视化的最大样本数。设为-1表示使用所有样本。")
    parser.add_argument(
        '--error_threshold',
        type=float,
        default=0.003125,
        help="重构误差阈值，用于高亮显示'难重构'的正常样本。")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="在浏览器中直接打开生成的交互式t-SNE图。")
    return parser.parse_args()


def plot_tsne_3d(tsne_data, tsne_centers, labels, recon_errors, hard_norm_cluster_labels,
                 output_path, error_threshold, title='Latent Space 3D t-SNE Visualization', show_plot=False):
    """
    使用Plotly生成并保存一个可交互的3D t-SNE散点图。
    修改功能:
    1. 简单正常样本 (Blue)
    2. 故障样本 (Red)
    3. 困难正常样本 (Yellow/Orange) - 根据K-Means结果区分颜色
    4. 聚类中心 (Black X)
    """
    fig = go.Figure()

    # --- 1. 分离正例（故障）样本 ---
    pos_indices = np.where(labels == 1)[0]

    # --- 2. 将正常样本根据重构误差分为两组 ---
    # 首先获取所有正常样本的索引
    all_neg_indices = np.where(labels == 0)[0]

    # 在正常样本中，根据误差阈值筛选“简单”和“困难”的样本
    neg_errors = recon_errors[all_neg_indices]
    easy_neg_indices = all_neg_indices[neg_errors <= error_threshold]
    hard_neg_indices = all_neg_indices[neg_errors > error_threshold]

    logging.info("-" * 30)
    logging.info(f"绘图点数统计:")
    logging.info(f"  - 正常 (误差 <= {error_threshold:.4f}): {len(easy_neg_indices)} 个")
    logging.info(f"  - 正常 (误差 >  {error_threshold:.4f}): {len(hard_neg_indices)} 个 (聚类目标)")
    logging.info(f"  - 故障: {len(pos_indices)} 个")
    logging.info("-" * 30)

    # --- 3. 依次绘制三个类别的点 ---
    # 绘制“简单”的正常样本（蓝色）
    fig.add_trace(go.Scatter3d(
        x=tsne_data[easy_neg_indices, 0],
        y=tsne_data[easy_neg_indices, 1],
        z=tsne_data[easy_neg_indices, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.5,
        ),
        name=f'Normal (Easy, err &le; {error_threshold:.4f})'
    ))

    # 绘制“困难”的正常样本（基于K-Means聚类结果绘制不同深浅的黄色/橙色）
    unique_clusters = np.unique(hard_norm_cluster_labels) if hard_norm_cluster_labels is not None else []

    # --- 调整绘制顺序：先画聚类中心 (Black)，作为背景 ---
    if tsne_centers is not None and len(tsne_centers) > 0:
        fig.add_trace(go.Scatter3d(
            x=tsne_centers[:, 0],
            y=tsne_centers[:, 1],
            z=tsne_centers[:, 2],
            mode='markers+text',
            marker=dict(
                size=3,  # 稍微改小一点，避免完全遮挡
                color='black',
                symbol='circle',
                opacity=0.6,  # 稍微透明一点
                line=dict(width=0)
            ),
            text=[f'C{i}' for i in range(len(tsne_centers))],
            textposition="top center",
            name='Cluster Centers'
        ))

    for i, c_id in enumerate(unique_clusters):
        # 获取属于该簇的原始索引（注意：hard_norm_cluster_labels 的顺序对应 hard_neg_indices）
        current_cluster_mask = (hard_norm_cluster_labels == c_id)
        data_indices = hard_neg_indices[current_cluster_mask]

        # 绘制连接样本到聚类中心的连线
        if tsne_centers is not None and len(tsne_centers) > c_id:
            center_pt = tsne_centers[c_id]
            cluster_pts = tsne_data[data_indices]
            # 构造线段数据: Point -> Center -> None (断开)
            line_x, line_y, line_z = [], [], []
            for pt in cluster_pts:
                line_x.extend([pt[0], center_pt[0], None])
                line_y.extend([pt[1], center_pt[1], None])
                line_z.extend([pt[2], center_pt[2], None])

            fig.add_trace(go.Scatter3d(
                x=line_x, y=line_y, z=line_z,
                mode='lines',
                line=dict(color='rgba(255, 215, 0, 0.3)', width=1),  # 半透明淡黄色连线
                hoverinfo='skip', showlegend=False
            ))

        fig.add_trace(go.Scatter3d(
            x=tsne_data[data_indices, 0],
            y=tsne_data[data_indices, 1],
            z=tsne_data[data_indices, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='yellow',  # 统一使用黄色
                opacity=0.9,
                symbol='circle',
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=f'Hard Normal - Cluster {c_id}'
        ))

    # 绘制故障样本（红色）
    fig.add_trace(go.Scatter3d(
        x=tsne_data[pos_indices, 0],
        y=tsne_data[pos_indices, 1],
        z=tsne_data[pos_indices, 2],
        mode='markers',
        marker=dict(
            size=3.5,
            color='red',
            opacity=0.8,
        ),
        name='Positive (Fault)'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Sample Type'
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存为HTML文件
    fig.write_html(output_path)
    logging.info(f"交互式t-SNE图已保存至: {output_path}")

    # 如果用户选择，则在浏览器中打开
    if show_plot:
        logging.info("在默认浏览器中打开交互式图表...")
        fig.show()


def main():
    """主执行函数"""
    setup_logging()
    args = parse_args()

    if not PLOTLY_AVAILABLE:
        logging.error("Plotly 未安装。请使用 'pip install plotly' 命令安装。")
        return
    if not SKLEARN_AVAILABLE:
        logging.error("scikit-learn 未安装。请使用 'pip install scikit-learn' 命令安装。")
        return

    # --- 1. 环境设置 ---
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    logging.info(f"使用设备: {device}")

    if not os.path.exists(args.model_path):
        logging.error(f"模型文件未找到: {args.model_path}")
        return
    if not os.path.exists(args.data_path):
        logging.error(f"数据文件未找到: {args.data_path}")
        return

    # --- 2. 加载模型 ---
    model_map = {
        'ae': NetAFDAE,
        'unet': NetAFDAE_UNet,
        'mem-ae': NetAFDAE_Mem,
        'unet-mem': NetAFDAE_UNet_Mem,
        'mem-flow-ae': NetAFDAE_Mem_Flow
    }
    if args.ae_model_type not in model_map:
        logging.error(f"不支持的模型类型: {args.ae_model_type}")
        return

    try:
        model_class = model_map[args.ae_model_type]
        model = model_class()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"成功从 {args.model_path} 加载模型")
    except Exception as e:
        logging.error(f"初始化或加载模型失败: {e}")
        return

    # --- 3. 准备数据集 ---
    features_generator = FeaturesGeneratorCNN()
    dataset = HDF5PeakAlignedDataset(
        hdf5_file_path=args.data_path,
        transform=features_generator.transform_sample_ae,
        seq_len=features_generator.seq_len,
        min_delta=MIN_VAL_TH
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"已加载数据集，包含 {len(dataset)} 个样本。")
    if args.max_samples != -1:
        logging.info(f"将使用最多 {args.max_samples} 个样本进行可视化。")

    # --- 4. 提取特征和重构误差 ---
    all_features = []
    all_labels = []
    all_recon_errors = []
    total_samples_processed = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="正在提取特征"):
            if args.max_samples != -1 and total_samples_processed >= args.max_samples:
                break

            inputs = inputs.to(device)

            # 模型输出约定: (reconstructions, latent_vectors, ...)
            model_outputs = model(inputs)
            reconstructions = model_outputs[0]
            latent_vectors = model_outputs[1]

            # 计算每个样本的均方重构误差
            recon_errors = torch.mean((inputs - reconstructions) ** 2, dim=tuple(range(1, inputs.dim())))

            # 将特征向量展平为 [N, feature_dim]
            if latent_vectors.dim() > 2:
                latent_vectors = latent_vectors.view(latent_vectors.size(0), -1)

            all_features.append(latent_vectors.cpu())
            all_labels.append(labels.cpu())
            all_recon_errors.append(recon_errors.cpu())
            total_samples_processed += inputs.size(0)

    # 拼接所有批次的数据
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    errors_tensor = torch.cat(all_recon_errors, dim=0)

    # 如果设置了max_samples，需截断数据
    if args.max_samples != -1 and features_tensor.shape[0] > args.max_samples:
        features_tensor = features_tensor[:args.max_samples]
        labels_tensor = labels_tensor[:args.max_samples]
        errors_tensor = errors_tensor[:args.max_samples]

    features_np = features_tensor.numpy()
    # labels通常是 [N, 1]，需展平为 [N]
    labels_np = labels_tensor.numpy().flatten()
    errors_np = errors_tensor.numpy()

    logging.info(f"特征提取完成，共 {features_np.shape[0]} 个样本，特征维度为 {features_np.shape[1]}。")

    # --- 5. 执行 t-SNE 并绘图 ---
    # 准备聚类数据
    logging.info("正在处理 Hard Normal 样本聚类...")

    all_neg_indices = np.where(labels_np == 0)[0]
    neg_errors = errors_np[all_neg_indices]
    hard_neg_indices = all_neg_indices[neg_errors > args.error_threshold]

    kmeans_centers = None
    hard_cluster_labels = None
    tsne_input = features_np

    if len(hard_neg_indices) >= args.n_clusters:
        # 仅对 Hard Normal 进行聚类
        hard_features = features_np[hard_neg_indices]
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
        hard_cluster_labels = kmeans.fit_predict(hard_features)
        kmeans_centers = kmeans.cluster_centers_

        # 将聚类中心拼接到数据末尾，以便一起进行 t-SNE 变换
        tsne_input = np.vstack([features_np, kmeans_centers])
        logging.info(f"已完成 K-Means (k={args.n_clusters})，将 {kmeans_centers.shape[0]} 个中心加入 t-SNE 计算。")
    else:
        logging.warning("Hard Normal 样本数量少于聚类数，跳过聚类步骤。")

    if tsne_input.shape[0] <= args.perplexity:
        new_perplexity = max(1, tsne_input.shape[0] - 1)
        logging.warning(f"样本数 ({features_np.shape[0]}) 小于或等于 perplexity ({args.perplexity})。")
        logging.warning(f"将 perplexity 自动调整为 {new_perplexity}。")
        args.perplexity = new_perplexity

    if args.perplexity > 0:
        logging.info("开始执行t-SNE变换 (这可能需要一些时间)...")
        tsne = TSNE(
            n_components=3,
            perplexity=args.perplexity,
            random_state=42,
            n_iter=1000,
            n_jobs=-1,  # 使用所有可用的CPU核心
            verbose=1
        )
        tsne_all_results = tsne.fit_transform(tsne_input)
        logging.info("t-SNE变换完成。")

        # 如果进行了聚类，需要拆分数据点和聚类中心
        if kmeans_centers is not None:
            tsne_data_points = tsne_all_results[:-len(kmeans_centers)]
            tsne_centers_points = tsne_all_results[-len(kmeans_centers):]
        else:
            tsne_data_points = tsne_all_results
            tsne_centers_points = None

        plot_tsne_3d(tsne_data_points, tsne_centers_points, labels_np, errors_np, hard_cluster_labels,
                     args.output_file, error_threshold=args.error_threshold, show_plot=args.show_plot)
    else:
        logging.error("无法执行t-SNE，因为样本数量过少 (<=1)。")

    logging.info("-" * 40)
    logging.info("可视化脚本执行完毕。")
    logging.info("-" * 40)


if __name__ == '__main__':
    main()
