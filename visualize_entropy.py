import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- 健壮的路径修正方案 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.classifier import MemoryHead
from cores.features_generator import FeaturesGeneratorCNN, HDF5SequentialSliceDataset

# -- 提早检查依赖库 --
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
        description="使用预训练的Phase-2 AutoEncoder模型计算注意力熵，并将其分布可视化。")
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp_v6/phase2_best_base.pt",
        help="预训练的Phase-2模型组件的路径 (例如, '..._base.pt' 或 '..._head.pt')。")
    parser.add_argument(
        '--data_path',
        type=str,
        default="/home/manu/tmp/afd_pm_hdf5/test_data.h5",
        help="HDF5数据文件 (例如, test_data.h5) 的路径。")
    parser.add_argument(
        '--ae_model_type',
        type=str,
        default='ae',
        choices=['mem-ae', 'unet-mem', 'mem-flow-ae', 'ae'],
        help="要加载的记忆自编码器模型架构类型。")
    parser.add_argument(
        '--output_file',
        type=str,
        default="/home/manu/tmp/afdd_ae_results/entropy_distribution.html",
        help="用于保存注意力熵分布交互式图的输出HTML文件路径。")
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
        '--max_samples',
        type=int,
        default=20000,
        help="用于可视化的最大样本数。设为-1表示使用所有样本。")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="在浏览器中直接打开生成的交互式图。")
    return parser.parse_args()


def plot_entropy_distribution(scores, labels, output_path, title='Attention Entropy Distribution', show_plot=False):
    """
    使用Plotly生成并保存一个可交互的注意力熵分布直方图。
    """
    fig = go.Figure()

    # 分离正负样本的得分
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    logging.info(
        f"正例样本 (故障) 数量: {len(pos_scores)}, "
        f"熵均值: {np.mean(pos_scores):.4f}, 标准差: {np.std(pos_scores):.4f}"
    )
    logging.info(
        f"负例样本 (正常) 数量: {len(neg_scores)}, "
        f"熵均值: {np.mean(neg_scores):.4f}, 标准差: {np.std(neg_scores):.4f}"
    )

    # 绘制负例（正常）样本的分布
    fig.add_trace(go.Histogram(
        x=neg_scores,
        name='Normal (label=0)',
        marker_color='blue',
        opacity=0.7,
        histnorm='probability density'  # 归一化以比较分布形状
    ))

    # 绘制正例（故障）样本的分布
    fig.add_trace(go.Histogram(
        x=pos_scores,
        name='Fault (label=1)',
        marker_color='red',
        opacity=0.7,
        histnorm='probability density'
    ))

    fig.update_layout(
        barmode='overlay',  # 叠加直方图
        title=dict(text=title, x=0.5),
        xaxis_title_text='Attention Entropy Score',
        yaxis_title_text='Density',
        legend_title_text='Sample Type',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存为HTML文件
    fig.write_html(output_path)
    logging.info(f"交互式熵分布图已保存至: {output_path}")

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

    # --- 1. 环境与路径设置 ---
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    logging.info(f"使用设备: {device}")

    # 智能判断模型路径
    model_path = args.model_path
    base_model_path, head_model_path = None, None

    if "_base.pt" in model_path:
        base_model_path = model_path
        head_model_path = model_path.replace("_base.pt", "_head.pt")
    elif "_head.pt" in model_path:
        head_model_path = model_path
        base_model_path = model_path.replace("_head.pt", "_base.pt")
    else:
        logging.error("模型路径必须指向一个Phase-2训练的检查点，其文件名应包含 '_base.pt' 或 '_head.pt'。")
        return

    if not os.path.exists(base_model_path) or not os.path.exists(head_model_path):
        logging.error(f"无法找到模型对。请确保基础模型和记忆头模型都存在。")
        logging.error(f"  - 检查的基础模型路径: {base_model_path}")
        logging.error(f"  - 检查的记忆头路径: {head_model_path}")
        return
    if not os.path.exists(args.data_path):
        logging.error(f"数据文件未找到: {args.data_path}")
        return

    # --- 2. 加载模型 ---
    try:
        # 根据 ae_model_type 确定基础模型类
        if 'unet' in args.ae_model_type:
            base_model_class = NetAFDAE_UNet
        else:
            base_model_class = NetAFDAE

        # 加载基础模型
        base_model = base_model_class()
        base_model.load_state_dict(torch.load(base_model_path, map_location=device))
        base_model.to(device)
        base_model.eval()
        logging.info(f"成功从 {base_model_path} 加载基础模型")

        # 加载记忆头
        latent_dim = base_model.get_latent_dim()
        memory_head = MemoryHead(latent_dim=latent_dim)
        memory_head.load_state_dict(torch.load(head_model_path, map_location=device))
        memory_head.to(device)
        memory_head.eval()
        logging.info(f"成功从 {head_model_path} 加载记忆头")

    except Exception as e:
        logging.error(f"初始化或加载模型失败: {e}", exc_info=True)
        return

    # --- 3. 准备数据集 ---
    features_generator = FeaturesGeneratorCNN()
    dataset = HDF5SequentialSliceDataset(
        hdf5_file_path=args.data_path,
        transform=features_generator.transform_sample_ae,
        seq_len=features_generator.seq_len,
        step=features_generator.seq_len  # 无重叠滑窗
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"已加载数据集，包含 {len(dataset)} 个样本。")
    if args.max_samples != -1:
        logging.info(f"将使用最多 {args.max_samples} 个样本进行可视化。")

    # --- 4. 提取注意力熵 ---
    all_scores = []
    all_labels = []
    total_samples_processed = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="正在计算注意力熵"):
            if args.max_samples != -1 and total_samples_processed >= args.max_samples:
                break

            inputs = inputs.to(device)

            # 核心计算逻辑
            latents = base_model.encode(inputs)
            attention_weights = memory_head(latents)

            epsilon = 1e-12
            entropy = -attention_weights * torch.log(attention_weights + epsilon)
            entropy_scores = torch.sum(entropy, dim=1)  # 在记忆单元维度上求和

            all_scores.append(entropy_scores.cpu())
            all_labels.append(labels.cpu())
            total_samples_processed += inputs.size(0)

    # 拼接所有批次的数据
    scores_tensor = torch.cat(all_scores, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 如果设置了max_samples，需截断数据
    if args.max_samples != -1 and scores_tensor.shape[0] > args.max_samples:
        scores_tensor = scores_tensor[:args.max_samples]
        labels_tensor = labels_tensor[:args.max_samples]

    scores_np = scores_tensor.numpy().flatten()
    labels_np = labels_tensor.numpy().flatten()

    logging.info(f"熵计算完成，共处理 {scores_np.shape[0]} 个样本。")

    # --- 5. 绘图 ---
    plot_entropy_distribution(scores_np, labels_np, args.output_file, show_plot=args.show_plot)

    logging.info("-" * 40)
    logging.info("可视化脚本执行完毕。")
    logging.info("-" * 40)


if __name__ == '__main__':
    main()
