# FILE: visualize_attention_animation.py
# (Originally visualize_entropy.py, modified to create animations)

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
        description="使用预训练的Phase-2 AutoEncoder模型生成注意力权重分布的动画。")
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
        default="/home/manu/tmp/afdd_ae_results/attention_animation.html",
        help="用于保存注意力权重分布动画的输出HTML文件路径。")
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
        default=200,
        help="用于动画的最大样本数。注意：大数值会产生非常大的HTML文件。")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="在浏览器中直接打开生成的交互式图。")
    return parser.parse_args()


def plot_attention_animation(attentions, labels, output_path, title='Attention Weight Animation', show_plot=False):
    """
    使用Plotly生成并保存一个可交互的注意力权重分布动画。
    """
    num_samples, mem_dim = attentions.shape
    x_axis = np.arange(mem_dim)

    # --- 1. 创建基础图形（显示第一个样本） ---
    initial_label = labels[0]
    initial_color = 'red' if initial_label == 1 else 'blue'
    initial_label_text = 'Fault' if initial_label == 1 else 'Normal'

    fig = go.Figure(
        data=[go.Bar(x=x_axis, y=attentions[0], marker_color=initial_color, name='Attention Weight')],
        layout=go.Layout(
            title=f"{title}<br>Sample 1 ({initial_label_text})",
            xaxis_title="Memory Slot Index",
            yaxis_title="Attention Weight",
            yaxis_range=[0, np.max(attentions) * 1.1]  # 固定Y轴范围以便于比较
        ),
        frames=[go.Frame(
            data=[go.Bar(y=att.flatten(), marker_color='red' if lab == 1 else 'blue')],
            name=str(i),
            layout=go.Layout(title_text=f"{title}<br>Sample {i + 1} ({'Fault' if lab == 1 else 'Normal'})")
        ) for i, (att, lab) in enumerate(zip(attentions, labels))]
    )

    # --- 2. 配置动画控件（播放/暂停按钮和滑块） ---
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, frame_args(50)]},
                {"label": "Pause", "method": "animate", "args": [[None], frame_args(0)]},
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "label": f"Sample {i + 1}",
                    "method": "animate",
                    "args": [[str(i)], frame_args(0)],
                }
                for i in range(num_samples)
            ],
            "pad": {"t": 30, "b": 10},
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
            "len": 0.9,
        }]
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存为HTML文件
    logging.info(f"正在生成动画HTML文件，这可能需要一些时间，取决于样本数量...")
    fig.write_html(output_path)
    logging.info(f"交互式注意力动画已保存至: {output_path}")

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
        if args.max_samples > 1000:
            logging.warning(f"选择的样本数 ({args.max_samples}) 非常大，生成的动画文件可能会很大且加载缓慢。")
        logging.info(f"将使用最多 {args.max_samples} 个样本进行可视化。")

    # --- 4. 提取注意力权重 ---
    all_attentions = []
    all_labels = []
    total_samples_processed = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="正在计算注意力权重"):
            if args.max_samples != -1 and total_samples_processed >= args.max_samples:
                break

            inputs = inputs.to(device)

            # 核心计算逻辑
            # 确保我们只取 latent vector (z), 兼容返回 tuple 的情况 (例如 UNet)
            output = base_model.encode(inputs)
            latents = output[0] if isinstance(output, tuple) else output

            attention_weights = memory_head(latents)

            all_attentions.append(attention_weights.cpu())
            all_labels.append(labels.cpu())
            total_samples_processed += inputs.size(0)

    # 拼接所有批次的数据
    attentions_tensor = torch.cat(all_attentions, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 如果设置了max_samples，需截断数据
    if args.max_samples != -1 and attentions_tensor.shape[0] > args.max_samples:
        attentions_tensor = attentions_tensor[:args.max_samples]
        labels_tensor = labels_tensor[:args.max_samples]

    attentions_np = attentions_tensor.numpy()
    labels_np = labels_tensor.numpy().flatten()

    if attentions_np.shape[0] == 0:
        logging.error("未能提取任何样本的注意力权重，无法生成动画。请检查数据和模型。")
        return

    logging.info(f"权重计算完成，共处理 {attentions_np.shape[0]} 个样本。")

    # --- 5. 绘图 ---
    plot_attention_animation(attentions_np, labels_np, args.output_file, show_plot=args.show_plot)

    logging.info("-" * 40)
    logging.info("动画生成脚本执行完毕。")
    logging.info("-" * 40)


if __name__ == '__main__':
    main()
