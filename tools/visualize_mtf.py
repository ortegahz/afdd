# FILE: visualize_mtf.py

import argparse
import logging
import os
import sys

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- 健壮的路径修正方案，确保能找到项目模块 --
try:
    # 假设此脚本在项目的某个子目录中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from cores.features_generator import FeaturesGeneratorCNN, HDF5SequentialSliceDataset
except (ImportError, ModuleNotFoundError):
    print("警告：无法从 'cores' 模块导入。请确保此脚本位于正确的项目结构下，或者 'cores' 目录在 PYTHONPATH 中。")
    # 如果找不到模块，程序将无法继续，因为 HDF5SequentialSliceDataset 是核心依赖
    sys.exit(1)

# -- 提早检查核心依赖库 --
try:
    from pyts.image import MarkovTransitionField
except ImportError:
    print("错误：'pyts' 库未安装。请运行 'pip install pyts' 进行安装。")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("错误：'plotly' 库未安装。请运行 'pip install plotly' 进行安装。")
    sys.exit(1)


def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为时序信号生成一个交互式的可视化工具，同步展示原始波形及其马尔科夫转移场 (MTF) 图像。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="/home/manu/tmp/afd_pm_hdf5/test_data.h5",
        help="包含测试数据的 HDF5 文件路径。")
    parser.add_argument(
        '--output_file',
        type=str,
        default="/home/manu/tmp/afdd_ae_results/waveform_mtf_visualization.html",
        help="用于保存交互式HTML可视化的输出文件路径。")
    parser.add_argument(
        '--max_samples',
        type=int,
        default=200,
        help="用于可视化的最大样本数。-1 表示使用所有样本。注意：大数值会产生非常大的HTML文件。")
    parser.add_argument(
        '--n_bins',
        type=int,
        default=8,
        help="用于MTF计算的分箱（quantiles）数量。")
    parser.add_argument(
        '--strategy',
        type=str,
        default='quantile',
        choices=['uniform', 'quantile', 'normal'],
        help="MTF分箱策略。'quantile' 通常是最好的选择。")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help="处理数据时使用的批大小。")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="执行完毕后，在浏览器中直接打开生成的可视化图表。")
    return parser.parse_args()


def create_interactive_plot(waveforms, mtf_images, labels, output_path, show_plot=False):
    """
    使用Plotly创建并保存一个交互式图表，同步显示波形和MTF图像。
    """
    num_samples, seq_len = waveforms.shape

    # --- 1. 创建包含两个子图的基础图形 ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Waveform", "Markov Transition Field (MTF)"),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
    )

    # --- 2. 为每个样本创建一个 'Frame' ---
    frames = []
    for i in range(num_samples):
        label = labels[i]
        label_text = 'Fault' if label == 1 else 'Normal'
        color = 'red' if label == 1 else 'blue'

        frame = go.Frame(
            data=[
                go.Scatter(
                    y=waveforms[i],
                    mode='lines',
                    line=dict(color=color),
                    name='Waveform'
                ),
                go.Heatmap(
                    z=mtf_images[i],
                    colorscale='Viridis',
                    showscale=False,
                )
            ],
            # 动画帧的名称，用于滑块控制
            name=str(i),
            layout=go.Layout(
                title_text=f"Sample {i + 1} / {num_samples} ({label_text})"
            )
        )
        frames.append(frame)

    fig.frames = frames

    # --- 3. 设置初始状态（显示第一个样本）---
    initial_label = labels[0]
    initial_label_text = 'Fault' if initial_label == 1 else 'Normal'
    initial_color = 'red' if initial_label == 1 else 'blue'

    fig.add_trace(
        go.Scatter(y=waveforms[0], mode='lines', line=dict(color=initial_color), name='Waveform'),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=mtf_images[0], colorscale='Viridis', showscale=False),
        row=1, col=2
    )

    # --- 4. 配置动画控件（播放/暂停按钮和滑块） ---
    def frame_args(duration):
        return {
            "frame": {"duration": duration, "redraw": True},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    fig.update_layout(
        title=(
            f"Interactive Waveform and MTF Visualization<br>"
            f"Sample 1 / {num_samples} ({initial_label_text})"
        ),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, frame_args(50)]},
                {"label": "Pause", "method": "animate", "args": [[None], frame_args(0)]},
            ],
            "direction": "left", "pad": {"r": 10, "t": 70},
            "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"label": f"{i + 1}", "method": "animate", "args": [[str(i)], frame_args(0)]}
                for i in range(num_samples)
            ],
            "pad": {"t": 30, "b": 10},
        }]
    )

    # --- 5. 更新子图布局和样式 ---
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    # 翻转MTF图像的Y轴，使(0,0)在左上角
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    # 添加滑窗和缩放功能到波形图
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=100, label="100pts", step="all"),
                    dict(count=200, label="200pts", step="all"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=False),
            type="linear"
        )
    )

    # --- 6. 保存并显示 ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logging.info(f"正在生成交互式HTML文件，这可能需要一些时间...")
    fig.write_html(output_path)
    logging.info(f"交互式可视化已保存至: {output_path}")

    if show_plot:
        logging.info("在默认浏览器中打开图表...")
        fig.show()


def main():
    """主执行函数"""
    setup_logging()
    args = parse_args()

    if not os.path.exists(args.data_path):
        logging.error(f"数据文件未找到: {args.data_path}")
        return

    # --- 1. 准备数据集 ---
    features_generator = FeaturesGeneratorCNN()
    dataset = HDF5SequentialSliceDataset(
        hdf5_file_path=args.data_path,
        transform=features_generator.transform_sample_ae,  # 我们需要原始数据
        seq_len=features_generator.seq_len,
        step=features_generator.seq_len  # 无重叠滑窗
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"已加载数据集，包含 {len(dataset)} 个样本。")

    # --- 2. 提取数据 ---
    all_waveforms, all_labels = [], []
    samples_to_process = args.max_samples if args.max_samples != -1 else len(dataset)
    with tqdm(total=samples_to_process, desc="提取波形数据") as pbar:
        for inputs, labels in loader:
            num_to_take = min(inputs.size(0), samples_to_process - len(all_waveforms))
            if num_to_take <= 0:
                break
            # .squeeze() 移除通道维度，如果存在的话
            all_waveforms.append(inputs[:num_to_take].squeeze().numpy())
            all_labels.append(labels[:num_to_take].numpy())
            pbar.update(num_to_take)

    if not all_waveforms:
        logging.error("未能从数据文件中提取任何样本。")
        return

    waveforms_np = np.concatenate(all_waveforms, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).flatten()

    # --- 3. 计算MTF ---
    logging.info(
        f"正在为 {waveforms_np.shape[0]} 个样本计算MTF图像 (n_bins={args.n_bins}, strategy='{args.strategy}')...")
    mtf = MarkovTransitionField(n_bins=args.n_bins, strategy=args.strategy)
    mtf_images = mtf.fit_transform(waveforms_np)
    logging.info("MTF图像计算完成。")

    # --- 4. 创建可视化 ---
    create_interactive_plot(waveforms_np, mtf_images, labels_np, args.output_file, args.show_plot)
    logging.info("可视化脚本执行完毕。")


if __name__ == '__main__':
    main()
