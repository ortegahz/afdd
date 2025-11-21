# FILE: visualize_peaks.py

import argparse
import logging
import os
import sys

import h5py
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# -- 健壮的路径修正方案，确保能找到项目模块 (如果需要引用项目内的常量) --
try:
    # 假设此脚本在项目的某个子目录中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.macros import MIN_VAL_TH  # 尝试导入默认阈值
except (ImportError, ModuleNotFoundError):
    MIN_VAL_TH = 100  # 如果导入失败，设置一个默认的回退阈值
    print(f"提示：无法从项目导入 MIN_VAL_TH，使用默认值 {MIN_VAL_TH}")

try:
    import plotly.graph_objects as go
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
        description="交互式可视化工具：展示原始信号并在其上标记检测到的峰值。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v1",
        help="包含数据的 HDF5 文件路径。")
    parser.add_argument(
        '--output_file',
        type=str,
        default="/home/manu/tmp/afdd_ae_results/peak_detection_viz.html",
        help="输出的 HTML 文件路径。")
    parser.add_argument(
        '--max_samples',
        type=int,
        default=64,
        help="可视化样本的最大数量。由于包含大量数据点，建议不要设置过大。")
    parser.add_argument(
        '--view_len',
        type=int,
        default=8192 * 64,
        help="每个样本可视化的最大长度（点数）。如果信号过长，将被截断以保证浏览器性能。")
    parser.add_argument(
        '--min_delta',
        type=float,
        default=float(MIN_VAL_TH),
        help="峰值检测的 prominence (突起幅度) 阈值。")
    parser.add_argument(
        '--distance',
        type=int,
        default=100,
        help="峰值检测的最小间距 (distance)。")
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help="生成后直接在浏览器中打开。")
    return parser.parse_args()


def create_peak_plot(data_list, output_path, show_plot=False):
    """
    创建交互式 Plotly 图表。
    data_list: list of dict, 每个元素包含 {'key', 'signal', 'peaks', 'label'}
    """
    if not data_list:
        logging.warning("没有数据可供绘图。")
        return

    num_samples = len(data_list)

    # --- 1. 初始化 Figure ---
    fig = go.Figure()

    # --- 2. 构建 Frames (用于动画/滑块切换) ---
    frames = []

    for i, item in enumerate(data_list):
        signal = item['signal']
        peaks = item['peaks']
        key_name = item['key']
        label = item['label']

        # 标签文本与颜色
        label_text = "Fault" if label > 0 else "Normal"
        color_line = 'red' if label > 0 else 'royalblue'

        # 构造每一帧的数据
        # Trace 0: 原始信号 (Line)
        trace_signal = go.Scatter(
            y=signal,
            mode='lines',
            line=dict(color=color_line, width=1.5),
            name='Signal'
        )

        # Trace 1: 峰值标记 (Markers)
        # peaks 是索引，y值需要从 signal 中取
        peak_y = signal[peaks]
        trace_peaks = go.Scatter(
            x=peaks,
            y=peak_y,
            mode='markers',
            marker=dict(symbol='x', size=10, color='orange', line=dict(width=2, color='black')),
            name='Detected Peaks'
        )

        frame = go.Frame(
            data=[trace_signal, trace_peaks],
            name=str(i),
            layout=go.Layout(
                title_text=f"Sample [{i + 1}/{num_samples}] Key: {key_name} ({label_text}) | Peaks Found: {len(peaks)}"
            )
        )
        frames.append(frame)

    fig.frames = frames

    # --- 3. 设置初始显示内容 (使用第0个样本) ---
    first_item = data_list[0]
    fig.add_trace(go.Scatter(
        y=first_item['signal'],
        mode='lines',
        line=dict(color='red' if first_item['label'] > 0 else 'royalblue'),
        name='Signal'
    ))

    fig.add_trace(go.Scatter(
        x=first_item['peaks'],
        y=first_item['signal'][first_item['peaks']],
        mode='markers',
        marker=dict(symbol='x', size=10, color='orange', line=dict(width=2, color='black')),
        name='Detected Peaks'
    ))

    # --- 4. 配置布局与控件 ---
    def frame_args(duration):
        return {
            "frame": {"duration": duration, "redraw": True},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    fig.update_layout(
        title=f"Signal & Peak Detection Visualization<br>Sample 1/{num_samples} Key: {first_item['key']}",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=600,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, frame_args(500)]},  # 播放速度稍慢以便观察
                {"label": "Pause", "method": "animate", "args": [[None], frame_args(0)]},
            ],
            "direction": "left", "pad": {"r": 10, "t": 87},
            "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20}, "prefix": "Sample: ", "visible": True, "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9, "x": 0.1, "y": 0,
            "steps": [
                {"label": f"{i + 1}", "method": "animate", "args": [[str(i)], frame_args(0)]}
                for i in range(num_samples)
            ]
        }]
    )

    # --- 5. 保存 ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logging.info("正在生成 HTML 文件...")
    fig.write_html(output_path)
    logging.info(f"可视化结果已保存至: {output_path}")

    if show_plot:
        fig.show()


def main():
    setup_logging()
    args = parse_args()

    if not os.path.exists(args.data_path):
        logging.error(f"文件未找到: {args.data_path}")
        return

    logging.info(f"读取 HDF5 文件: {args.data_path}")
    logging.info(f"峰值检测参数: Prominence (min_delta)={args.min_delta}, Distance={args.distance}")

    visual_data = []

    # --- 读取数据与检测峰值 ---
    with h5py.File(args.data_path, 'r') as f:
        keys = list(f.keys())
        # 如果需要随机抽样，可以在这里 shuffle keys
        # import random; random.shuffle(keys)

        count = 0
        with tqdm(total=min(len(keys), args.max_samples), desc="Processing Signals") as pbar:
            for key in keys:
                if count >= args.max_samples:
                    break

                group = f[key]
                # 获取原始信号，并根据 view_len 截断，避免浏览器渲染卡顿
                raw_signal = group['signal'][:]
                label_seq = group['label_seq'][:]

                # 确定是否为故障样本 (如果序列中包含 >0 的值)
                is_fault = 1 if np.any(label_seq > 0) else 0

                # 截断用于显示的长度
                display_signal = raw_signal[:args.view_len]

                # 使用 Scipy 查找峰值
                # 注意：我们在截断后的信号上查找峰值，以确保标记与显示的波形对应
                peaks, properties = find_peaks(
                    display_signal,
                    prominence=args.min_delta,
                    distance=args.distance
                )

                visual_data.append({
                    'key': key,
                    'signal': display_signal,
                    'peaks': peaks,
                    'label': is_fault
                })

                count += 1
                pbar.update(1)

    create_peak_plot(visual_data, args.output_file, args.show_plot)


if __name__ == '__main__':
    main()
