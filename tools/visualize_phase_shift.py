# FILE: visualize_phase_shift.py
"""
该脚本修改为：从 HDF5 数据文件中加载一个约1.2个周期的信号样本，
精确估计其周期，然后生成一个无缝、连续的多周期信号，并进行可视化对比。
"""

import argparse
import logging
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- 健壮的路径修正方案，确保能找到项目模块 --
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from cores.features_generator import FeaturesGeneratorCNN, HDF5SequentialSliceDataset
except (ImportError, ModuleNotFoundError):
    print("警告：无法从 'cores' 模块导入。请确保此脚本位于正确的项目结构下，或者 'cores' 目录在 PYTHONPATH 中。")
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
        description="从HDF5加载一个样本，生成一个平滑、连续的多周期信号。",
        epilog="示例运行: python visualize_phase_shift.py --data_path <path_to_h5> --sample_index 10 --num_periods 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path', type=str, default="/home/manu/tmp/afd_pm_hdf5/test_data.h5",
        help="包含测试数据的 HDF5 文件路径。")
    parser.add_argument(
        '--output_file', type=str, default="/home/manu/tmp/afdd_ae_results/continuous_signal_generation.html",
        help="用于保存交互式HTML可视化的输出文件路径。")
    parser.add_argument(
        '--sample_index', type=int, default=0,
        help="要从HDF5文件中加载并进行处理的样本索引。")
    parser.add_argument(
        '--num_periods', type=int, default=5,
        help="要生成的目标信号周期数。")
    # parser.add_argument(
    #     '--fs', type=float, default=20000.0,
    #     help="信号的采样率 (Hz)。(暴力搜索方法不需要此参数)")
    parser.add_argument(
        '--show_plot', action='store_true',
        help="执行完毕后，在浏览器中直接打开生成的可视化图表。")
    return parser.parse_args()


def estimate_period_brute_force(x, T_min_ratio=0.2, T_max_ratio=0.9):
    """
    通过暴力搜索（归一化互相关）来估计信号的周期长度。

    Args:
        x (np.array): 输入信号。
        T_min_ratio (float): 相对于信号长度的最小周期搜索比例。
        T_max_ratio (float): 相对于信号长度的最大周期搜索比例。

    Returns:
        int: 估计的最佳周期长度（采样点数）。
    """
    n = len(x)
    T_min = int(n * T_min_ratio)
    T_max = int(n * T_max_ratio)

    if T_min < 10: T_min = 10  # 保证最小搜索周期不小于10个样本点
    if T_max <= T_min:
        logging.warning(f"周期搜索范围无效 (T_min={T_min}, T_max={T_max})，返回默认周期。")
        return n // 2

    # 去除直流分量以获得更准确的相关性
    x_norm = x - np.mean(x)

    best_corr = -1
    best_T = T_min

    logging.info(f"开始在 [{T_min}, {T_max}] 范围内暴力搜索周期 T...")
    for T in range(T_min, T_max):
        sig1 = x_norm[:n - T]
        sig2 = x_norm[T:]
        # 使用np.corrcoef计算归一化相关系数，更稳健
        corr = np.corrcoef(sig1, sig2)[0, 1]

        if corr > best_corr:
            best_corr = corr
            best_T = T

    logging.info(f"通过暴力搜索估计的最佳周期: {best_T} 个采样点 (相关性: {best_corr:.3f})。")
    return best_T


def generate_continuous_signal(x, num_periods_to_generate, crossfade_ratio=0.1):
    """
    从一个不完整周期的样本中，生成一个平滑、连续的多周期信号。
    """
    # 1. 估计单个周期的精确长度
    period_samples = estimate_period_brute_force(x)

    # 2. 提取一个周期的模板
    if len(x) < period_samples:
        logging.error("输入信号长度小于一个估计周期，无法生成。")
        return None
    template_period = x[0:period_samples]

    # 3. 通过首尾交叉渐变，使模板无缝
    fade_len = int(period_samples * crossfade_ratio)
    if fade_len == 0:
        logging.warning("周期太短，无法应用交叉渐变。可能会有轻微跳变。")
        return np.tile(template_period, num_periods_to_generate)

    # 创建渐变窗口
    fade_out_window = np.linspace(1, 0, fade_len)
    fade_in_window = np.linspace(0, 1, fade_len)

    # 提取模板的开头和结尾部分
    end_part = template_period[-fade_len:]
    start_part = template_period[0:fade_len]

    # 应用交叉渐变，生成一个能平滑连接尾部和头部的“接头”
    blended_join = end_part * fade_out_window + start_part * fade_in_window

    # 构建无缝模板
    # 错误根源在于直接替换结尾，而未改变开头。
    # 正确做法是：将原始模板中渐变区之后的部分作为主体，再把“接头”拼在末尾。
    # 这样，“接头”就充当了下一个周期的平滑“开头”。
    main_part = template_period[fade_len:]
    seamless_template = np.concatenate((main_part, blended_join))

    # 4. 拼接成多周期信号
    continuous_signal = np.tile(seamless_template, num_periods_to_generate)

    return continuous_signal, seamless_template


def create_comparison_plot(original_waveform, generated_signal, seamless_template, output_path, show_plot=False):
    """
    使用Plotly创建并保存一个交互式图表，对比原始信号和生成的多周期信号。
    """
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("原始样本 vs. 生成的连续信号", "用于拼接的“无缝模板”周期"),
                        shared_xaxes=False)

    # 上图：对比
    fig.add_trace(go.Scatter(y=original_waveform, mode='lines', name='原始样本 (~1.2周期)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=generated_signal, mode='lines', name=f'生成的连续信号 ({args.num_periods}周期)'), row=1,
                  col=1)

    # 下图：展示无缝模板
    fig.add_trace(go.Scatter(y=seamless_template, mode='lines', name='无缝模板'), row=2, col=1)
    # 在模板图上标记首尾点，验证其连续性
    fig.add_trace(go.Scatter(x=[0, len(seamless_template) - 1],
                             y=[seamless_template[0], seamless_template[-1]],
                             mode='markers', name='模板首尾点',
                             marker=dict(color='red', size=10)), row=2, col=1)

    fig.update_layout(
        title="连续周期信号生成结果对比",
        legend_title="信号类型",
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="采样点", row=1, col=1)
    fig.update_xaxes(title_text="采样点 (单个模板周期)", row=2, col=1)
    fig.update_yaxes(title_text="幅值", row=1, col=1)
    fig.update_yaxes(title_text="幅值", row=2, col=1)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logging.info(f"正在将交互式图表保存到: {output_path}")
    fig.write_html(output_path)
    logging.info("保存完成。")

    if show_plot:
        logging.info("在默认浏览器中打开图表...")
        fig.show()


def main():
    """主执行函数"""
    global args
    setup_logging()
    args = parse_args()

    if not os.path.exists(args.data_path):
        logging.error(f"数据文件未找到: {args.data_path}")
        return

    features_generator = FeaturesGeneratorCNN()
    dataset = HDF5SequentialSliceDataset(
        hdf5_file_path=args.data_path,
        transform=features_generator.transform_sample_ae,
        seq_len=features_generator.seq_len,
        step=features_generator.seq_len
    )
    logging.info(f"已从 '{args.data_path}' 加载数据集，总共包含 {len(dataset)} 个样本。")

    if not (0 <= args.sample_index < len(dataset)):
        logging.error(f"指定的样本索引 {args.sample_index} 超出范围 [0, {len(dataset) - 1}]。")
        return

    logging.info(f"正在提取索引为 {args.sample_index} 的样本...")
    input_tensor, label = dataset[args.sample_index]
    original_waveform = input_tensor.squeeze().numpy()

    # 生成连续信号
    generated_signal, seamless_template = generate_continuous_signal(original_waveform, args.num_periods)

    if generated_signal is None:
        return

    logging.info("连续信号生成完成。")

    create_comparison_plot(original_waveform, generated_signal, seamless_template, args.output_file, args.show_plot)

    logging.info("脚本执行完毕。")


if __name__ == '__main__':
    main()
