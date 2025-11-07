# FILE: save_all_reconstructions.py

import argparse
import logging
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保可以找到 cores 子目录下的模块
# 假设此脚本位于项目根目录，或者项目根目录已添加到 PYTHONPATH
# -- 健壮的路径修正方案 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cores.nets import NetAFDAE_Mem_Flow, NetAFDAE_UNet, NetAFDAE_Mem, NetAFDAE_UNet_Mem
from cores.features_generator import FeaturesGeneratorCNN, HDF5SequentialSliceDataset

# 提早检查matplotlib，如果未安装则给出明确错误
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def calculate_attention_entropy(attention_weights, epsilon=1e-12):
    """
    计算注意力权重的熵，以评估稀疏性。
    熵越低，分布越稀疏（集中）。
    """
    # H(p) = - sum(p * log(p))
    entropy = -attention_weights * torch.log(attention_weights + epsilon)
    # 在记忆单元维度上求和，得到每个样本的熵
    return torch.sum(entropy, dim=1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用预训练的AutoEncoder模型处理HDF5数据集，并为每个样本保存重构图。"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        # default="/media/manu/ST8000DM004-2U91/afdd/models/models_ae/v6 - mem flow ae/afdd_models_mp/ae_best_e2770_acc1.0012.pt",
        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/ae_best_e780_acc1.0643.pt",
        help="预训练的AutoEncoder模型 (.pt 文件) 的路径。"
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        # default="/home/manu/tmp/afd_pm_hdf5/train_data.h5",
        default="/home/manu/tmp/afd_pm_hdf5_v3/train_data.h5",
        help="HDF5测试数据文件 (例如, test_data.h5) 的路径。"
    )
    parser.add_argument(
        '--ae_model_type',
        type=str,
        default='mem-flow-ae',
        choices=['unet', 'mem-ae', 'unet-mem', 'mem-flow-ae'],
        help="要加载的自编码器模型架构类型。"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/home/manu/tmp/afdd_ae_results",
        help="用于保存输出图像的目录。程序将在此目录下创建 'positive' 和 'negative' 子目录。"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="运行评估的设备，例如 'cpu' or 'cuda:0'。"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="处理数据时使用的批大小。"
    )
    return parser.parse_args()


def plot_reconstruction(original_signal, reconstructed_signal, error, label, sample_index, output_dir,
                        log_likelihood=None, attention_entropy=None):
    """
    生成并保存一个比较原始信号和其重构信号的图像。
    """
    label_text = "Positive" if label == 1 else "Negative"

    # 确保保存目录存在
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    title = f'Reconstruction of {sample_index}-th {label_text} Sample\n'
    title += f'MSE: {error:.6f}'
    if log_likelihood is not None:
        title += f' | Log-Likelihood: {log_likelihood:.4f}'
    if attention_entropy is not None:
        title += f' | Attention Entropy: {attention_entropy:.4f}'

    fig.suptitle(title, fontsize=16)

    axs[0].plot(original_signal, color='blue', label='Original')
    axs[0].set_title('Original Signal')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].plot(reconstructed_signal, color='orange', label='Reconstructed')
    axs[1].set_title('Reconstructed Signal')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].plot(original_signal, label='Original', color='blue', alpha=0.9)
    axs[2].plot(reconstructed_signal, label='Reconstructed', color='red', linestyle='--', alpha=0.8)
    axs[2].set_title('Overlay')
    axs[2].set_xlabel('Time Step')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 使用格式化的索引以保证文件名排序一致
    plot_filename = f'sample_{sample_index:06d}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)


def main():
    """主执行函数"""
    setup_logging()
    args = parse_args()

    if not MATPLOTLIB_AVAILABLE:
        logging.error("Matplotlib 未安装。请使用 'pip install matplotlib' 命令安装。")
        return

    # --- 1. 环境设置 ---
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    logging.info(f"Using device: {device}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    pos_dir = os.path.join(args.output_dir, 'positive')
    neg_dir = os.path.join(args.output_dir, 'negative')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    logging.info(f"输出图像将保存至: {args.output_dir}")

    if not os.path.exists(args.model_path):
        logging.error(f"模型文件未找到: {args.model_path}")
        return
    if not os.path.exists(args.test_data_path):
        logging.error(f"测试数据文件未找到: {args.test_data_path}")
        return

    # --- 2. 加载模型 ---
    model_map = {
        'unet': NetAFDAE_UNet,
        'mem-ae': NetAFDAE_Mem,
        'unet-mem': NetAFDAE_UNet_Mem,
        'mem-flow-ae': NetAFDAE_Mem_Flow
    }
    if args.ae_model_type not in model_map:
        logging.error(f"不支持的模型类型: {args.ae_model_type}")
        return

    try:
        # 直接实例化模型
        model_class = model_map[args.ae_model_type]
        model = model_class()
        # 加载权重
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"成功从 {args.model_path} 加载模型")
    except Exception as e:
        logging.error(f"初始化或加载模型失败: {e}")
        return

    # --- 3. 准备数据集 ---
    features_generator = FeaturesGeneratorCNN()

    # dataset = HDF5SequentialSliceDataset(
    #     hdf5_file_path=args.test_data_path,
    #     transform=features_generator.transform_sample_ae,
    #     seq_len=features_generator.seq_len,
    #     step=features_generator.seq_len  # step=seq_len 表示无重叠切片
    # )

    dataset = HDF5SequentialSliceDataset(
        hdf5_file_path=args.test_data_path,
        transform=features_generator.transform_sample_ae,
        seq_len=features_generator.seq_len,
        step=features_generator.seq_len // 8
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"已加载数据集，包含 {len(dataset)} 个样本。")

    # --- 4. 处理数据并保存图像 ---
    pos_counter = 0
    neg_counter = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="正在处理样本"):
            inputs = inputs.to(device)

            # 根据模型类型处理不同的输出
            model_outputs = model(inputs)
            reconstructions = model_outputs[0]
            log_probs = None
            attention_weights = None

            if args.ae_model_type == 'mem-flow-ae':
                attention_weights = model_outputs[2]
                log_probs = model_outputs[3]  # Mem-Flow AE返回4个值，第4个是log_prob
            elif args.ae_model_type in ['mem-ae', 'unet-mem']:
                attention_weights = model_outputs[2]

            # 注意: 计算每个样本的均方误差 (MSE)。
            # 假设输入形状为 [N, C, L]，例如 [64, 1, 448]。
            # 我们在特征维度 (1, 2) 上计算均值来为每个样本获得一个误差分数。
            errors = torch.mean((inputs - reconstructions) ** 2, dim=(1, 2))

            # 将数据移至CPU以进行numpy转换和绘图
            entropies = None
            if attention_weights is not None:
                entropies = calculate_attention_entropy(attention_weights)

            inputs_cpu = inputs.cpu().numpy()
            recons_cpu = reconstructions.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            errors_cpu = errors.cpu().numpy()
            log_probs_cpu = log_probs.cpu().numpy() if log_probs is not None else None
            entropies_cpu = entropies.cpu().numpy() if entropies is not None else None

            for i in range(inputs.size(0)):
                label = int(labels_cpu[i, 0])

                if label == 1:
                    pos_counter += 1
                    sample_index = pos_counter
                    save_dir = pos_dir
                else:
                    neg_counter += 1
                    sample_index = neg_counter
                    save_dir = neg_dir

                ll_for_sample = log_probs_cpu[i] if log_probs_cpu is not None else None
                entropy_for_sample = entropies_cpu[i] if entropies_cpu is not None else None

                # 调用绘图函数
                plot_reconstruction(
                    original_signal=inputs_cpu[i].flatten(),
                    reconstructed_signal=recons_cpu[i].flatten(),
                    error=errors_cpu[i],
                    label=label,
                    sample_index=sample_index,
                    output_dir=save_dir,
                    log_likelihood=ll_for_sample,
                    attention_entropy=entropy_for_sample
                )

    logging.info("-" * 40)
    logging.info(f"处理完成。")
    logging.info(f"已保存 {pos_counter} 个正例样本图像至: {pos_dir}")
    logging.info(f"已保存 {neg_counter} 个负例样本图像至: {neg_dir}")
    logging.info("-" * 40)


if __name__ == '__main__':
    main()
