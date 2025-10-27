# FILE: test_ae_manual_threshold.py

import argparse
import logging
import os
import sys

import torch

# 确保可以找到 cores 子目录下的模块
from cores.classifier import ClassifierCNNAE


# 将项目根目录添加到Python路径中，以确保能够正确导入模块
# 假设此脚本与 demo_classifier.py 在同一目录或项目的根目录下
# 如果目录结构不同，请相应调整路径
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, project_root)


def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained AutoEncoder model with a manual threshold."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/manu/tmp/afdd_models_mp/ae_best_e338_acc0.9785.pt",
        help="Path to the pre-trained AutoEncoder model (.pt file)."
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default="/home/manu/tmp/test_data.h5",
        help="Path to the HDF5 test data file (e.g., test_data.h5)."
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.000005,
        help="Manual threshold to use for classifying anomalies."
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="Device to run evaluation on, e.g., 'cpu' or 'cuda:0'. Defaults to 'cuda:0' if available, else 'cpu'."
    )
    return parser.parse_args()


def main():
    """主执行函数"""
    setup_logging()
    args = parse_args()

    # --- 1. 设置设备 ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        return

    if not os.path.exists(args.test_data_path):
        logging.error(f"Test data file not found: {args.test_data_path}")
        return

    # --- 2. 准备ClassifierCNNAE所需的参数 ---
    # ClassifierCNNAE的构造函数需要一个类似argparse的命名空间对象
    # 我们创建一个最小化的对象，只包含它需要的属性
    classifier_args = argparse.Namespace(
        rank=0,  # 非分布式模式下，rank为0
        path_ckpt=args.model_path,
        save_dir=None  # 评估时不需要保存目录
    )

    # --- 3. 初始化分类器并加载模型 ---
    logging.info(f"Loading model from: {args.model_path}")
    try:
        # is_infer=True的等价设置
        classifier = ClassifierCNNAE(args=classifier_args, ddp=False)
        classifier.model.to(device)
        # 切换到评估模式
        classifier.model.eval()
    except Exception as e:
        logging.error(f"Failed to initialize or load the model: {e}")
        return

    # --- 4. 调用evaluate函数并传入手动阈值 ---
    logging.info(f"Evaluating model on data: {args.test_data_path}")
    logging.info(f"Using manual threshold: {args.threshold}")

    # 调用修改后的evaluate函数
    overall_accuracy = classifier.evaluate(
        test_path=args.test_data_path,
        threshold=args.threshold
    )

    logging.info("-" * 40)
    logging.info(f"Evaluation complete.")
    logging.info(f"Overall Accuracy with threshold {args.threshold:.6f}: {overall_accuracy:.4f}")
    logging.info("-" * 40)


if __name__ == '__main__':
    main()
