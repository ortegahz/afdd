# FILE: phase2_eval_search.py

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 路径修正 (确保能导入项目模块) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 项目模块导入 ---
from utils.macros import MIN_VAL_TH
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset
from cores.classifier import MemoryHead, ArcMarginProduct

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 全量评估与参数搜索脚本")

    # 路径参数
    parser.add_argument('--path_ckpt_base', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_base.pt")
    parser.add_argument('--path_ckpt_head', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_head.pt")
    parser.add_argument('--path_ckpt_clf', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_clf.pt")
    parser.add_argument('--data_path', type=str,
                        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3")

    # 模型与计算配置
    parser.add_argument('--ae_model_type', type=str, default='ae', choices=['ae', 'unet'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--step_size', type=float, default=0.1, help="网格搜索权重的步长 (例如 0.1)")

    return parser.parse_args()


class Phase2Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        logging.info(">>> 正在加载模型...")
        # 1. Base AE
        model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
        self.base_model = model_map[self.args.ae_model_type]().to(self.device).eval()
        state_base = torch.load(self.args.path_ckpt_base, map_location=self.device, weights_only=False)
        self.base_model.load_state_dict({k.replace('module.', ''): v for k, v in state_base.items()})

        # Get latent dim
        with torch.no_grad():
            _, dummy_z, _ = self.base_model(torch.randn(1, 1, 448).to(self.device))
            latent_dim = dummy_z.shape[1]

        # 2. Memory Head
        self.mem_head = MemoryHead(latent_dim, mem_dim=1024, hidden_dim=latent_dim).to(self.device).eval()
        state_head = torch.load(self.args.path_ckpt_head, map_location=self.device, weights_only=False)
        self.mem_head.load_state_dict({k.replace('module.', ''): v for k, v in state_head.items()})

        # 3. Aux Classifier
        self.aux_clf = ArcMarginProduct(latent_dim, 2, s=30.0, m=0.50).to(self.device).eval()
        if os.path.exists(self.args.path_ckpt_clf):
            state_clf = torch.load(self.args.path_ckpt_clf, map_location=self.device, weights_only=False)
            self.aux_clf.load_state_dict({k.replace('module.', ''): v for k, v in state_clf.items()})
        else:
            logging.warning("未使用 AuxClassifier 权重，使用随机初始化。")

    def extract_all_scores(self):
        """
        遍历整个数据集，提取三个维度的原始分数：
        1. Reconstruction Error
        2. Distance to nearest Memory Slot
        3. Classifier Probability (Abnormal)
        """
        logging.info(">>> 开始全量推理与特征提取...")
        gen = FeaturesGeneratorCNN()
        dataset = HDF5PeakAlignedDataset(
            hdf5_file_path=self.args.data_path,
            transform=gen.transform_sample_ae,
            seq_len=gen.seq_len,
            min_delta=MIN_VAL_TH
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        all_labels = []
        raw_recon_errs = []
        raw_mem_dists = []
        raw_cls_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Inference"):
                inputs = inputs.to(self.device)

                # 1. Base AE Forward
                recons, latents, _ = self.base_model(inputs)

                # 2. Memory Head Forward
                z = self.mem_head(latents)

                # 3. Classifier Forward
                logits = self.aux_clf(z, label=None)
                probs = F.softmax(logits, dim=1)[:, 1]  # 取异常类的概率

                # --- Metrics Calculation ---

                # A. Recon Error (MSE)
                errs = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

                # B. Memory Distance (Min Euclidean Dist)
                # z: [B, D], mem: [M, D]
                mem_bank = self.mem_head.memory
                # cdist 计算成对距离
                dists = torch.cdist(z, mem_bank, p=2)  # [B, M]
                min_dists, _ = torch.min(dists, dim=1)

                # Collect
                all_labels.append(labels.numpy())
                raw_recon_errs.append(errs.cpu().numpy())
                raw_mem_dists.append(min_dists.cpu().numpy())
                raw_cls_probs.append(probs.cpu().numpy())

        # Concatenate
        self.labels = np.concatenate(all_labels).flatten()
        self.recon_errs = np.concatenate(raw_recon_errs)
        self.mem_dists = np.concatenate(raw_mem_dists)
        self.cls_probs = np.concatenate(raw_cls_probs)

        logging.info(f"推理完成。总样本数: {len(self.labels)}")
        logging.info(f"正例数: {np.sum(self.labels == 1)}, 负例数: {np.sum(self.labels == 0)}")

    def search_optimal_params(self):
        """
        搜索最优的权重组合 (w_recon, w_dist, w_cls) 和 阈值
        """
        logging.info(">>> 开始参数搜索 (Grid Search)...")

        # 1. 归一化 (Normalization)
        # 因为三个指标量纲不同，必须归一化到 [0, 1] 才能加权
        scaler = MinMaxScaler()
        norm_recon = scaler.fit_transform(self.recon_errs.reshape(-1, 1)).flatten()
        norm_dist = scaler.fit_transform(self.mem_dists.reshape(-1, 1)).flatten()
        norm_cls = self.cls_probs  # 概率本身就在 0-1 之间，通常不需要再归一化，或者也可以归一化

        # 2. 生成权重组合
        # w1 + w2 + w3 = 1.0
        step = self.args.step_size
        weights = []
        for w1 in np.arange(0, 1.0 + step / 2, step):  # w_recon
            for w2 in np.arange(0, 1.0 - w1 + step / 2, step):  # w_dist
                w3 = 1.0 - w1 - w2  # w_cls
                if w3 < -1e-5: continue
                weights.append((round(w1, 2), round(w2, 2), round(w3, 2)))

        logging.info(f"待搜索的权重组合数量: {len(weights)}")

        best_f1 = -1.0
        best_params = {}
        best_scores = None

        # 3. 遍历搜索
        for (w_r, w_d, w_c) in tqdm(weights, desc="Searching Weights"):
            # 加权融合分数
            combined_scores = (w_r * norm_recon) + (w_d * norm_dist) + (w_c * norm_cls)

            # 使用 precision_recall_curve 快速找到该组合下的最佳 F1 和 阈值
            # 这是一个高效的方法，避免了手动遍历阈值
            precisions, recalls, thresholds = precision_recall_curve(self.labels, combined_scores)

            # 计算每个阈值下的 F1
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)  # 处理除零

            # 找到当前权重组合下的最大 F1
            idx = np.argmax(f1_scores)
            current_max_f1 = f1_scores[idx]

            if current_max_f1 > best_f1:
                best_f1 = current_max_f1
                # thresholds 数组比 precision/recall 数组短 1
                best_thresh = thresholds[idx] if idx < len(thresholds) else thresholds[-1]

                best_params = {
                    'w_recon': w_r,
                    'w_dist': w_d,
                    'w_cls': w_c,
                    'threshold': best_thresh
                }
                best_scores = combined_scores

        logging.info("-" * 50)
        logging.info(f"搜索结束。最佳 F1 Score: {best_f1:.6f}")
        logging.info(f"最佳参数: {best_params}")

        return best_params, best_scores

    def evaluate_final(self, params, scores):
        """
        使用最佳参数计算最终指标
        """
        threshold = params['threshold']
        preds = (scores >= threshold).astype(int)

        precision = precision_score(self.labels, preds)
        recall = recall_score(self.labels, preds)
        f1 = f1_score(self.labels, preds)
        accuracy = accuracy_score(self.labels, preds)
        report = classification_report(self.labels, preds, target_names=['Normal', 'Abnormal'])

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'report': report,
            'best_params': params
        }


def main():
    args = parse_args()

    # 1. 初始化与推理
    evaluator = Phase2Evaluator(args)
    evaluator.extract_all_scores()

    # 2. 搜索最优参数
    best_params, best_scores = evaluator.search_optimal_params()

    # 3. 计算最终指标
    results = evaluator.evaluate_final(best_params, best_scores)

    # 4. 打印报告
    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT (Optimized)")
    print("=" * 60)
    print(f"Optimal Weights:")
    print(f"  - Reconstruction Error: {best_params['w_recon']:.2f}")
    print(f"  - Memory Distance:      {best_params['w_dist']:.2f}")
    print(f"  - Classifier Prob:      {best_params['w_cls']:.2f}")
    print(f"Optimal Threshold:        {best_params['threshold']:.6f}")
    print("-" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(results['report'])
    print("=" * 60)


if __name__ == "__main__":
    main()
