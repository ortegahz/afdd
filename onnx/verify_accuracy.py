#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the accuracy of a float32 ONNX model and its quantized version
and output a set of suitable similarity / error metrics.

Example:
python verify_accuracy.py
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from tqdm import tqdm

try:
    from fastdtw import fastdtw  # 可选

    HAS_FDTW = True
except ImportError:
    HAS_FDTW = False

# -------------------------------------------------
# 常量
# -------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_EPS = 1e-10


# -------------------------------------------------
# 数据预处理
# -------------------------------------------------
def preprocess(img_path: str, img_size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')

    # Resize（保持短边 256）
    short, long = (256, int(256 * img.width / img.height)) if img.height < img.width \
        else (int(256 * img.height / img.width), 256)
    img = img.resize((long, short), Image.BILINEAR)

    # 中心裁剪 224×224
    left = (img.width - img_size) // 2
    top = (img.height - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    # 归一化 -> NCHW
    img = np.asarray(img).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]
    return img


def load_labels(label_file: str):
    mapping = {}
    with open(label_file, 'r') as f:
        for line in f:
            name, lab = line.strip().split()
            mapping[name] = int(lab)
    return mapping


# -------------------------------------------------
# 推理
# -------------------------------------------------
def run_session(session: ort.InferenceSession,
                input_name: str,
                img_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    preds, probs = [], []
    for img in img_batch:
        logits = session.run(None, {input_name: img})[0]  # (1, C)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        preds.append(np.argmax(prob, axis=1))
        probs.append(prob)
    return np.concatenate(preds, axis=0), np.concatenate(probs, axis=0)


# -------------------------------------------------
# Metric helpers
# -------------------------------------------------
def topk_correct(probs: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([t in topk[i] for i, t in enumerate(targets)])


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - np.max(v))
    return e / (np.sum(e) + _EPS)


def calc_metrics(fp32_probs, int8_probs, fp32_preds,
                 int8_preds, labels_provided, gt=None):
    metrics = {}

    # 误差（绝对/平方）
    diff = fp32_probs - int8_probs
    metrics['MSE'] = mean_squared_error(fp32_probs, int8_probs)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(fp32_probs, int8_probs)
    metrics['MedianAE'] = median_absolute_error(fp32_probs, int8_probs)
    metrics['MaxError'] = float(np.max(np.abs(diff)))
    metrics['MBE'] = np.mean(int8_probs - fp32_probs)  # Bias

    # Cosine 相似度（扁平化）
    flat1, flat2 = fp32_probs.flatten(), int8_probs.flatten()
    metrics['CosineSim'] = 1 - cosine(flat1, flat2)

    # 鲁棒损失
    delta = 1.0
    huber = np.where(np.abs(diff) <= delta,
                     0.5 * diff ** 2,
                     delta * (np.abs(diff) - 0.5 * delta))
    metrics['Huber'] = np.mean(huber)
    metrics['LogCosh'] = np.mean(np.log(np.cosh(diff)))

    # 分布差异（Jensen-Shannon 平均）
    js_list = []
    for p, q in zip(fp32_probs, int8_probs):
        p = _softmax(p)
        q = _softmax(q)
        js = 0.5 * (entropy(p, q) + entropy(q, p))
        js_list.append(js)
    metrics['MeanJS'] = float(np.mean(js_list))

    # DTW（可选）
    if HAS_FDTW:
        dtw_dist, _ = fastdtw(flat1, flat2)
        metrics['DTW'] = dtw_dist

    # Top-k 准确率
    if labels_provided:
        metrics['Top1_FP32'] = float(np.mean(fp32_preds == gt))
        metrics['Top1_INT8'] = float(np.mean(int8_preds == gt))
        metrics['ΔTop1'] = metrics['Top1_FP32'] - metrics['Top1_INT8']

        metrics['Top5_FP32'] = topk_correct(fp32_probs, gt, 5)
        metrics['Top5_INT8'] = topk_correct(int8_probs, gt, 5)
        metrics['ΔTop5'] = metrics['Top5_FP32'] - metrics['Top5_INT8']
    return metrics


def print_metrics(metrics: dict, labels_provided: bool):
    print("\n========== Metric Summary ==========")
    if labels_provided:
        print(f"Top-1  FP32: {metrics['Top1_FP32']:.4f}  "
              f"INT8: {metrics['Top1_INT8']:.4f}  "
              f"Δ: {metrics['ΔTop1']:+.4f}")
        print(f"Top-5  FP32: {metrics['Top5_FP32']:.4f}  "
              f"INT8: {metrics['Top5_INT8']:.4f}  "
              f"Δ: {metrics['ΔTop5']:+.4f}")
    print(f"MSE:        {metrics['MSE']:.6e}")
    print(f"RMSE:       {metrics['RMSE']:.6e}")
    print(f"MAE:        {metrics['MAE']:.6e}")
    print(f"Median AE:  {metrics['MedianAE']:.6e}")
    print(f"Max Error:  {metrics['MaxError']:.6e}")
    print(f"MBE:        {metrics['MBE']:.6e}")
    print(f"Cosine Sim: {metrics['CosineSim']:.6f}")
    print(f"Huber Loss: {metrics['Huber']:.6e}")
    print(f"Log-Cosh:   {metrics['LogCosh']:.6e}")
    print(f"Mean JS Divergence: {metrics['MeanJS']:.6e}")
    if 'DTW' in metrics:
        print(f"DTW Dist.:  {metrics['DTW']:.6e}")
    else:
        print("DTW Dist.:  fastdtw 未安装，未计算")


# -------------------------------------------------
# main
# -------------------------------------------------
def main(args):
    # 1. 收集图片
    img_files = sorted([f for f in os.listdir(args.data_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    labels = load_labels(args.label_file) if args.label_file else None
    if labels:
        img_files = [f for f in img_files if f in labels]

    if not img_files:
        raise RuntimeError("未找到可用测试图片！")

    # 2. 创建 ORT Session
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess_fp32 = ort.InferenceSession(args.float_model, so, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(args.quant_model, so, providers=['CPUExecutionProvider'])
    input_name = sess_fp32.get_inputs()[0].name

    # 3. 推理
    fp32_preds, int8_preds = [], []
    fp32_probs, int8_probs = [], []

    for img_name in tqdm(img_files, desc='Inference'):
        img_np = preprocess(os.path.join(args.data_dir, img_name))
        p1, prob1 = run_session(sess_fp32, input_name, [img_np])
        p2, prob2 = run_session(sess_int8, input_name, [img_np])

        fp32_preds.append(p1)
        int8_preds.append(p2)
        fp32_probs.append(prob1)
        int8_probs.append(prob2)

    fp32_preds = np.concatenate(fp32_preds, axis=0)
    int8_preds = np.concatenate(int8_preds, axis=0)
    fp32_probs = np.concatenate(fp32_probs, axis=0)
    int8_probs = np.concatenate(int8_probs, axis=0)

    # 4. 计算指标
    gt = np.array([labels[f] for f in img_files], dtype=np.int64) if labels else None
    metrics = calc_metrics(fp32_probs, int8_probs,
                           fp32_preds, int8_preds,
                           labels_provided=bool(labels), gt=gt)

    # 5. 打印
    print_metrics(metrics, labels_provided=bool(labels))


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check accuracy loss after ONNX QDQ quantization')
    parser.add_argument('--float_model',
                        default="/home/manu/tmp/classifier_sim.onnx")
    parser.add_argument('--quant_model',
                        default="/home/manu/tmp/classifier_sim-quant.onnx")
    parser.add_argument('--data_dir',
                        default="/home/manu/tmp/test_images/")
    parser.add_argument('--label_file',
                        default=None,
                        help='optional label file (txt) for accuracy computation')
    args = parser.parse_args()
    main(args)
