#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Static INT8 quantization (QOperator 格式) for ONNX models.

示例:
    python quantize_static_int8.py \
           --input  fp32_model.onnx \
           --output int8_qlinear.onnx \
           --calib-data calib.npz \
           --per-channel        \
           --reduce-range
"""

import argparse
import os

import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)


# ------------------------------------------------------------
# 1. DataReader —— 负责把校准数据逐条喂给 quantize_static
# ------------------------------------------------------------
class NPZDataReader(CalibrationDataReader):
    """
    把 .npz 里的 ndarray 序列作为校准数据。
    npz 结构要求:
        key == 模型第一输入的名字 (或通过 --input-name 指定)
        value shape == (N, ...)  其中 N 是样本数
    """

    def __init__(self, npz_path: str, input_name: str):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(npz_path)
        self.npz = np.load(npz_path, allow_pickle=True)
        if input_name not in self.npz:
            raise ValueError(
                f'input_name "{input_name}" 不在 {npz_path} 内的 keys {list(self.npz.keys())}'
            )
        self.data = self.npz[input_name]  # shape = (N, ...)
        self.n_samples = self.data.shape[0]
        self.input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= self.n_samples:
            return None
        sample = self.data[self._idx]
        self._idx += 1
        # 返回 dict{输入名: ndarray}
        return {self.input_name: sample}


# ------------------------------------------------------------
# 2. 量化函数
# ------------------------------------------------------------
def run_static_quant(
        model_fp32: str,
        model_int8: str,
        calib_data: str,
        per_channel: bool = False,
        reduce_range: bool = False,
        input_name: str = None
):
    # 若未显式给出输入名，则取模型的第一个 graph.input
    if input_name is None:
        m = onnx.load(model_fp32)
        input_name = m.graph.input[0].name
        print(f'[Info] 使用模型首输入名: {input_name}')

    data_reader = NPZDataReader(calib_data, input_name)

    print('[Info] 开始静态量化 …')
    quantize_static(
        calibrate_method=CalibrationMethod.MinMax,
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,  # ★ 产生 QLinearConv / QLinearMatMul
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        # op_types_to_quantize=['MatMul'],
    )

    # # （可选）把 IR 版本下调到 9，兼容旧 ORT（1.17 最高只认 9）
    # model = onnx.load(model_int8)
    # if model.ir_version > 9:
    #     model.ir_version = 9
    #     onnx.save(model, model_int8)

    print(f'[Done] INT8(QLinear) 模型已保存到: {model_int8}')


# ------------------------------------------------------------
# 3. CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-training *static* INT8 quantization with QLinearConv'
    )
    parser.add_argument('--input', '-i', default="/home/manu/tmp/afdd_e447.onnx",
                        help='FP32 ONNX model path')
    parser.add_argument('--output', '-o', default="/home/manu/tmp/afdd_e447_int8.onnx",
                        help='INT8 model output path')
    parser.add_argument('--calib-data', '-c', default="/home/manu/tmp/calib.npz",
                        help='.npz containing calibration data')
    parser.add_argument('--input-name',
                        help='Model input name inside npz (default: first graph.input)')
    parser.add_argument('--per-channel', default=True,
                        help='Enable per-channel weight quantization')
    parser.add_argument('--reduce-range', default=True,
                        help='Use reduced quantization range [-64, 63]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_static_quant(
        model_fp32=args.input,
        model_int8=args.output,
        calib_data=args.calib_data,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        input_name=args.input_name
    )
