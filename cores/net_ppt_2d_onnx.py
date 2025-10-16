#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_and_draw_2d.py
--------------------------------------------------------------
1) 导出 & 简化 ONNX（224×224 灰度图）
2) torch / onnxruntime 结果比对
3) torchviz 全图（可选）
4) Netron 横向交互（可选）
5) Graphviz 极简 PPT 示意图（默认开启）
--------------------------------------------------------------
"""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify

# ========= 用户可配置区域 =====================================
from net_ppt_2d import SimpleSE2DNet224  # ← 你的 2-D 网络

TMP_DIR = Path('/home/manu/tmp')  # 临时输出目录
TMP_DIR.mkdir(parents=True, exist_ok=True)

VIS_TORCHVIZ = False
VIS_NETRON = False
GEN_PPT_FIG = True
# ==============================================================

# ---------- 依赖检查（略） ----------
if VIS_NETRON:
    need_ver = (5, 9, 0)
    try:
        import netron

        ver = tuple(map(int, netron.__version__.split('.')))
        if ver < need_ver:
            print(f'Upgrading Netron {netron.__version__}  →  >=5.9.0')
            subprocess.check_call([sys.executable, "-m", "pip",
                                   "install", "-U", "netron"])
            importlib.reload(netron)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "netron"])
        import netron
# ------------------------------------

# =============================================================
# 1. 构建 PyTorch 模型
# =============================================================
model = SimpleSE2DNet224(num_classes=1)  # 根据需要改分类数
model.eval()
dummy = torch.randn(1, 1, 224, 224)  # (B,C,H,W)

# =============================================================
# 2. 导出 ONNX
# =============================================================
onnx_path = TMP_DIR / 'se2d_ppt.onnx'
torch.onnx.export(
    model,
    dummy,
    onnx_path.as_posix(),
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output', 'feat'],
    dynamic_axes={
        'input': {0: 'batch'},  # 如要支持动态 H/W 再加 2、3 轴
        'output': {0: 'batch'},
        'feat': {0: 'batch'}
    }
)
print(f'[ONNX] raw saved → {onnx_path}')

# =============================================================
# 3. 简化 ONNX
# =============================================================
print('[ONNX] Simplifying ...')
model_onnx = onnx.load(onnx_path)
model_simp, checked = simplify(model_onnx)
assert checked, 'Simplified model check failed!'
simp_path = TMP_DIR / 'se2d_ppt_sim.onnx'
onnx.save(model_simp, simp_path)
print(f'[ONNX] simplified saved → {simp_path}')

# =============================================================
# 4. Runtime 结果对齐
# =============================================================
ort_sess = ort.InferenceSession(simp_path.as_posix(),
                                providers=['CPUExecutionProvider'])
ort_outs = ort_sess.run(None, {'input': dummy.numpy()})

pt_out, pt_feat = model(dummy)

print('[CHECK] max diff  output:', np.abs(ort_outs[0] - pt_out.detach().numpy()).max())
print('[CHECK] max diff  feat  :', np.abs(ort_outs[1] - pt_feat.detach().numpy()).max())

# =============================================================
# 5. （可选）后续画图、Netron 展示，参考 1-D 版脚本
# =============================================================
