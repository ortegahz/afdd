# FILE: visualize_peaks_residual.py

import argparse
import os
import sys

import dash
import dash_bootstrap_components as dbc
import h5py
import numpy as np
import plotly.graph_objects as go
import pywt
import torch
from dash import dcc, html, Input, Output, State
from scipy.signal import find_peaks

# --- 1. 项目路径 Hack (确保能导入 cores) ---
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设 tools/ 上一级是根目录)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入模型定义
try:
    from cores.nets import NetAFDAE, NetAFDAE_2D_CWT
except ImportError:
    print(f"错误: 无法导入 cores.nets。")
    print(f"请确保您的工作目录结构正确，或者手动设置 PYTHONPATH。")
    print(f"尝试添加的路径: {project_root}")
    sys.exit(1)

# --- 2. 配置部分 ---
SEQ_LEN = 448
DEFAULT_DATA_PATH = "/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3"
# 默认模型路径
DEFAULT_CKPT_PATH = "/home/manu/mnt/8gpu_3090/afdd_models_mp_v11/ae_best.pt"
MIN_VAL_TH = 100

# 模型配置 (必须与训练 classifier.py 时的配置一致)
MODEL_CONFIG = {
    "latent_dim": 128,
}

# --- 3. 全局变量 ---
ARGS = None
MODEL = None  # Phase 1/2 模型
MODEL_CWT = None  # Phase 3 模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 4. 辅助函数 ---

def load_keys(path):
    if not os.path.exists(path): return []
    with h5py.File(path, 'r') as f: return list(f.keys())


def get_signal_data(path, key):
    """同时读取信号和标签"""
    with h5py.File(path, 'r') as f:
        if key not in f: return None, None
        sig = f[key]['signal'][:]
        # 这里假设如果h5里有label_seq就读，没有就全0
        lab = f[key]['label_seq'][:] if 'label_seq' in f[key] else np.zeros_like(sig)
        return sig, lab


def normalize_min_max(segment):
    """归一化到 [-1, 1]"""
    mi, ma = np.min(segment), np.max(segment)
    if ma == mi: return np.zeros_like(segment)
    return 2 * (segment - mi) / (ma - mi) - 1


def load_model(ckpt_path):
    """
    加载 Stage 1 MemoryAE / NetAFDAE 模型。
    """
    if not os.path.exists(ckpt_path):
        print(f"警告: 模型文件不存在 {ckpt_path}")
        return None

    print(f"Loading Base model from {ckpt_path} using device {DEVICE} ...")

    # 实例化模型
    model = NetAFDAE(**MODEL_CONFIG)

    # 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

        # 处理 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        print("Base Model loaded successfully!")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_cwt_model(ckpt_path):
    """
    加载 Phase 3 CWT 2D-CNN-AE 模型
    """
    if not os.path.exists(ckpt_path):
        print(f"提示: CWT 模型文件不存在 {ckpt_path} (本次可视化将跳过 Phase 3)")
        return None

    print(f"Loading CWT model from {ckpt_path} using device {DEVICE} ...")

    # 实例化 CWT 模型
    model = NetAFDAE_2D_CWT(latent_dim=128)

    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            # 若是有 module. 前缀则去除
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        print("CWT Model loaded successfully!")
        return model
    except Exception as e:
        print(f"CWT 模型加载失败: {e}")
        return None


def compute_residual(segment):
    """
    核心计算逻辑：
    1. 接收归一化后的 numpy waveform (448,)
    2. Phase 1推理 -> Recon, Residual
    3. Residual -> CWT (Orig)
    4. Phase 3推理 -> CWT Recon
    """
    if MODEL is None:
        # 如果没有模型，返回全0以防止报错
        return np.zeros_like(segment), np.zeros_like(segment), None, None

    # 预处理: (SEQ_LEN,) -> (1, 1, SEQ_LEN)
    input_tensor = torch.from_numpy(segment).float().view(1, 1, -1).to(DEVICE)

    # --- Phase 1/2 Inference ---
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        if isinstance(outputs, tuple):
            recon = outputs[0]
        else:
            recon = outputs

    recon_np = recon.cpu().numpy().squeeze()

    # 计算残差
    residual = segment - recon_np

    # --- Phase 3 Inference (CWT) ---
    cwt_power = None
    cwt_recon_np = None

    # 无论有无模型，先计算 CWT 图以便展示 "Original CWT"
    # 尺度范围 1-64
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(residual, scales, 'cmor1.5-1.0')
    cwt_power = np.abs(coef)  # Shape: (64, 448)

    if MODEL_CWT is not None:
        # 构建输入 Tensor: (1, 1, 64, 448)
        cwt_tensor = torch.from_numpy(cwt_power).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_cwt = MODEL_CWT(cwt_tensor)
            # 处理可能的元组返回
            if isinstance(output_cwt, tuple):
                output_cwt = output_cwt[0]

            cwt_recon_np = output_cwt.cpu().numpy().squeeze()  # Shape: (64, 448)

    return residual, recon_np, cwt_power, cwt_recon_np


# --- 5. Dash App 初始化 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("AFDD Multi-Phase Analysis Dashboard"), className="my-3")),

    # 控制区
    dbc.Row([
        dbc.Col([
            html.Label("Select Sample (H5 Key):"),
            dcc.Dropdown(id='key-dropdown', placeholder="Loading keys...", searchable=True),
        ], width=8),
        dbc.Col(html.Div(id='status-text', className="mt-4 text-muted"), width=4)
    ]),

    html.Hr(),

    # 1. 全局长波形视图
    dbc.Row([
        dbc.Col([
            html.H5("1. Global View (Click on a Peak to inspect)"),
            dcc.Graph(id='global-graph', style={'height': '350px'})
        ])
    ]),

    html.Hr(),

    # 2. 局部详情视图：Phase 1 (Input vs Recon) & Phase 1 Residual
    dbc.Row([
        # 左图：模型输入 vs 重构
        dbc.Col([
            html.H5("2. Phase 1: Input (Green) vs Recon (Orange)"),
            dcc.Graph(id='recon-graph', style={'height': '400px'})
        ], width=6),

        # 右图：残差信号
        dbc.Col([
            html.H5("3. Residual Signal"),
            dcc.Graph(id='resid-graph', style={'height': '400px'})
        ], width=6)
    ]),

    html.Hr(),

    # 3. Phase 3 视图：Original CWT vs Reconstructed CWT
    dbc.Row([
        # 左图：原始残差 CWT
        dbc.Col([
            html.H5("4. Phase 3 Input: Residual CWT Scalogram"),
            dcc.Graph(id='cwt-org-graph', style={'height': '400px'})
        ], width=6),

        # 右图：重构 CWT
        dbc.Col([
            html.H5("5. Phase 3 Output: Reconstructed CWT"),
            dcc.Graph(id='cwt-rec-graph', style={'height': '400px'})
        ], width=6),
    ]),

    # 隐藏存储，用于避免重复加载 IO
    dcc.Store(id='current-signal-store'),
], fluid=True)


# --- 回调逻辑 ---

@app.callback(
    Output('key-dropdown', 'options'),
    Output('key-dropdown', 'value'),
    Input('key-dropdown', 'search_value')
)
def init_keys(_):
    k = load_keys(ARGS.data_path)
    options = [{'label': i, 'value': i} for i in k]
    value = k[0] if k else None
    return options, value


@app.callback(
    Output('global-graph', 'figure'),
    Output('current-signal-store', 'data'),
    Output('status-text', 'children'),
    Input('key-dropdown', 'value')
)
def update_global_view(key):
    if not key:
        return go.Figure(), None, "No data selected."

    sig, lab = get_signal_data(ARGS.data_path, key)
    if sig is None:
        return go.Figure(), None, "Error loading signal."

    # 寻找峰值用于标记
    peaks, _ = find_peaks(sig, prominence=ARGS.min_delta, distance=ARGS.distance)

    # === Step 1: 批量计算所有 Peaks 的重构误差 (MSE) ===
    mses = np.zeros(len(peaks))

    if len(peaks) > 0 and MODEL is not None:
        try:
            segments = []
            for p in peaks:
                # 对齐策略：Peak 位于窗口的最末端
                end_idx = p + 1
                start_idx = end_idx - SEQ_LEN

                if start_idx < 0:
                    seg = sig[:end_idx]
                    seg = np.pad(seg, (SEQ_LEN - len(seg), 0), 'constant')
                else:
                    seg = sig[start_idx:end_idx]

                # 归一化 [-1, 1]
                segments.append(normalize_min_max(seg))

            # 批量推理
            batch_tensor = torch.from_numpy(np.array(segments)).float().view(-1, 1, SEQ_LEN).to(DEVICE)
            with torch.no_grad():
                outputs = MODEL(batch_tensor)
                recons = outputs[0] if isinstance(outputs, tuple) else outputs

            # 计算 MSE
            recons_np = recons.cpu().numpy().squeeze(1)
            mses = np.mean((np.array(segments) - recons_np) ** 2, axis=1)

        except Exception as e:
            print(f"Batch inference failed: {e}")

    # === Step 2: 确定 Peak 的颜色 ===
    # 优先级: MSE > 0.001 (Yellow) > Label Abnormal (Red) > Normal (Blue)
    peak_colors = []
    for i, p in enumerate(peaks):
        if mses[i] > 0.001:
            peak_colors.append('#FFD700')  # Yellow (High Reconstruction Error)
        elif lab[p] > 0:
            peak_colors.append('#d62728')  # Red (Labeled Abnormal)
        else:
            peak_colors.append('#1f77b4')  # Blue (Normal)

    # === Step 3: 绘制全局图 ===
    fig = go.Figure()

    # (A) 原始波形
    fig.add_trace(go.Scatter(y=sig, name='Signal', line=dict(color='#1f77b4', width=1)))

    # (B) 异常部分 (Label > 0)
    sig_abnormal = sig.copy().astype(float)
    sig_abnormal[lab == 0] = np.nan
    fig.add_trace(go.Scatter(y=sig_abnormal, name='Abnormal', line=dict(color='#d62728', width=1.5)))

    # (C) Peaks 标记
    fig.add_trace(go.Scatter(
        x=peaks, y=sig[peaks],
        mode='markers', name='Peaks',
        marker=dict(size=10, color=peak_colors, symbol='x', line=dict(width=3, color=peak_colors)),
        customdata=peaks
    ))

    num_abnormal_peaks = np.sum(lab[peaks] > 0)
    num_high_error_peaks = np.sum(mses > 0.001)

    fig.update_layout(
        title=f"Sample: {key} (Peaks: {len(peaks)} | Labeled Abnormal: {num_abnormal_peaks} | High Res: {num_high_error_peaks})",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )

    return fig, sig.tolist(), f"Loaded {key}. Yellow peaks: MSE > 0.001."


@app.callback(
    Output('recon-graph', 'figure'),
    Output('resid-graph', 'figure'),
    Output('cwt-org-graph', 'figure'),
    Output('cwt-rec-graph', 'figure'),
    Input('global-graph', 'clickData'),
    State('current-signal-store', 'data')
)
def update_detail_view(clickData, sig_list):
    # 默认空图
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white")

    if not clickData or not sig_list:
        return empty_fig, empty_fig, empty_fig, empty_fig

    sig = np.array(sig_list)
    click_x = int(clickData['points'][0]['x'])

    # 截取窗口 [click_x - SEQ_LEN + 1 : click_x + 1]
    end_idx = click_x + 1
    start_idx = end_idx - SEQ_LEN

    if start_idx < 0:
        raw_seg = sig[:end_idx]
        pad_len = SEQ_LEN - len(raw_seg)
        raw_seg = np.pad(raw_seg, (pad_len, 0), 'constant')
    else:
        raw_seg = sig[start_idx:end_idx]

    # 1. 归一化
    norm_input = normalize_min_max(raw_seg)

    # 2. 推理计算 (包含 Phase 1 & 3)
    residual, recon, cwt_org, cwt_rec = compute_residual(norm_input)

    # === 图 1: Input vs Recon ===
    fig_rec = go.Figure()
    fig_rec.add_trace(go.Scatter(y=norm_input, name='Input', line=dict(color='green', width=2)))
    fig_rec.add_trace(go.Scatter(y=recon, name='Recon', line=dict(color='orange', width=2, dash='dash')))
    fig_rec.add_trace(go.Scatter(x=[SEQ_LEN - 1], y=[norm_input[-1]], mode='markers', marker=dict(color='red', size=8),
                                 showlegend=False))
    fig_rec.update_layout(title=f"Phase 1 Reconstruction (Peak @ {click_x})", template="plotly_white",
                          yaxis=dict(range=[-1.2, 1.2]))

    # === 图 2: Residual ===
    res_mse = np.mean(residual ** 2)
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(y=residual, name='Residual', line=dict(color='#d62728', width=1.5)))
    fig_res.update_layout(title=f"Residual (MSE: {res_mse:.6f})", template="plotly_white",
                          yaxis=dict(range=[-1.2, 1.2]))

    # === 图 3: CWT Original ===
    # 尺度轴 y
    scales = np.arange(1, 65)

    fig_cwt_org = go.Figure(data=go.Heatmap(
        z=cwt_org,
        x=np.arange(len(residual)),
        y=scales,
        colorscale='Viridis',
        zmin=0, zmax=np.max(cwt_org) if cwt_org is not None else 1,
        colorbar=dict(title="Mag")
    ))
    fig_cwt_org.update_layout(
        title="Phase 3 Input: Residual CWT",
        xaxis_title="Time", yaxis_title="Scale",
        template="plotly_white"
    )

    # === 图 4: CWT Reconstructed ===
    if cwt_rec is not None:
        cwt_mse = np.mean((cwt_org - cwt_rec) ** 2)
        title_text = f"Phase 3 Output: Reconstructed CWT (MSE: {cwt_mse:.6f})"

        fig_cwt_rec = go.Figure(data=go.Heatmap(
            z=cwt_rec,
            x=np.arange(len(residual)),
            y=scales,
            colorscale='Viridis',
            # 为了方便对比，使用与原图相同的 zmax
            zmin=0, zmax=np.max(cwt_org),
            colorbar=dict(title="Mag")
        ))
    else:
        title_text = "Phase 3 Model Not Loaded"
        fig_cwt_rec = go.Figure()

    fig_cwt_rec.update_layout(
        title=title_text,
        xaxis_title="Time", yaxis_title="Scale",
        template="plotly_white"
    )

    return fig_rec, fig_res, fig_cwt_org, fig_cwt_rec


def parse_args():
    parser = argparse.ArgumentParser(description="AFDD Multi-Phase Analysis Tool")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help="Path to H5 data")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH, help="Path to Phase 1/2 .pt model")
    parser.add_argument('--cwt_checkpoint', type=str, default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase3_cwt_best.pt", help="Path to Phase 3 CWT model (optional)")
    parser.add_argument('--port', type=int, default=8051, help="Dash port")
    parser.add_argument('--min_delta', type=float, default=float(MIN_VAL_TH), help="Peak threshold")
    parser.add_argument('--distance', type=int, default=100, help="Peak distance")
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()

    # 1. 检查数据
    if not os.path.exists(ARGS.data_path):
        print(f"Error: 数据文件未找到: {ARGS.data_path}")
        sys.exit(1)

    # 2. 加载 Phase 1/2 模型
    MODEL = load_model(ARGS.checkpoint)

    # 3. 加载 Phase 3 CWT 模型
    cwt_path = ARGS.cwt_checkpoint
    if cwt_path is None:
        # 自动推断：假设在 checkpoint 同级目录下，名为 phase3_cwt_best.pt
        base_dir = os.path.dirname(ARGS.checkpoint)
        cwt_path = os.path.join(base_dir, "phase3_cwt_best.pt")

    MODEL_CWT = load_cwt_model(cwt_path)

    if MODEL is None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("严重警告: 基础模型加载失败！无法进行任何分析。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # 3. 启动应用
    print(f"Starting Dashboard...")
    print(f" > Data: {ARGS.data_path}")
    print(f" > Base Model: {ARGS.checkpoint}")
    print(f" > CWT Model:  {cwt_path} ({'Loaded' if MODEL_CWT else 'Not Found'})")
    print(f" > URL: http://0.0.0.0:{ARGS.port}")

    app.run(debug=True, port=ARGS.port, host='0.0.0.0')
