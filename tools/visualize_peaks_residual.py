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
    from cores.nets import NetAFDAE
except ImportError:
    print(f"错误: 无法导入 cores.nets.NetAFDAE。")
    print(f"请确保您的工作目录结构正确，或者手动设置 PYTHONPATH。")
    print(f"尝试添加的路径: {project_root}")
    sys.exit(1)

# --- 2. 配置部分 ---
SEQ_LEN = 448
DEFAULT_DATA_PATH = "/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3"
# 默认模型路径，请修改为您实际的 .pt / .pth 文件路径
DEFAULT_CKPT_PATH = "/home/manu/mnt/8gpu_3090/afdd_models_mp_v11/ae_best.pt"
MIN_VAL_TH = 100

# 模型配置 (必须与训练 classifier.py 时的配置一致)
MODEL_CONFIG = {
    "latent_dim": 128,
}

# --- 3. 全局变量 ---
ARGS = None
MODEL = None
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
    包含处理 DDP (DistributedDataParallel) 产生的 'module.' 前缀的逻辑。
    """
    if not os.path.exists(ckpt_path):
        print(f"警告: 模型文件不存在 {ckpt_path}")
        return None

    print(f"Loading model from {ckpt_path} using device {DEVICE} ...")

    # 实例化模型
    model = NetAFDAE(**MODEL_CONFIG)

    # 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        # 兼容不同的保存格式 (直接保存 state_dict 或保存 dict 包含 'model_state_dict')
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

        # 处理 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # 加载参数 (strict=False 允许一定的灵活性，但在生产环境建议 True)
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_residual(segment):
    """
    核心计算逻辑：
    1. 接收归一化后的 numpy waveform (448,)
    2. 转 Tensor 送入模型
    3. 获取重构结果 Recon
    4. 计算 Residual = Input - Recon
    """
    if MODEL is None:
        # 如果没有模型，返回全0以防止报错
        return np.zeros_like(segment), np.zeros_like(segment)

    # 预处理: (SEQ_LEN,) -> (1, 1, SEQ_LEN)
    input_tensor = torch.from_numpy(segment).float().view(1, 1, -1).to(DEVICE)

    with torch.no_grad():
        # MemoryAutoEncoder forward 返回: (recon, hidden, ...) depending on impl
        # 这里解包第一个返回值作为重构
        outputs = MODEL(input_tensor)
        if isinstance(outputs, tuple):
            recon = outputs[0]
        else:
            recon = outputs

    recon_np = recon.cpu().numpy().squeeze()

    # 计算残差
    residual = segment - recon_np

    return residual, recon_np


# --- 5. Dash App 初始化 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Stage 1 Residual Analysis (Model-Based)"), className="my-3")),

    # 控制区
    dbc.Row([
        dbc.Col([
            html.Label("Select Sample (H5 Key):"),
            dcc.Dropdown(id='key-dropdown', placeholder="Loading keys...", searchable=True),
        ], width=8),
        dbc.Col(html.Div(id='status-text', className="mt-4 text-muted"), width=4)
    ]),

    html.Hr(),

    # 全局长波形视图
    dbc.Row([
        dbc.Col([
            html.H5("1. Global View (Click on a Peak to inspect)"),
            dcc.Graph(id='global-graph', style={'height': '350px'})
        ])
    ]),

    html.Hr(),

    # 局部详情视图：对比与残差
    dbc.Row([
        # 左图：模型输入 vs 重构
        dbc.Col([
            html.H5("2. Input (Green) vs Reconstruction (Orange)"),
            dcc.Graph(id='recon-graph', style={'height': '400px'})
        ], width=6),

        # 右图：残差信号
        dbc.Col([
            html.H5("3. Residual (Input - Recon)"),
            dcc.Graph(id='resid-graph', style={'height': '400px'})
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("4. Residual Wavelet Transform (CWT Magnitude)"),
            dcc.Graph(id='cwt-graph', style={'height': '400px'})
        ], width=12)
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
    # 用于决定 Peak 显示的颜色 (Normal/Abnormal/HighError)
    mses = np.zeros(len(peaks))

    if len(peaks) > 0 and MODEL is not None:
        try:
            segments = []
            for p in peaks:
                # 提取 Peak 对应的片段 [end-SEQ_LEN : end]
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

    # (A) 原始波形 - 正常部分 (蓝色底)
    # 直接画整条线为蓝色，作为 base
    fig.add_trace(go.Scatter(y=sig, name='Normal Signal', line=dict(color='#1f77b4', width=1)))

    # (B) 原始波形 - 异常部分 (红色叠加)
    # 利用 NaN 截断不连续的线段
    sig_abnormal = sig.copy().astype(float)
    sig_abnormal[lab == 0] = np.nan  # 将正常部分设为 NaN，使其不显示
    # 只绘制 Label > 0 的部分
    fig.add_trace(go.Scatter(y=sig_abnormal, name='Abnormal Signal', line=dict(color='#d62728', width=1.5)))

    # (C) Peaks 标记 (X)
    fig.add_trace(go.Scatter(
        x=peaks, y=sig[peaks],
        mode='markers', name='Peaks',
        # symbol='x', 加粗线条 (width=3) 确保黄色在白底下可见
        marker=dict(size=10, color=peak_colors, symbol='x', line=dict(width=3, color=peak_colors)),
        customdata=peaks  # 存储索引，点击事件使用
    ))

    # 计算一些统计信息
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
    Output('cwt-graph', 'figure'),
    Input('global-graph', 'clickData'),
    State('current-signal-store', 'data')
)
def update_detail_view(clickData, sig_list):
    # 默认空图
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white")

    if not clickData or not sig_list:
        return empty_fig, empty_fig, empty_fig

    sig = np.array(sig_list)
    click_x = int(clickData['points'][0]['x'])

    # 对齐逻辑：以点击点为窗口的终点 (Peak at end)
    # 窗口: [click_x - SEQ_LEN + 1 : click_x + 1]
    end_idx = click_x + 1
    start_idx = end_idx - SEQ_LEN

    # 处理边界 padding
    if start_idx < 0:
        raw_seg = sig[:end_idx]
        pad_len = SEQ_LEN - len(raw_seg)
        raw_seg = np.pad(raw_seg, (pad_len, 0), 'constant')
    else:
        raw_seg = sig[start_idx:end_idx]

    # 1. 预处理：归一化 (Input) -> [-1, 1]
    norm_input = normalize_min_max(raw_seg)

    # 2. 模型推理 (Reconstruction + Residual)
    residual, recon = compute_residual(norm_input)

    # === 绘图 1: 对比图 ===
    fig_rec = go.Figure()
    # 输入波形
    fig_rec.add_trace(go.Scatter(
        y=norm_input, mode='lines',
        name='Normalized Input',
        line=dict(color='green', width=2)
    ))
    # 重构波形
    fig_rec.add_trace(go.Scatter(
        y=recon, mode='lines',
        name='Reconstruction',
        line=dict(color='orange', width=2, dash='dash')  # 虚线以便观察重合度
    ))
    # 标记末尾点(Peak位置)
    fig_rec.add_trace(go.Scatter(
        x=[SEQ_LEN - 1], y=[norm_input[-1]],
        mode='markers', marker=dict(color='red', size=8),
        showlegend=False
    ))

    fig_rec.update_layout(
        title=f"Reconstruction at Peak {click_x}",
        xaxis_title="Time Step (0-447)",
        yaxis_title="Normalized Amplitude [-1, 1]",
        template="plotly_white",
        legend=dict(x=0, y=1),
        yaxis=dict(range=[-1.2, 1.2])  # 固定尺度
    )

    # === 绘图 2: 残差图 ===
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        y=residual, mode='lines',
        name='Residual',
        line=dict(color='#d62728', width=1.5)  # 红色
    ))

    # 计算 MSE 作为参考
    mse_val = np.mean(residual ** 2)
    fig_res.update_layout(
        title=f"Residual Signal (MSE: {mse_val:.5f})",
        xaxis_title="Time Step",
        yaxis_title="Error (Input - Recon)",
        template="plotly_white",
        yaxis=dict(range=[-1.2, 1.2])  # 固定尺度
    )

    # === 绘图 3: 残差小波变换 (CWT) ===
    # 尺度范围 1-64，使用复Morlet小波提取特征
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(residual, scales, 'cmor1.5-1.0')
    cwt_power = np.abs(coef)  # 取模值

    fig_cwt = go.Figure(data=go.Heatmap(
        z=cwt_power,
        x=np.arange(len(residual)),
        y=scales,
        colorscale='Viridis',
        colorbar=dict(title="Magnitude")
    ))
    fig_cwt.update_layout(
        title="Residual CWT Scalogram",
        xaxis_title="Time Step",
        yaxis_title="Scale (Low Freq -> High Freq)",
        template="plotly_white"
    )

    return fig_rec, fig_res, fig_cwt


def parse_args():
    parser = argparse.ArgumentParser(description="AFDD Residual Analysis Tool")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help="Path to H5 data file")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH, help="Path to .pt/.pth model checkpoint")
    parser.add_argument('--port', type=int, default=8051, help="Web server port")
    parser.add_argument('--min_delta', type=float, default=float(MIN_VAL_TH), help="Peak detection threshold")
    parser.add_argument('--distance', type=int, default=100, help="Peak detection distance")
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()

    # 1. 检查数据
    if not os.path.exists(ARGS.data_path):
        print(f"Error: 数据文件未找到: {ARGS.data_path}")
        sys.exit(1)

    # 2. 加载模型 (只需加载一次)
    MODEL = load_model(ARGS.checkpoint)
    if MODEL is None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("警告: 模型加载失败！")
        print("右侧的 'Residual' 图表将只显示全0直线。")
        print("请检查 --checkpoint 参数路径是否正确。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # 3. 启动应用
    print(f"Starting Dashboard...")
    print(f" > Data: {ARGS.data_path}")
    print(f" > Model: {ARGS.checkpoint}")
    print(f" > URL: http://0.0.0.0:{ARGS.port}")

    # 修正 'run_server' 为 'run' (Dash v2)
    # debug=True 允许热重载，方便调试
    app.run(debug=True, port=ARGS.port, host='0.0.0.0')
