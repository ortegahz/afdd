# FILE: visualize_peaks_residual.py

import argparse
import os
import sys

import dash
import dash_bootstrap_components as dbc
import h5py
import numpy as np
import plotly.graph_objects as go
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
DEFAULT_DATA_PATH = "/media/manu/ST8000DM004-2U91/tmp/afd.h5.v2"
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
    with h5py.File(path, 'r') as f:
        if key not in f: return None
        return f[key]['signal'][:]


def normalize_min_max(segment):
    """归一化到 [-1, 1]"""
    mi, ma = np.min(segment), np.max(segment)
    if ma == mi: return np.zeros_like(segment)
    return 2 * (segment - mi) / (ma - mi) - 1


def load_model(ckpt_path):
    """
    加载 Stage 1 MemoryAE 模型。
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

    sig = get_signal_data(ARGS.data_path, key)
    if sig is None:
        return go.Figure(), None, "Error loading signal."

    # 寻找峰值用于标记
    peaks, _ = find_peaks(sig, prominence=ARGS.min_delta, distance=ARGS.distance)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sig, name='Raw Signal', line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(
        x=peaks, y=sig[peaks],
        mode='markers', name='Peaks',
        marker=dict(size=8, color='red', symbol='x-thin', line=dict(width=2)),
        customdata=peaks  # 存储索引，点击事件使用
    ))

    fig.update_layout(
        title=f"Sample: {key} (Found {len(peaks)} peaks)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )

    return fig, sig.tolist(), f"Loaded {key} with {len(peaks)} peaks."


@app.callback(
    Output('recon-graph', 'figure'),
    Output('resid-graph', 'figure'),
    Input('global-graph', 'clickData'),
    State('current-signal-store', 'data')
)
def update_detail_view(clickData, sig_list):
    # 默认空图
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white")

    if not clickData or not sig_list:
        return empty_fig, empty_fig

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

    # 1. 预处理：归一化 (Input)
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
        yaxis_title="Normalized Amplitude",
        template="plotly_white",
        legend=dict(x=0, y=1)
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
        template="plotly_white"
    )

    return fig_rec, fig_res


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
