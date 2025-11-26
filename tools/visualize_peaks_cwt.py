# FILE: visualize_peaks_cwt.py

import argparse
import os
import sys

# Dash 库
import dash
import dash_bootstrap_components as dbc
import h5py
import numpy as np
import plotly.graph_objects as go
import pywt
from dash import dcc, html, Input, Output, State
from scipy.signal import find_peaks

# --- 配置部分 ---
SEQ_LEN = 448  # 切片长度
WAVELET = 'cmor1.5-1.0'
DEFAULT_DATA_PATH = "/media/manu/ST8000DM004-2U91/tmp/afd.h5.v2"
MIN_VAL_TH = 100  # 峰值阈值


# --- 辅助函数 ---

def load_keys(path):
    if not os.path.exists(path):
        return []
    with h5py.File(path, 'r') as f:
        return list(f.keys())


def get_signal_data(path, key):
    with h5py.File(path, 'r') as f:
        if key not in f:
            return None, None
        signal = f[key]['signal'][:]
        # label = f[key]['label_seq'][:] # 如果需要
    return signal


def normalize_min_max(segment):
    """
    对切片进行 Min-Max 归一化，模拟训练时的预处理
    """
    mi = np.min(segment)
    ma = np.max(segment)
    if ma == mi:
        return np.zeros_like(segment)
    return (segment - mi) / (ma - mi)


def compute_cwt(segment):
    # 针对448长度，使用合适的scale
    scales = np.arange(1, 48)
    coef, freqs = pywt.cwt(segment, scales, WAVELET)
    return np.abs(coef)


# --- Dash App 初始化 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 全局参数容器 (实际部署建议不用全局变量，简单脚本可接受)
ARGS = None

# --- 布局 Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("AFDD Peak Analysis Dashboard"), width=12)
    ], className="my-3"),

    # 控制区
    dbc.Row([
        dbc.Col([
            html.Label("Select H5 File Key (Sample):"),
            dcc.Dropdown(
                id='key-dropdown',
                options=[],  # 动态填充
                placeholder="Loading keys...",
            ),
        ], width=8),
        dbc.Col([
            html.Br(),
            html.Div(id='status-text', style={'color': 'gray'})
        ], width=4)
    ]),

    html.Hr(),

    # 图表区 1: 全局视图
    dbc.Row([
        dbc.Col([
            html.H5("1. Global View (Click on a Peak to inspect)"),
            dcc.Graph(id='global-graph', style={'height': '400px'})
        ])
    ]),

    # 图表区 2: 细节视图 (左右并列)
    dbc.Row([
        dbc.Col([
            html.H5("2. Selected Slice (Min-Max Normalized)"),
            # 这里的切片是用于模型输入的最终形态
            dcc.Graph(id='slice-graph', style={'height': '400px'})
        ], width=6),
        dbc.Col([
            html.H5("3. CWT Scalogram (Time-Freq)"),
            dcc.Graph(id='cwt-graph', style={'height': '400px'})
        ], width=6)
    ], className="mt-4"),

    # 隐藏存储，用于记录当前选中的 Peak Index
    dcc.Store(id='current-signal-store'),

], fluid=True)


# --- 回调逻辑 ---

# 1. 初始化：脚本启动加载 Keys
@app.callback(
    Output('key-dropdown', 'options'),
    Output('key-dropdown', 'value'),
    Input('key-dropdown', 'search_value')
)
def init_data(search_val):
    # 这里做了一个简化，只加载一次。
    # 实际应用中如果不希望每次刷新页面都加载，可以在全局载入。
    keys = load_keys(ARGS.data_path)
    options = [{'label': k, 'value': k} for k in keys]
    # 默认选第1个
    value = keys[0] if keys else None
    return options, value


# 2. 当选择了 Key，加载原始信号并绘制全局图
@app.callback(
    Output('global-graph', 'figure'),
    Output('current-signal-store', 'data'),
    Output('status-text', 'children'),
    Input('key-dropdown', 'value')
)
def update_global_view(key):
    if not key:
        return go.Figure(), None, "No data selected."

    signal = get_signal_data(ARGS.data_path, key)
    if signal is None:
        return go.Figure(), None, "Error loading signal."

    # 找峰值
    peaks, _ = find_peaks(signal, prominence=ARGS.min_delta, distance=ARGS.distance)

    # 绘图
    fig = go.Figure()
    # 原始波形
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Raw Signal', line=dict(color='royalblue', width=1)))
    # 峰值标记
    fig.add_trace(go.Scatter(
        x=peaks,
        y=signal[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(size=10, color='red', symbol='x-thin', line=dict(width=2)),
        # 将 peak index 存入 customdata，方便点击回调获取
        customdata=peaks
    ))

    fig.update_layout(
        title=f"Full Sequence: {key} (Found {len(peaks)} peaks)",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        template="plotly_white"
    )

    # 将信号转为 list 存入浏览器缓存 (store)，避免重复读取 IO
    # 注意：如果信号特别长 (>100k点)，store 可能会有点卡，建议改回 IO 读取
    # 这里假设信号长度适中 (e.g. < 50k)
    return fig, signal.tolist(), f"Loaded {key}"


# 3. 当点击了全局图的某个点，更新下方的切片图和 CWT
@app.callback(
    Output('slice-graph', 'figure'),
    Output('cwt-graph', 'figure'),
    Input('global-graph', 'clickData'),
    State('current-signal-store', 'data')
)
def update_detail_view(clickData, signal_list):
    if not clickData or not signal_list:
        return go.Figure(), go.Figure()

    signal = np.array(signal_list)

    # 获取点击点的坐标
    point = clickData['points'][0]
    click_x = point['x']

    # 确认点击的是 peak 还是 line
    # 如果点击的是 line，我们需要找最近的 peak？
    # 或者我们强制用户点击红色的叉叉。
    # 这里我们做一个智能判断：以 click_x 为中心，寻找最近的有效 peak
    # 但为了简单，我们假设用户点击的是 Peak Marker (Trace 1)

    target_peak_idx = int(click_x)

    # 检查是否越界 (前面是否有足够长度)
    if target_peak_idx < SEQ_LEN - 1:
        # 处理边界情况：如果长度不够，可以 Pad，或者显示不可用
        # 这里做 Zero Padding
        start_idx = 0
        end_idx = target_peak_idx + 1
        segment = signal[start_idx:end_idx]
        # Pad 前面
        pad_len = SEQ_LEN - len(segment)
        segment = np.pad(segment, (pad_len, 0), 'constant')
        display_idx = range(start_idx - pad_len, end_idx)  # 虚拟坐标
    else:
        start_idx = target_peak_idx - SEQ_LEN + 1
        end_idx = target_peak_idx + 1
        segment = signal[start_idx:end_idx]
        display_idx = range(start_idx, end_idx)

    # --- 核心处理：归一化 ---
    norm_segment = normalize_min_max(segment)

    # --- 核心处理：CWT ---
    cwt_img = compute_cwt(norm_segment)

    # === 左图：归一化切片 ===
    fig_slice = go.Figure()
    fig_slice.add_trace(go.Scatter(
        x=list(range(SEQ_LEN)),  # 这里的 X 轴变为 0~447 (模型视角)
        y=norm_segment,
        mode='lines',
        name='Normalized Input',
        line=dict(color='green')
    ))
    # 标记最后一个点是 Peak
    fig_slice.add_trace(go.Scatter(
        x=[SEQ_LEN - 1], y=[norm_segment[-1]], mode='markers', marker=dict(color='red', size=10)
    ))
    fig_slice.update_layout(
        title=f"Model Input (Window 448) at Peak {target_peak_idx} <br>(Min-Max Normalized)",
        xaxis_title="Window Step (0-447)",
        yaxis_title="Normalized Amp",
        template="plotly_white"
    )

    # === 右图：CWT ===
    fig_cwt = go.Figure()
    fig_cwt.add_trace(go.Heatmap(
        z=cwt_img,
        x=list(range(SEQ_LEN)),
        colorscale='Jet',
        showscale=True
    ))
    fig_cwt.update_layout(
        title=f"CWT Spectrum of Normalized Slice",
        xaxis_title="Window Step",
        yaxis_title="Scale (Low <-> High Freq)",
        template="plotly_white"
    )

    return fig_slice, fig_cwt


def parse_args():
    parser = argparse.ArgumentParser(description="Run AFDD Analysis Dashboard")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--min_delta', type=float, default=float(MIN_VAL_TH))
    parser.add_argument('--distance', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()

    if not os.path.exists(ARGS.data_path):
        print(f"Error: Data file not found at {ARGS.data_path}")
        sys.exit(1)

    print(f"Starting Dashboard...")
    print(f"Data Source: {ARGS.data_path}")
    print(f"Open your browser at: http://127.0.0.1:{ARGS.port}")

    # debug=True 方便调试，改代码后自动刷新
    app.run(debug=True, port=ARGS.port)
