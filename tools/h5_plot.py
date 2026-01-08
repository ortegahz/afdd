import argparse
import logging
import os

import dash
import h5py
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="HDF5 数据可视化仪表盘")

    # 默认路径设置为您代码中生成的路径
    parser.add_argument('--data_path', type=str,
                        default="/home/manu/tmp/afd_h5/train_data.h5",
                        help="HDF5 文件路径")

    parser.add_argument('--port', type=int, default=8050, help="Dash服务端口")
    parser.add_argument('--max_samples', type=int, default=1000, help="下拉列表中显示的最大样本数")

    return parser.parse_args()


class DataManager:
    """管理 HDF5 数据加载"""

    def __init__(self, args):
        self.args = args
        self.file_path = args.data_path
        self.keys = []
        self.labels = []

        self.load_metadata()

    def load_metadata(self):
        if not os.path.exists(self.file_path):
            logging.error(f"文件不存在: {self.file_path}")
            return

        logging.info(f"正在读取文件: {self.file_path}")
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 获取所有 sample_xxx 的 key
                all_keys = list(f.keys())

                # 简单的自然排序 (sample_0, sample_1, ... sample_10)
                # 假设 key 格式为 'sample_{int}'
                try:
                    all_keys.sort(key=lambda x: int(x.split('_')[1]))
                except:
                    all_keys.sort()

                # 限制加载数量，防止下拉列表过长卡顿
                self.keys = all_keys[:self.args.max_samples]

                # 预读取 Label 以便在下拉框中显示状态
                for k in self.keys:
                    lbl = f[k]['label'][()]
                    self.labels.append(int(lbl))

            logging.info(f"已加载 {len(self.keys)} 个样本的元数据。")

        except Exception as e:
            logging.error(f"读取 HDF5 失败: {e}")

    def get_sample_data(self, key):
        """读取指定 Key 的详细数据"""
        data = {}
        try:
            with h5py.File(self.file_path, 'r') as f:
                group = f[key]
                data['signal'] = group['signal'][:]
                data['label_seq'] = group['label_seq'][:]
                data['label'] = int(group['label'][()])

                # 如果有电压数据或其他数据，也可以在这里读取
                # data['voltage'] = group['voltage'][:] if 'voltage' in group else None
        except Exception as e:
            logging.error(f"读取样本 {key} 失败: {e}")
            return None
        return data


# --- 初始化 ---
args = parse_args()
dm = DataManager(args)

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# --- 布局 ---
app.layout = html.Div([
    html.H2("Arc Fault HDF5 Data Viewer", style={'textAlign': 'center'}),

    html.Div([
        # 顶部控制栏
        html.Div([
            html.Label("选择样本 (Sample):"),
            dcc.Dropdown(
                id='sample-dropdown',
                options=[
                    {'label': f"{k} [{'ARC' if l == 1 else 'NORMAL'}]", 'value': k}
                    for k, l in zip(dm.keys, dm.labels)
                ],
                value=dm.keys[0] if dm.keys else None,
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'width': '50%', 'margin': '0 auto', 'paddingBottom': '20px'}),

        # 信息展示区
        html.Div(id='info-box', style={
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': '#f0f0f0',
            'margin': '10px auto',
            'width': '80%',
            'borderRadius': '5px'
        }),

        # 图表区
        dcc.Graph(id='waveform-plot', style={'height': '70vh'}),
    ])
])


# --- 回调函数 ---
@app.callback(
    [Output('waveform-plot', 'figure'),
     Output('info-box', 'children')],
    [Input('sample-dropdown', 'value')]
)
def update_graph(selected_key):
    if not selected_key:
        return go.Figure(), "未选择数据或文件为空"

    # 读取数据
    data = dm.get_sample_data(selected_key)
    if data is None:
        return go.Figure(), f"读取 {selected_key} 错误"

    signal = data['signal']
    label_seq = data['label_seq']
    global_label = data['label']

    # 创建图表
    fig = go.Figure()

    # Trace 1: 原始信号 (Signal)
    fig.add_trace(go.Scatter(
        y=signal,
        mode='lines',
        name='Current Signal',
        line=dict(color='dodgerblue', width=1.5),
        opacity=0.8
    ))

    # Trace 2: 序列标签 (Label Sequence) - 使用次坐标轴或缩放显示
    # 为了直观，我们将 label_seq 映射到信号的幅度范围，或者使用填充区域

    # 方法 A: 使用次坐标轴显示 Label (0 或 1)
    fig.add_trace(go.Scatter(
        y=label_seq,
        mode='lines',
        name='GT Label Seq',
        line=dict(color='red', width=1.5, shape='hv'),  # hv step line
        yaxis='y2',
        fill='tozeroy',  # 填充下方区域
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))

    # 布局设置
    title_text = f"Sample: {selected_key} | Global Label: {'🔴 ARC FAULT' if global_label == 1 else '🔵 NORMAL'}"

    fig.update_layout(
        title=title_text,
        xaxis_title="Time Step",
        yaxis=dict(
            title="Signal Amplitude",
            side="left"
        ),
        yaxis2=dict(
            title="Ground Truth (0/1)",
            side="right",
            overlaying="y",
            range=[-0.1, 1.1],  # 固定 0-1 范围
            showgrid=False
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified"
    )

    # 信息框内容
    info_text = [
        html.Span(f"Key: {selected_key}", style={'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f"Length: {len(signal)} pts", style={'marginRight': '20px'}),
        html.Span(f"Max Val: {np.max(signal):.2f}", style={'marginRight': '20px'}),
        html.Span(f"Min Val: {np.min(signal):.2f}", style={'marginRight': '20px'}),
        html.Span(f"Arc Points: {np.sum(label_seq > 0)}",
                  style={'color': 'red' if np.sum(label_seq > 0) > 0 else 'black'})
    ]

    return fig, info_text


if __name__ == '__main__':
    print(f"Server starting on port {args.port}...")
    print(f"Reading data from: {args.data_path}")
    app.run(debug=True, host='0.0.0.0', port=args.port)
