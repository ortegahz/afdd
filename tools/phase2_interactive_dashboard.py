# FILE: phase2_interactive_dashboard.py

import argparse
import logging
import os
import sys

# --- Dash 相关库 ---
import dash
import h5py
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from dash import dcc, html, Input, Output, no_update
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.macros import RECONS_TH

# --- 路径修正 ---
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
    parser = argparse.ArgumentParser(description="Phase 2 交互式仪表盘：特征空间与原始波形联动分析")

    # 路径参数 (默认值按你提供的)
    parser.add_argument('--path_ckpt_base', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_base.pt")
    parser.add_argument('--path_ckpt_head', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_head.pt")
    parser.add_argument('--path_ckpt_clf', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_clf.pt")
    parser.add_argument('--data_path', type=str,
                        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3")

    # 其他配置
    parser.add_argument('--ae_model_type', type=str, default='ae', choices=['ae', 'unet'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=-1, help="为了交互流畅，建议不要超过10000个点")
    parser.add_argument('--port', type=int, default=8050, help="Dash服务端口")

    return parser.parse_args()


class DataManager:
    """管理数据加载、模型推理和缓存，供Dash回调使用"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.tsne_result = None
        self.metadata = []  # 存储每个点的元数据：(index, label, error, prob, key, offset)
        self.mem_centers = None

        self.load_models()
        self.process_data()

    def load_models(self):
        logging.info(">>> 正在加载模型...")
        # 1. Base AE
        model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
        self.base_model = model_map[self.args.ae_model_type]().to(self.device).eval()
        state_base = torch.load(self.args.path_ckpt_base, map_location=self.device)
        self.base_model.load_state_dict({k.replace('module.', ''): v for k, v in state_base.items()})

        # Get latent dim
        with torch.no_grad():
            _, dummy_z, _ = self.base_model(torch.randn(1, 1, 448).to(self.device))
            latent_dim = dummy_z.shape[1]

        # 2. Memory Head
        self.mem_head = MemoryHead(latent_dim, mem_dim=1024, hidden_dim=latent_dim).to(self.device).eval()
        state_head = torch.load(self.args.path_ckpt_head, map_location=self.device)
        self.mem_head.load_state_dict({k.replace('module.', ''): v for k, v in state_head.items()})

        # 3. Aux Classifier
        # s=30.0, m=0.5 必须与训练时保持一致
        self.aux_clf = ArcMarginProduct(latent_dim, 2, s=30.0, m=0.50).to(self.device).eval()
        if os.path.exists(self.args.path_ckpt_clf):
            state_clf = torch.load(self.args.path_ckpt_clf, map_location=self.device)
            self.aux_clf.load_state_dict({k.replace('module.', ''): v for k, v in state_clf.items()})
        else:
            logging.warning("未使用 AuxClassifier 权重，使用随机初始化。")

    def process_data(self):
        logging.info(">>> 正在处理数据与提取特征...")
        gen = FeaturesGeneratorCNN()
        # 注意：这里不仅需要提取特征，还需要知道每个样本对应HDF5里的哪个位置
        # HDF5PeakAlignedDataset.samples 存储了 [(key, end_idx), ...]
        dataset = HDF5PeakAlignedDataset(
            hdf5_file_path=self.args.data_path,
            transform=gen.transform_sample_ae,
            seq_len=gen.seq_len,
            min_delta=MIN_VAL_TH
        )

        # 使用 shuffle=False 确保 DataLoader 的顺序与 dataset.samples 一致
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        z_list, labels_list, errors_list, probs_list = [], [], [], []
        global_indices = []  # 记录原始dataset中的索引

        total_batches = len(loader)
        current_idx = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(loader)):
                if self.args.max_samples > 0 and current_idx >= self.args.max_samples:
                    break

                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)

                # Forward
                recons, latents, _ = self.base_model(inputs)
                z = self.mem_head(latents)
                # ArcFace Inference: label=None returns raw scaled logits [B, 2]
                logits = self.aux_clf(z, label=None)
                probs = F.softmax(logits, dim=1)[:, 1]

                # Metrics
                errs = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

                # Store
                z_list.append(z.cpu())
                labels_list.append(labels)
                errors_list.append(errs.cpu())
                probs_list.append(probs.cpu())

                # 记录这一个Batch对应的原始索引范围
                global_indices.extend(range(current_idx, current_idx + batch_size))
                current_idx += batch_size

        # Concat
        z_all = torch.cat(z_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0).view(-1).numpy()
        errors_all = torch.cat(errors_list, dim=0).numpy()
        probs_all = torch.cat(probs_list, dim=0).numpy()
        indices_all = np.array(global_indices)

        # --- Filter: Error > 0.001 (Hard Examples) ---
        # 修改逻辑：保留所有异常样本(Label=1) 以及 重构误差大的正常样本
        # 这样可以在仪表盘中观察到那些重构得很好但被分类器识别出来的故障
        mask = (errors_all > RECONS_TH) | (labels_all == 1)

        z_filtered = z_all[mask]
        labels_filtered = labels_all[mask]
        errors_filtered = errors_all[mask]
        probs_filtered = probs_all[mask]
        indices_filtered = indices_all[mask]

        # 截断（如果过滤后还太多）
        if self.args.max_samples > 0 and len(z_filtered) > self.args.max_samples:
            z_filtered = z_filtered[:self.args.max_samples]
            labels_filtered = labels_filtered[:self.args.max_samples]
            errors_filtered = errors_filtered[:self.args.max_samples]
            probs_filtered = probs_filtered[:self.args.max_samples]
            indices_filtered = indices_filtered[:self.args.max_samples]

        logging.info(f"最终用于可视化的样本数: {len(z_filtered)}")

        # --- Memory Slots & Distance ---
        self.mem_centers = self.mem_head.memory.detach().cpu().numpy()

        # 计算每个样本距离最近的 Slot 的距离
        # (简单计算欧氏距离，为了显示用)
        # z: [N, D], mem: [M, D]
        # 由于点数可能较多，这里只存特征供后续查询，或者算好
        logging.info("计算与Memory距离...")
        dists = []
        # 分批计算距离矩阵防止OOM
        z_np = z_filtered.numpy()
        for i in range(0, len(z_np), 1000):
            batch_z = torch.from_numpy(z_np[i:i + 1000]).to(self.device)  # [B, D]
            mem_cuda = self.mem_head.memory  # [M, D]
            # dist = ||z - m||^2
            # 扩展 broadcasting
            dist_mat = torch.cdist(batch_z, mem_cuda, p=2)  # [B, M]
            min_vals, _ = torch.min(dist_mat, dim=1)
            dists.append(min_vals.cpu().numpy())
        dists_all = np.concatenate(dists)

        # --- t-SNE ---
        tsne_input = np.vstack([z_filtered.numpy(), self.mem_centers])
        logging.info("Running t-SNE...")
        tsne = TSNE(n_components=3, perplexity=self.args.perplexity, n_iter=1000, n_jobs=-1)
        tsne_res = tsne.fit_transform(tsne_input)

        self.tsne_samples = tsne_res[:-len(self.mem_centers)]
        self.tsne_centers = tsne_res[-len(self.mem_centers):]

        # --- Pack Metadata ---
        # 我们需要保留足够的信息以便在回调函数中读取原始HDF5
        self.metadata = []
        for i in range(len(indices_filtered)):
            orig_idx = indices_filtered[i]
            # 从 dataset.samples 获取 (h5_key, end_idx)
            h5_key, end_idx = dataset.samples[orig_idx]

            self.metadata.append({
                'id': i,
                'orig_dataset_idx': orig_idx,
                'h5_key': h5_key,
                'end_idx': end_idx,
                'label': int(labels_filtered[i]),
                'error': float(errors_filtered[i]),
                'prob': float(probs_filtered[i]),
                'min_dist': float(dists_all[i])
            })

        logging.info("数据准备就绪。")

    def get_raw_waveform(self, meta_idx):
        """根据元数据索引读取原始波形"""
        meta = self.metadata[meta_idx]
        seq_len = 448  # Hardcoded or from config
        # 如果长度不够，dataset那边应该已经处理了，这里假设 idx 有效
        start_idx = meta['end_idx'] - seq_len

        with h5py.File(self.args.data_path, 'r') as f:
            sig = f[meta['h5_key']]['signal'][start_idx: meta['end_idx']]
        return sig


# --- 初始化 ---
args = parse_args()
dm = DataManager(args)

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# --- 布局 ---
app.layout = html.Div([
    html.H2("AFDD Phase 2 深度分析仪表盘 (ArcFace Enhanced)", style={'textAlign': 'center'}),

    html.Div([
        # 左侧：3D 特征空间
        html.Div([
            dcc.Graph(id='3d-scatter', style={'height': '80vh'}),
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # 右侧：详细信息与波形
        html.Div([
            html.Div(id='info-box', style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px',
                                           'borderRadius': '5px'}),
            dcc.Graph(id='waveform-plot', style={'height': '40vh'}),
        ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '10px'})
    ])
])


# --- 生成 3D 图表数据 ---
def get_3d_figure():
    tsne_s = dm.tsne_samples
    tsne_c = dm.tsne_centers
    meta = dm.metadata

    labels = np.array([m['label'] for m in meta])
    probs = np.array([m['prob'] for m in meta])

    # 颜色映射：正常为蓝，异常根据概率深浅（或直接红）
    # 为了突出分类器效果，我们用分类概率给异常点上色
    colors = []
    sizes = []
    symbols = []

    for i, l in enumerate(labels):
        if l == 0:
            colors.append('dodgerblue')
            sizes.append(3)
            symbols.append('circle')
        else:
            # 异常样本：颜色越红表示概率越高
            p = probs[i]
            colors.append(f'rgba({255}, {int(255 * (1 - p))}, 0, 0.8)')  # Prob越高越红
            sizes.append(5)
            symbols.append('diamond')

    # Trace 1: 样本点
    # 重要：customdata 存入索引，以便回调函数知道鼠标指向了谁
    trace_samples = go.Scatter3d(
        x=tsne_s[:, 0], y=tsne_s[:, 1], z=tsne_s[:, 2],
        mode='markers',
        marker=dict(size=sizes, color=colors, symbol=symbols, opacity=0.7),
        customdata=list(range(len(meta))),
        text=[f"Loss: {m['error']:.4f}<br>Prob (ArcFace): {m['prob']:.4f}" for m in meta],
        hoverinfo='text',
        name='Samples'
    )

    # Trace 2: Memory Slots (背景)
    trace_centers = go.Scatter3d(
        x=tsne_c[:, 0], y=tsne_c[:, 1], z=tsne_c[:, 2],
        mode='markers',
        marker=dict(size=4, color='black', symbol='cross', opacity=0.3),
        hoverinfo='skip',
        name='Memory Slots'
    )

    layout = go.Layout(
        title='Phase 2 Feature Space (t-SNE)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1)
    )

    return go.Figure(data=[trace_centers, trace_samples], layout=layout)


# 初始化图表
app.layout['3d-scatter'].figure = get_3d_figure()


# --- 回调函数 ---
@app.callback(
    [Output('waveform-plot', 'figure'),
     Output('info-box', 'children')],
    [Input('3d-scatter', 'hoverData')]
)
def display_hover_data(hoverData):
    if hoverData is None:
        return go.Figure(), "请将鼠标悬停在左侧图表的数据点上查看详情。"

    # 获取索引
    point = hoverData['points'][0]

    # 检查是否是样本点（曲线编号 1 是样本，0 是 MemorySlots）
    # Plotly trace 顺序：0->Memory, 1->Samples
    curve_num = point.get('curveNumber', -1)

    if curve_num != 1:
        return no_update, "当前选中的是记忆中心，无波形数据。"

    # 获取 customdata，这对应我们在 DM.metadata 里的索引
    idx = point.get('customdata')
    if idx is None:
        return no_update, "数据索引错误"

    meta = dm.metadata[idx]

    # 1. 读取波形
    try:
        waveform = dm.get_raw_waveform(idx)
    except Exception as e:
        return go.Figure(), f"读取波形失败: {str(e)}"

    # 2. 绘制波形
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(y=waveform, mode='lines', line=dict(color='black', width=1.5)))
    fig_wave.update_layout(
        title=f"Raw Waveform (Label: {'Abnormal' if meta['label'] == 1 else 'Normal'})",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=20, b=40, t=40),
        height=300
    )

    # 3. 构造信息面板
    label_text = "🔴 故障电弧 (Abnormal)" if meta['label'] == 1 else "🔵 正常信号 (Normal)"

    info_children = [
        html.H4(label_text),
        html.P(f"HDF5 Key: {meta['h5_key']}"),
        html.P(f"Slice Offset: {meta['end_idx']}"),
        html.Hr(),
        html.P([html.B("ArcFace 异常概率: "), f"{meta['prob']:.6f}"]),
        html.P([html.B("AE 重构误差: "), f"{meta['error']:.6f}"]),
        html.P([html.B("距最近Slot距离: "), f"{meta['min_dist']:.4f}"]),
    ]

    return fig_wave, info_children


if __name__ == '__main__':
    print(f"Server starting on port {args.port}...")
    # debug=False 在 Notebook 或远程服务器中更稳定
    app.run(debug=False, host='0.0.0.0', port=args.port)
