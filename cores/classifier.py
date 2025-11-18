# FILE: classifier.py

import logging
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import f1_score, roc_curve
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_qat_fx, convert_fx
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset, Subset

from cores.features_generator import (FeaturesGeneratorXGB, FeaturesGeneratorCNN, InferenceDataset,
                                      HDF5Dataset, HDF5SequentialSliceDataset, HDF5SPDataset,
                                      generate_augmented_samples)
from cores.loss import HardExampleMiningFocalLoss
from cores.nets import NetAFD, NetAFDAE, NetAFDAE_UNet, NetAFDAE_2D_MTF
from utils.macros import MIN_VAL_TH
from utils.utils import make_dirs


class ClassifierBase:
    def __init__(self):
        pass


class ClassifierXGB(ClassifierBase):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = None
        self.features_generator = FeaturesGeneratorXGB()

    def train(self, x, y, num_boost_round=100):
        data = self.features_generator.generate(x, y)
        self.model = xgb.train(self.params, data, num_boost_round=num_boost_round)

    def infer(self, x, y=None):
        data = self.features_generator.generate(x, y)
        return self.model.predict(data)


class ClassifierCNN(ClassifierBase):
    def __init__(self, args, ddp=False, is_infer=False):
        super().__init__()
        self.qat = args.qat if not is_infer else False
        self.local_rank = args.rank if not is_infer else 0
        self.num_epochs = 512
        self.lr = 1e-4
        float_model = NetAFD().to(self.local_rank)
        if self.qat and not is_infer:
            float_model.eval()
            example_inputs = (torch.randn(1, 1, 1, 448).to(self.local_rank),)
            # fused_model = fuse_fx(float_model, example_inputs)
            fused_model = fuse_fx(float_model)
            qconfig = get_default_qat_qconfig('qnnpack')  # arm
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            self.model = prepare_qat_fx(fused_model, qconfig_mapping, example_inputs).to(self.local_rank)
            self.model.train()
        else:
            self.model = float_model  # 普通浮点训练 / 推理

        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        if is_infer:
            self._load_checkpoint_v0(args)
        elif args.path_ckpt is not None:
            self._load_checkpoint_v0(args.path_ckpt)
            # self._load_checkpoint_v1(ckpt)
        if ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.criterion = HardExampleMiningFocalLoss().to(self.local_rank)
        self.features_generator = FeaturesGeneratorCNN()
        self.rank = args.rank if not is_infer else 0
        self.save_dir = args.save_dir if not is_infer else None

    def _load_checkpoint_v0(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{self.local_rank}')
        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)

    def _load_checkpoint_v1(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.local_rank}')
        state_dict = checkpoint['model_state_dict']
        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint.get('epoch', 0)
        # best_accuracy = checkpoint.get('best_accuracy', 0.0)
        # logging.info(f'Checkpoint loaded: starting at epoch {start_epoch}, best_accuracy {best_accuracy:.4f}')
        # return start_epoch, best_accuracy

    def _save_checkpoint(self, path, epoch, best_accuracy):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, path)

    def _loss_computation(self, outputs, feats, labels):
        std_loss = self.criterion(outputs, labels)
        _labels = labels.squeeze(1)
        pos_feats = feats[_labels == 1]
        neg_feats = feats[_labels == 0]
        cos_loss = 0
        if len(pos_feats) > 0 and len(neg_feats) > 0:
            pos_feats = F.normalize(pos_feats, p=2, dim=1)
            neg_feats = F.normalize(neg_feats, p=2, dim=1)

            # Computes cosine similarity here to ensure it gets into the computation graph
            cos_sim = 1 - F.cosine_similarity(pos_feats.unsqueeze(1), neg_feats.unsqueeze(0), dim=2)
            cos_sim = F.relu(cos_sim)
            # Convert cos_sim into a loss term
            cos_loss = cos_sim.mean()

        _w_cos_loss = 0.1
        loss = (1 - _w_cos_loss) * std_loss + _w_cos_loss * cos_loss
        return std_loss, std_loss, cos_loss

    def train(self, data):
        if self.save_dir is not None and not os.path.exists(self.save_dir) and self.rank == 0:
            os.makedirs(self.save_dir)
        # dataset = self.features_generator.dataset_generate(x_train, y_train)
        dataset = HDF5Dataset(data['train_path'], self.features_generator.transform_sample)

        is_distributed = isinstance(self.model, DDP)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
        loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=not is_distributed, sampler=train_sampler)

        best_accuracy = 0.0
        for epoch in range(self.num_epochs):
            if self.qat:
                if epoch == 3:  # 经验值，可调
                    self.model.apply(torch.ao.quantization.disable_observer)
            epoch_start_time = time.time()
            if is_distributed:
                train_sampler.set_epoch(epoch)
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs, feats = self.model(inputs)
                loss, std_loss, cos_loss = self._loss_computation(outputs, feats, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # val_accuracy = self.evaluate(x_val, y_val)
            val_accuracy = self.evaluate(data['test_path'])
            epoch_duration = time.time() - epoch_start_time
            if self.qat and self.rank == 0:  # 只在主进程做
                # self.model.cpu().eval()  # INT8 kernel 只在 CPU
                # int8_model = convert_fx(self.model)
                # torch.save(int8_model.state_dict(),
                #            os.path.join(self.save_dir, 'qat_int8_final.pt'))
                gm = self.model.module if isinstance(
                    self.model, torch.nn.parallel.DistributedDataParallel) else self.model
                gm.eval()
                gm.apply(torch.ao.quantization.disable_observer)  # 停止收集 min/max
                # gm.apply(torch.ao.quantization.freeze_bn_stats)  # 固定 BN 的均值方差
                int8_model = convert_fx(gm)
                torch.save(int8_model.state_dict(), os.path.join(self.save_dir, 'qat_int8_final.pt'))
            if self.rank == 0:
                logging.info(
                    f'Epoch [{epoch + 1}/{self.num_epochs}],'
                    f' Loss std: {std_loss.item():.8f},'
                    f' Loss cos: {cos_loss.item():.8f},'
                    f' Loss: {loss.item():.8f},'
                    f' Validation Accuracy: {val_accuracy:.4f},'
                    f' Time: {epoch_duration:.2f} seconds')

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    _path_save = os.path.join(self.save_dir, f'best_e{epoch}_b{best_accuracy:.4f}.pt')
                    torch.save(self.model.state_dict(), _path_save)
                    # torch.save(self.model.state_dict(),
                    #            f'/home/Huangzhe/test/manu-pc/tmp/afdd_models_local/best_e{epoch}.pt')
                    if self.qat:
                        self.model.cpu().eval()  # INT8 kernel 只在 CPU
                        int8_model = convert_fx(self.model)
                        _path_save_int8 = os.path.join(self.save_dir, f'i8_best_e{epoch}_b{best_accuracy:.4f}.pt')
                        torch.save(int8_model.state_dict(), _path_save_int8)
                        logging.info(f'Saved new best model with accuracy: {best_accuracy:.4f}')

                # torch.save(self.model.state_dict(), f'/home/Huangzhe/test/manu-pc/tmp/afdd_models/{epoch}.pt')

                # self._save_checkpoint(f'/home/Huangzhe/test/manu-pc/tmp/afdd_models/{epoch}.pt', epoch, best_accuracy)

    # =============================================================
    # 2. 使用 onnxruntime 推理
    #    与 self.classifier.infer 保持同样的输入输出格式
    # =============================================================
    def infer_onnx(self,
                   seq: np.ndarray,
                   onnx_path: str = '/home/manu/tmp/classifier_sim.onnx',
                   batch_size: int = 1):
        """
        seq: 1-D numpy array，原始序列
        return: y_pred, hidden   # 两个输出
        """

        # ────────────────────── 1. 创建 / 缓存 session ──────────────────────
        if not hasattr(self, '_ort_session'):
            self._ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

        # ────────────────────── 2. 处理输入 ──────────────────────
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()

        if seq.dtype != np.float32:
            seq = seq.astype(np.float32)

        # (batch, 1, seq_len)  ——  注意多了一个通道维
        seq = seq.reshape(batch_size, 1, -1)

        # 拿到输入 / 输出名字，避免手写出错
        input_name = self._ort_session.get_inputs()[0].name  # 'input'
        output_names = [o.name for o in self._ort_session.get_outputs()]  # ['y_pred', 'hidden']

        # ────────────────────── 3. 推理 ──────────────────────
        ort_outputs = self._ort_session.run(output_names,
                                            {input_name: seq})

        # ort_outputs 就是一个 list，对应 output_names 的顺序
        y_pred, hidden = ort_outputs  # 也可以直接 return ort_outputs

        return y_pred, hidden

    # def infer(self, x, batch_size=16):
    #     dataset = InferenceDataset(x, transform=self.features_generator.transform_sample,
    #                                seq_len=self.features_generator.seq_len)
    #     loader = DataLoader(dataset, batch_size=batch_size)
    #     self.model.eval()
    #     predictions, features = [], []
    #     with torch.no_grad():
    #         for batch_x in loader:
    #             batch_x = batch_x.to(self.local_rank)
    #             outputs, feats = self.model(batch_x)
    #             outputs_onnx, feats_onnx = self.infer_onnx(batch_x)
    #             outputs_onnx, feats_onnx = torch.from_numpy(outputs_onnx), torch.from_numpy(feats_onnx)
    #             batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
    #             predictions.extend(batch_predictions)
    #             batch_feats = feats.flatten().cpu().numpy()
    #             features.extend(batch_feats)
    #     return np.array(predictions), np.array(features)

    def infer(self, x, batch_size=16,
              check_onnx=False,  # 是否做对齐校验
              rtol=1e-02, atol=1e-05  # allclose 误差阈值
              ):
        """
        返回:
            predictions : (N,) numpy  – sigmoid 概率
            features    : (N,) numpy  – 你模型返回的特征
        如果 check_onnx=True，会在每个 batch 上验证
        PyTorch 与 ONNX 输出(含特征)是否一致
        """
        dataset = InferenceDataset(
            x,
            transform=self.features_generator.transform_sample,
            seq_len=self.features_generator.seq_len
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        predictions, features = [], []

        max_diff_out, max_diff_feat = 0.0, 0.0  # 方便事后查看最大误差

        with torch.no_grad():
            for batch_x in loader:
                # ---------------- 1. PyTorch forward ----------------
                batch_x = batch_x.to(self.local_rank)
                out_pt, feat_pt = self.model(batch_x)  # Tensor, Tensor

                # ---------------- 2. ONNX forward ------------------
                # infer_onnx 里可以接收 Tensor，所以直接传
                out_onnx_np, feat_onnx_np = self.infer_onnx(batch_x)  # ndarray
                out_onnx = torch.from_numpy(out_onnx_np).to(out_pt.device)
                feat_onnx = torch.from_numpy(feat_onnx_np).to(feat_pt.device)

                # ---------------- 3. 一致性校验 --------------------
                if check_onnx:
                    try:
                        assert_close(out_pt, out_onnx, rtol=rtol, atol=atol)
                        assert_close(feat_pt, feat_onnx, rtol=rtol, atol=atol)
                    except AssertionError as e:
                        # 打印最大误差方便排查
                        diff_out = (out_pt - out_onnx).abs().max().item()
                        diff_feat = (feat_pt - feat_onnx).abs().max().item()
                        print(f"[WARN] Batch mismatch! "
                              f"max|Δoutput|={diff_out:.4e}, "
                              f"max|Δfeat|={diff_feat:.4e}")
                        raise  # 如果想继续跑可注释掉

                    # 统计全局最大 diff（可选）
                    max_diff_out = max(max_diff_out,
                                       (out_pt - out_onnx).abs().max().item())
                    max_diff_feat = max(max_diff_feat,
                                        (feat_pt - feat_onnx).abs().max().item())

                # ---------------- 4. 收集结果 ----------------------
                probs = torch.sigmoid(out_onnx).flatten().cpu().numpy()
                predictions.extend(probs)
                feats_np = feat_onnx.flatten().cpu().numpy()
                features.extend(feats_np)

        if check_onnx:
            print(f"[OK] PyTorch ↔ ONNX 校验通过，"
                  f"max|Δoutput|={max_diff_out:.4e}, "
                  f"max|Δfeat|={max_diff_feat:.4e}")

        return np.array(predictions), np.array(features)

    def evaluate(self, test_path, batch_size=1024, n_total_samples=-4096):
        # dataset = SignalDataset(x, y, transform=self.features_generator.transform_sample,
        #                         seq_len=self.features_generator.seq_len)
        dataset = HDF5Dataset(test_path, self.features_generator.transform_sample)
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            _cnt = 0
            for inputs, labels in loader:
                if _cnt > n_total_samples > 0:
                    break
                inputs = inputs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs, _ = self.model(inputs)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                all_predictions.extend(batch_predictions)
                all_labels.extend(labels.cpu().numpy())
                _cnt += batch_size
        predictions = [1 if prob > 0.5 else 0 for prob in all_predictions]
        # result = accuracy_score(all_labels, predictions)
        result = f1_score(all_labels, predictions)
        return result


class WeightedReconstructionLoss(nn.Module):
    """
    一种用于重构任务的损失函数，旨在关注重构误差较大的"硬"样本/像素点。
    它通过将逐元素的L1损失提升到一个可配置的幂(alpha)来实现这一点。
    类似于Focal Loss对分类任务的作用，此损失函数用于重构任务。

    当 alpha = 1.0 时, 该损失等价于 nn.L1Loss。
    当 alpha > 1.0 时, 它会不成比例地惩罚更大的误差，从而迫使模型
    优先学习那些难以重构的部分。
    """

    def __init__(self, alpha=2.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def forward(self, input, target):
        error = torch.abs(input - target)
        weighted_loss = torch.pow(error, self.alpha)
        return weighted_loss.mean()


class MemoryHead(nn.Module):
    """
    一个独立的记忆头，作用于已经提取的latent space上。
    设计用于在冻结的encoder之上进行训练。
    - V2: 增加了一个MLP来转换潜向量，以增加可学习参数和模型的表达能力。
    """

    def __init__(self, latent_dim, mem_dim=512, hidden_dim=256, num_layers=8, initial_memory=None):
        super().__init__()
        # --- MLP for latent space transformation ---
        # 增加一个多层感知机 (MLP) 来转换输入特征 z, 增加模型的复杂度。
        # 这为 Phase 2 训练提供了更多可学习的参数。
        layers = []
        input_d = latent_dim
        # 创建一个包含 `num_layers` 个隐藏层的MLP
        for _ in range(num_layers):
            layers.append(nn.Linear(input_d, hidden_dim))
            # 使用 BatchNorm1d 稳定训练
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_d = hidden_dim

        # 最后一层将特征投影回原始的 latent_dim，以便与 memory bank 进行交互
        layers.append(nn.Linear(input_d, latent_dim))
        self.mlp = nn.Sequential(*layers)

        # --- Memory Matrix (as before) ---
        if initial_memory is not None:
            if initial_memory.shape != (mem_dim, latent_dim):
                raise ValueError(
                    f"Shape of initial_memory {initial_memory.shape} does not match expected shape {(mem_dim, latent_dim)}")
            self.memory = nn.Parameter(initial_memory)
            logging.info("Initialized memory head with provided initial memory (e.g., from K-Means).")
        else:
            self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim))
            nn.init.kaiming_uniform_(self.memory)  # 使用较好的初始化

    def forward(self, z):
        """
        z: latent vector, shape [batch_size, latent_dim]
        """
        # 1. 将潜向量 z 通过 MLP 进行转换
        z_transformed = self.mlp(z)

        # 2. 使用转换后的向量 z_transformed 来查询记忆库
        # 使用归一化的点积（余弦相似度）计算注意力
        # 这比简单的矩阵乘法更稳定
        attention = F.linear(F.normalize(z_transformed, dim=1), F.normalize(self.memory, dim=1))
        # attention_weights = F.softmax(attention, dim=1)
        attention_weights = F.softmax(attention / 1.0, dim=1)

        return attention_weights


class ClassifierCNNAE(ClassifierBase):
    def __init__(self, args, ddp=False):
        """
        Initializes the AutoEncoder-based classifier.

        Args:
            args: Command line arguments, should include `ae_model_type`
                  to select between 'unet' and 'mem-ae'.
            ddp (bool): Flag for distributed data parallel.
        """
        super().__init__()
        self.num_epochs = 1024
        self.lr = 1e-4
        self.ae_model_type = getattr(args, 'ae_model_type', 'ae')
        self.training_phase = getattr(args, 'training_phase', 1)
        self.hard_example_threshold = getattr(args, 'hard_example_threshold', 0.8)
        self.diversity_loss_weight = 0.5
        self.contrastive_loss_weight = getattr(args, 'contrastive_loss_weight', 0.0)
        if self.contrastive_loss_weight > 0 and self.training_phase == 1:
            logging.info(f"Contrastive learning enabled in phase 1 with weight: {self.contrastive_loss_weight}")
        self.features_generator = FeaturesGeneratorCNN()
        self.rank = args.rank
        self.local_rank = args.rank

        make_dirs(args.save_dir, reset=True)

        # Choose the appropriate transformation function based on the model type
        if self.ae_model_type == '2d-cnn-ae-mtf':
            self.transform_fn = self.features_generator.transform_sample_ae_mtf
        else:
            self.transform_fn = self.features_generator.transform_sample_ae

        # --- Phase-dependent model initialization ---
        model = None
        if self.ae_model_type == 'ae':
            model = NetAFDAE().to(self.local_rank)
            self.use_mem_ae = False
        elif self.ae_model_type == 'unet':
            model = NetAFDAE_UNet().to(self.local_rank)
            self.use_mem_ae = False
        elif self.ae_model_type == '2d-cnn-ae-mtf':
            if self.training_phase == 2:
                raise ValueError("Phase 2 training is not implemented for the '2d-cnn-ae-mtf' model.")
            model = NetAFDAE_2D_MTF().to(self.local_rank)
            self.use_mem_ae = False
        elif self.ae_model_type in ['mem-ae', 'unet-mem', 'mem-flow-ae']:
            logging.warning("Phase 1 training is for reconstruction. Forcing a non-memory AE model.")
            # This is a memory model, but memory head is only used in phase 2.
            self.use_mem_ae = self.training_phase == 2
            self.ae_model_type = 'ae' if 'unet' not in self.ae_model_type else 'unet'
            if self.training_phase == 1:
                model = NetAFDAE().to(self.local_rank) if self.ae_model_type == 'ae' else NetAFDAE_UNet().to(
                    self.local_rank)
        else:
            raise ValueError(f"Unsupported AE model type: {self.ae_model_type}")

        if self.training_phase == 1:
            self.model = model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            # 在阶段2，我们加载基础AE，冻结它，并只训练MemoryHead
            if args.path_ckpt is None:
                raise ValueError("Phase 2 training requires a checkpoint from Phase 1 via --path_ckpt.")

            # 1. 初始化模型 (base AE 和 memory head)
            base_model_type = 'ae' if 'unet' not in self.ae_model_type else 'unet'
            self.model = NetAFDAE().to(self.local_rank) if base_model_type == 'ae' else NetAFDAE_UNet().to(
                self.local_rank)
            latent_dim = self.model.get_latent_dim()
            self.memory_head = MemoryHead(latent_dim=latent_dim, mem_dim=512).to(self.local_rank)

            # 2. 智能判断和分配检查点路径
            ckpt_path = args.path_ckpt
            base_ckpt_path, head_ckpt_path = None, None

            if "_base.pt" in ckpt_path:
                base_ckpt_path = ckpt_path
                head_ckpt_path = ckpt_path.replace("_base.pt", "_head.pt")
            elif "_head.pt" in ckpt_path:
                head_ckpt_path = ckpt_path
                base_ckpt_path = ckpt_path.replace("_head.pt", "_base.pt")
            else:  # 认为是第一阶段的检查点
                base_ckpt_path = ckpt_path

            # 3. 加载检查点
            if not os.path.exists(base_ckpt_path):
                raise FileNotFoundError(f"Base model checkpoint not found at a presumed path: {base_ckpt_path}")
            self._load_checkpoint_v0(base_ckpt_path, model=self.model, strict=False)

            latent_dim = self.model.get_latent_dim()
            if head_ckpt_path and os.path.exists(head_ckpt_path):
                self.memory_head = MemoryHead(latent_dim=latent_dim, mem_dim=512).to(self.local_rank)
                self._load_checkpoint_v0(head_ckpt_path, model=self.memory_head, strict=True)
                logging.info(f"Resuming phase 2 with loaded memory head from: {head_ckpt_path}")
            else:
                logging.info("No head checkpoint found. Initializing new memory head.")
                initial_memory_centers = None
                if self.rank == 0:  # Perform clustering only on rank 0
                    logging.info("Attempting K-Means clustering to initialize memory slots...")
                    try:
                        from sklearn.cluster import KMeans
                        train_data_path = os.path.join(args.load_dir, 'train_data.h5')
                        if not os.path.exists(train_data_path):
                            raise FileNotFoundError(f"Training data for K-Means not found: {train_data_path}")

                        temp_dataset = HDF5SequentialSliceDataset(train_data_path,
                                                                  self.transform_fn,
                                                                  seq_len=self.features_generator.seq_len,
                                                                  step=self.features_generator.seq_len,
                                                                  only_normal=True)
                        num_samples = min(20000, len(temp_dataset))
                        indices = np.random.choice(len(temp_dataset), num_samples, replace=False)
                        subset = Subset(temp_dataset, indices)
                        temp_loader = DataLoader(dataset=subset, batch_size=2048, shuffle=False)

                        self.model.eval()
                        all_latents = []
                        with torch.no_grad():
                            for inputs, _ in temp_loader:
                                _, latents, _ = self.model(inputs.to(self.local_rank))
                                all_latents.append(latents.cpu().numpy())

                        if all_latents:
                            all_latents_np = np.concatenate(all_latents, axis=0)
                            n_clusters = 512
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
                            kmeans.fit(all_latents_np)
                            initial_memory_centers = torch.from_numpy(kmeans.cluster_centers_).float()
                            logging.info(f"Memory bank initialized with {n_clusters} K-Means cluster centers.")
                    except ImportError:
                        logging.warning("scikit-learn not found. Skipping K-Means. pip install scikit-learn.")
                    except Exception as e:
                        logging.error(f"K-Means failed: {e}. Falling back to random init.")
                if ddp:
                    obj_list = [initial_memory_centers] if self.rank == 0 else [None]
                    dist.broadcast_object_list(obj_list, src=0)
                    initial_memory_centers = obj_list[0]

            # 4. 冻结基础AE并配置优化器
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            logging.info("Froze base AE model parameters.")
            self.optimizer = optim.Adam(self.memory_head.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=0.)

        if self.local_rank == 0:
            logging.info(f"Initialized classifier for Training Phase: {self.training_phase}")

        # 在阶段1加载检查点是可选的，但在阶段2是强制的（已在上面处理）
        if self.training_phase == 1 and args.path_ckpt is not None:
            self._load_checkpoint_v0(args.path_ckpt)

        self.ddp = ddp
        if self.ddp:
            if self.training_phase == 1:
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:  # Phase 2
                # Only wrap the trainable part (memory_head) in DDP
                # The base model is frozen and does not need to be wrapped
                self.memory_head = DDP(self.memory_head, device_ids=[self.local_rank], output_device=self.local_rank)

        # self.criterion = nn.MSELoss().to(self.local_rank)  # Reconstruction loss
        self.criterion = nn.L1Loss().to(self.local_rank)
        # self.criterion = WeightedReconstructionLoss(alpha=8.0).to(self.local_rank)
        self.save_dir = args.save_dir  # type: ignore

    def _load_checkpoint_v0(self, checkpoint_path, model=None, strict=True):
        target_model = self.model if model is None else model
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{self.local_rank}')
        # Handle DDP-saved models
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            state_dict = new_state_dict
        # strict=False allows loading only encode part if a classifier model is provided
        target_model.load_state_dict(state_dict, strict=strict)
        logging.info(
            f'Loaded checkpoint into {target_model.__class__.__name__} from {checkpoint_path} (strict={strict})')

    def _entropy_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算注意力权重的熵损失，以鼓励稀疏性。
        熵越小，表示注意力分布越集中（越稀疏）。
        L_sparsity = sum(entropy(attention))
        """
        # 添加一个小的 epsilon 防止 log(0)
        epsilon = 1e-12
        # H(p) = - sum(p * log(p))
        entropy = -attention_weights * torch.log(attention_weights + epsilon)
        # 在所有记忆单元维度上求和，然后在批次维度上求平均
        return torch.mean(torch.sum(entropy, dim=1))

    def train(self, data):
        if self.training_phase == 1:
            self._train_phase1(data)
        elif self.training_phase == 2:
            self._train_phase2(data)
        else:
            raise ValueError(f"Invalid training phase: {self.training_phase}")

    def _train_phase2(self, data):
        """
        第二阶段训练：冻结AE，仅训练MemoryHead，使用难例挖掘。
        """
        if self.save_dir is not None and not os.path.exists(self.save_dir) and self.rank == 0:
            os.makedirs(self.save_dir)

        # --- 1. Hard Example Mining (on rank 0 only to avoid file contention and redundant work) ---
        hard_examples_tensor = None
        if self.rank == 0:
            logging.info(f"Starting Phase 2: Hard Example Mining with threshold {self.hard_example_threshold}...")

            # Use the full, non-random dataset to calculate errors for all normal samples
            full_dataset = HDF5SequentialSliceDataset(
                data['train_path'],
                self.transform_fn,
                seq_len=self.features_generator.seq_len,
                step=self.features_generator.seq_len,  # no overlap
                only_normal=True
            )
            full_loader = DataLoader(dataset=full_dataset, batch_size=2048, shuffle=False)

            all_normal_inputs, all_recon_errors = [], []
            self.model.eval()  # Ensure AE is in eval mode
            with torch.no_grad():
                # Since the dataset is now pre-filtered to contain only normal samples,
                # we can simplify the loop and directly use the inputs.
                for inputs, _ in full_loader:
                    inputs_normal = inputs.to(self.local_rank)
                    recons, _, _ = self.model(inputs_normal)
                    # Generic reconstruction error calculation for 1D and 2D
                    errors = torch.mean((inputs_normal - recons) ** 2, dim=tuple(range(1, inputs_normal.dim()))).cpu()

                    all_normal_inputs.append(inputs_normal.cpu())
                    all_recon_errors.append(errors)

            if not all_recon_errors:
                logging.error("No normal samples found for hard example mining.")
                # Create a sentinel empty tensor
                hard_examples_tensor = torch.empty(0)
            else:
                all_normal_inputs = torch.cat(all_normal_inputs, dim=0)
                all_recon_errors_np = torch.cat(all_recon_errors, dim=0).numpy()

                error_threshold = np.percentile(all_recon_errors_np, self.hard_example_threshold * 100)
                hard_indices = np.where(all_recon_errors_np >= error_threshold)[0]
                hard_examples_tensor = all_normal_inputs[hard_indices]

                logging.info(f"Found {len(all_recon_errors_np)} normal samples. Recon error stats: "
                             f"min={np.min(all_recon_errors_np):.6f}, max={np.max(all_recon_errors_np):.6f}, "
                             f"mean={np.mean(all_recon_errors_np):.6f}.")
                logging.info(
                    f"Error threshold at {self.hard_example_threshold * 100}th percentile is {error_threshold:.6f}. "
                    f"Found {len(hard_examples_tensor)} hard examples for training.")

        if self.ddp:
            # Broadcast the list containing the tensor from rank 0 to all other processes
            obj_list = [hard_examples_tensor] if self.rank == 0 else [None]
            dist.broadcast_object_list(obj_list, src=0)
            hard_examples_tensor = obj_list[0]

        if hard_examples_tensor is None or len(hard_examples_tensor) == 0:
            if self.rank == 0:
                logging.warning("No hard examples found above the threshold. Phase 2 training cannot proceed.")
            return

        # --- 2. Train MemoryHead ---
        hard_dataset = TensorDataset(hard_examples_tensor)
        is_distributed = self.ddp
        train_sampler = torch.utils.data.distributed.DistributedSampler(hard_dataset) if is_distributed else None
        loader = DataLoader(dataset=hard_dataset, batch_size=1024, shuffle=not is_distributed, sampler=train_sampler)

        best_loss = float('inf')
        self.memory_head.train()

        for epoch in range(self.num_epochs):  # Phase 2 usually requires fewer epochs
            epoch_start_time = time.time()
            if is_distributed:
                train_sampler.set_epoch(epoch)

            epoch_total_loss, epoch_entropy_loss, epoch_diversity_loss = 0.0, 0.0, 0.0
            num_batches = 0
            for (inputs,) in loader:  # TensorDataset returns a tuple
                inputs = inputs.to(self.local_rank)

                with torch.no_grad():
                    # The encode method returns a tuple, extract the latent tensor (second element)
                    _, latents, _ = self.model(inputs)

                attention_weights = self.memory_head(latents)

                entropy_loss = self._entropy_loss(attention_weights)
                diversity_loss = self._diversity_loss()
                total_loss = entropy_loss + self.diversity_loss_weight * diversity_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_diversity_loss += diversity_loss.item()
                num_batches += 1

            self.scheduler.step()

            avg_epoch_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
            avg_epoch_entropy_loss = epoch_entropy_loss / num_batches if num_batches > 0 else 0
            avg_epoch_diversity_loss = epoch_diversity_loss / num_batches if num_batches > 0 else 0
            epoch_duration = time.time() - epoch_start_time

            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(
                    f'Phase 2 - Epoch [{epoch + 1}/{self.num_epochs}], '
                    f'LR: {current_lr:.2e}, Time: {epoch_duration:.2f}s, '
                    f'Loss: {avg_epoch_total_loss:.8f} (E: {avg_epoch_entropy_loss:.8f}, D: {avg_epoch_diversity_loss:.8f})'
                )

                if avg_epoch_total_loss < best_loss:
                    best_loss = avg_epoch_total_loss
                    # In phase 2, we save both the memory_head and the base model
                    model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
                    head_to_save = self.memory_head.module if is_distributed else self.memory_head

                    # base_path = os.path.join(self.save_dir, f'phase2_best_e{epoch}_loss{best_loss:.4f}_base.pt')
                    # head_path = os.path.join(self.save_dir, f'phase2_best_e{epoch}_loss{best_loss:.4f}_head.pt')
                    base_path = os.path.join(self.save_dir, f'phase2_best_base.pt')
                    head_path = os.path.join(self.save_dir, f'phase2_best_head.pt')

                    torch.save(model_to_save.state_dict(), base_path)
                    torch.save(head_to_save.state_dict(), head_path)

                    logging.info(f'Saved new best Phase 2 models with loss: {best_loss:.4f} to {head_path}')

    def _diversity_loss(self) -> torch.Tensor:
        """
        Calculates a diversity loss to encourage memory slots to be dissimilar.
        This is done by penalizing the cosine similarity between memory vectors.
        """
        if not hasattr(self, 'memory_head'):
            return torch.tensor(0.0).to(self.local_rank)

        # Get the memory bank from the (potentially DDP-wrapped) memory_head
        memory_bank = self.memory_head.module.memory if self.ddp and isinstance(self.memory_head,
                                                                                DDP) else self.memory_head.memory

        # Normalize memory vectors to unit length
        mem_normalized = F.normalize(memory_bank, p=2, dim=1)

        # Calculate the cosine similarity matrix (M * M^T)
        cos_sim_matrix = torch.matmul(mem_normalized, mem_normalized.t())

        # We want to minimize off-diagonal similarities. Subtract the identity matrix.
        identity = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device)
        return torch.mean((cos_sim_matrix - identity) ** 2)

    def _train_phase1(self, data, loss_ckp=True):
        if self.save_dir is not None and not os.path.exists(self.save_dir) and self.rank == 0:
            os.makedirs(self.save_dir)

        dataset = HDF5SPDataset(
            data['train_path'],
            self.transform_fn,
            seq_len=self.features_generator.seq_len,
            min_delta=MIN_VAL_TH)
        # # 使用新的序贯滑窗数据集替换旧的随机采样数据集
        # # 步长 step 设为 seq_len // 4 提供了75%的重叠，是一种有效的数据增强
        # dataset = HDF5SequentialSliceDataset(
        #     hdf5_file_path=data['train_path'],
        #     transform=self.features_generator.transform_sample_ae,
        #     seq_len=self.features_generator.seq_len,
        #     step=self.features_generator.seq_len // 8
        # )

        is_distributed = isinstance(self.model, DDP)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
        loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=not is_distributed, sampler=train_sampler)

        best_accuracy = 0.0
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            if is_distributed:
                train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_recon_loss, epoch_sparsity_loss, epoch_flow_loss, epoch_contrastive_loss = 0.0, 0.0, 0.0, 0.0
            num_batches = 0
            for inputs, labels in loader:  # Labels are used to filter for normal data
                # Filter for normal data (label == 0)
                # --- FIX: Define inputs_normal at the beginning of the loop ---
                normal_indices = (labels.view(-1) == 0)
                if not torch.any(normal_indices):
                    continue

                inputs_normal = inputs[normal_indices].to(self.local_rank)

                # Unpack model outputs
                # Base reconstruction loss is always calculated
                model_outputs = self.model(inputs_normal)
                reconstructions, _, aux_output = model_outputs[:3]
                recon_loss = self.criterion(reconstructions, inputs_normal)
                loss = recon_loss

                # Sparsity loss (for memory models)
                sparsity_loss = torch.tensor(0.0).to(self.local_rank)
                if self.use_mem_ae and aux_output is not None:
                    sparsity_loss = self._entropy_loss(aux_output)
                    loss += self.sparsity_weight * sparsity_loss

                # Flow loss (for flow-based models)
                flow_loss = torch.tensor(0.0).to(self.local_rank)
                if self.ae_model_type == 'mem-flow-ae':
                    log_prob = model_outputs[3]
                    flow_loss = -log_prob.mean()  # Minimize negative log-likelihood
                    loss += self.flow_loss_weight * flow_loss

                # --- Add Contrastive Loss ---
                if self.contrastive_loss_weight > 0:
                    # 1. Generate augmented samples (phase-shifted)
                    inputs_aug = generate_augmented_samples(inputs_normal)

                    # 2. Get embeddings for original and augmented samples
                    model_to_encode = self.model.module if isinstance(self.model, DDP) else self.model
                    # Handle tuple output from encode method (e.g., NetAFDAE_UNet)
                    latents_orig = model_to_encode.encode(inputs_normal)
                    if isinstance(latents_orig, tuple): latents_orig = latents_orig[0]
                    latents_aug = model_to_encode.encode(inputs_aug)
                    if isinstance(latents_aug, tuple): latents_aug = latents_aug[0]

                    # 3. Calculate cosine similarity loss and add to total loss
                    contrastive_loss = 1 - F.cosine_similarity(latents_orig, latents_aug, dim=1).mean()
                    loss += self.contrastive_loss_weight * contrastive_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                if self.use_mem_ae:
                    epoch_sparsity_loss += sparsity_loss.item()
                if self.ae_model_type == 'mem-flow-ae':
                    epoch_flow_loss += flow_loss.item()
                if self.contrastive_loss_weight > 0:
                    epoch_contrastive_loss += contrastive_loss.item()
                num_batches += 1

            self.scheduler.step()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0
            avg_sparsity_loss = epoch_sparsity_loss / num_batches if num_batches > 0 else 0
            avg_flow_loss = epoch_flow_loss / num_batches if num_batches > 0 else 0
            avg_contrastive_loss = epoch_contrastive_loss / num_batches if num_batches > 0 else 0

            val_f1_score = self.evaluate(data['test_path']) if not loss_ckp else 1 - avg_epoch_loss
            epoch_duration = time.time() - epoch_start_time

            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (f'Phase 1 - Epoch [{epoch + 1}/{self.num_epochs}]: LR: {current_lr:.2e}, Time: {epoch_duration:.2f}s, Val F1: {val_f1_score:.4f}, '
                           f'Loss: {avg_epoch_loss:.6f} (Recon: {avg_recon_loss:.6f}, Contrastive: {avg_contrastive_loss:.6f})')
                logging.info(log_msg)

                if val_f1_score > best_accuracy:
                    best_accuracy = val_f1_score
                    model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
                    # _path_save = os.path.join(self.save_dir, f'ae_best_e{epoch}_acc{best_accuracy:.4f}.pt')
                    _path_save = os.path.join(self.save_dir, f'ae_best.pt')
                    torch.save(model_to_save.state_dict(), _path_save)
                    logging.info(
                        f'Saved new best AE model with validation accuracy: {best_accuracy:.4f} to {_path_save}')

    def infer(self, x, batch_size=16):
        """
        Performs inference using the AutoEncoder model.

        Args:
            x: Input data, can be a numpy array, list, or any format supported by InferenceDataset.
            batch_size (int): The batch size for inference.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - A 1D numpy array with the reconstruction error (MSE) for each sample.
                - A 1D numpy array with the attention entropy score for each sample.
                  (Returns zeros if not in Phase 2).
                - A 2D numpy array with the latent vector for each sample.
        """
        # 确保模型处于评估模式
        if self.training_phase == 2:
            self.memory_head.eval()

        model_to_infer = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_infer.eval()

        # Use the same 'ae' transform as in training/evaluation
        dataset = InferenceDataset(
            x,
            transform=self.transform_fn,
            seq_len=self.features_generator.seq_len
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_recon_errors = []
        all_entropy_scores = []
        all_latents = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.local_rank)

                # 1. 统一获取重构结果和潜向量
                reconstructions, latents, *_ = model_to_infer(batch_x)

                # 2. 计算重构误差 (Reconstruction Error)，始终计算
                recon_error = torch.mean((batch_x - reconstructions) ** 2, dim=tuple(range(1, batch_x.dim())))

                # 3. 计算注意力熵 (Entropy Score)，只在阶段2计算
                if self.training_phase == 2:
                    attention_weights = self.memory_head(latents)
                    epsilon = 1e-12
                    entropy = -attention_weights * torch.log(attention_weights + epsilon)
                    entropy_score = torch.sum(entropy, dim=1)
                else:
                    # 阶段1没有 memory_head，熵分数为0
                    entropy_score = torch.zeros_like(recon_error)

                all_recon_errors.append(recon_error.cpu().numpy())
                all_entropy_scores.append(entropy_score.cpu().numpy())
                all_latents.append(latents.cpu().numpy())

        # 推理结束后，恢复模式
        model_to_infer.train()
        if self.training_phase == 2:
            self.memory_head.train()

        return np.concatenate(all_recon_errors), np.concatenate(all_latents), np.concatenate(all_entropy_scores)

    def evaluate(self, test_path, batch_size=1024, threshold=None, plot_positive_index=None, plot_negative_index=None):
        """
        在测试集上评估自编码器模型，并计算详细的准确率指标。

        工作流程:
        1. 对测试集中的每个样本计算其均方重构误差（MSE）。
        2. 如果提供了 plot_positive_index 或 plot_negative_index, 则找到指定次序的样本并将其原始信号、
           重构信号与叠加图保存为图像。
        3. 使用真实标签和重构误差，通过ROC曲线分析找到一个最佳阈值（除非手动指定），
           该阈值旨在最大化Youden指数 J = TPR - FPR (真阳性率 - 假阳性率)。
        4. 如果样本的重构误差高于此阈值，则将其分类为异常（1），否则为正常（0）。
        5. 基于这些预测计算F1分数、正例准确率（召回率）和负例准确率（特异性）。
        6. 打印详细的统计信息并返回总体准确率。
        """
        # 使用新的序贯切片数据集进行评估，确保覆盖所有数据
        dataset = HDF5SequentialSliceDataset(
            hdf5_file_path=test_path,
            transform=self.transform_fn,
            seq_len=self.features_generator.seq_len,
            step=self.features_generator.seq_len  # step=seq_len 表示无重叠切片
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.training_phase == 2 and hasattr(self, 'memory_head'):
            self.memory_head.eval()

        model_to_eval = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_eval.eval()

        # --- 新增：绘图目标设置 ---
        plot_target_label = None
        plot_target_ordinal = None
        if plot_positive_index is not None:
            plot_target_label = 1
            plot_target_ordinal = plot_positive_index - 1  # 用户从1开始计数，代码从0开始
        elif plot_negative_index is not None:
            plot_target_label = 0
            plot_target_ordinal = plot_negative_index - 1

        if plot_target_ordinal is not None and plot_target_ordinal < 0:
            logging.error("Plot index must be a positive integer (>= 1).")
            plot_target_label = None  # Invalidate plotting

        all_scores, all_labels = [], []
        plotted = False
        sample_type_counter = 0  # 计数找到的目标类型样本数量

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)

                if self.training_phase == 1:
                    reconstructions, *_ = model_to_eval(inputs)
                    # Generic reconstruction error calculation
                    scores = torch.mean((inputs - reconstructions) ** 2, dim=tuple(range(1, inputs.dim()))).cpu().numpy()
                else:  # Phase 2
                    latents = model_to_eval.encode(inputs)
                    attention_weights = self.memory_head(latents)
                    epsilon = 1e-12
                    entropy = -attention_weights * torch.log(attention_weights + epsilon)
                    scores = torch.sum(entropy, dim=1).cpu().numpy()
                    reconstructions, *_ = model_to_eval(inputs)  # For plotting only

                # --- 新增：查找并绘制指定次序的样本 ---
                _save_dir = self.save_dir
                if plot_target_label is not None and not plotted and self.rank == 0:
                    # 找到当前批次中所有目标标签的索引
                    target_indices_in_batch = (labels.view(-1) == plot_target_label).nonzero(as_tuple=True)[0]
                    num_targets_in_batch = len(target_indices_in_batch)

                    if sample_type_counter <= plot_target_ordinal < sample_type_counter + num_targets_in_batch:
                        try:
                            import matplotlib.pyplot as plt
                            # 计算目标样本在当前批次的目标样本列表中的位置
                            ordinal_in_batch = plot_target_ordinal - sample_type_counter
                            # 获取其在完整批次中的索引
                            idx_in_batch = target_indices_in_batch[ordinal_in_batch]

                            if self.ae_model_type == '2d-cnn-ae-mtf':
                                original_image = inputs[idx_in_batch].cpu().numpy().squeeze()
                                reconstructed_image = reconstructions[idx_in_batch].cpu().numpy().squeeze()
                                score_for_sample = scores[idx_in_batch]

                                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                                title = (f'MTF Reconstruction of {plot_target_ordinal + 1}-th {"Positive" if plot_target_label == 1 else "Negative"} Sample\n'
                                         f'Score: {score_for_sample:.6f}')
                                fig.suptitle(title, fontsize=16)

                                im1 = axs[0].imshow(original_image, cmap='rainbow', origin='lower'); axs[0].set_title('Original MTF'); fig.colorbar(im1, ax=axs[0])
                                im2 = axs[1].imshow(reconstructed_image, cmap='rainbow', origin='lower'); axs[1].set_title('Reconstructed MTF'); fig.colorbar(im2, ax=axs[1])
                                diff_image = np.abs(original_image - reconstructed_image)
                                im3 = axs[2].imshow(diff_image, cmap='hot', origin='lower'); axs[2].set_title('Absolute Difference'); fig.colorbar(im3, ax=axs[2])
                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                            else: # Original 1D signal plotting
                                original_signal = inputs[idx_in_batch].cpu().numpy().flatten()
                                reconstructed_signal = reconstructions[idx_in_batch].cpu().numpy().flatten()
                                score_for_sample = scores[idx_in_batch]

                                fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
                                title = (f'Reconstruction of {plot_target_ordinal + 1}-th {"Positive" if plot_target_label == 1 else "Negative"} Sample\n'
                                         f'MSE: {score_for_sample:.6f}') if self.training_phase == 1 else (
                                    f'Reconstruction with Entropy Score: {score_for_sample:.6f}')
                                fig.suptitle(title, fontsize=16)

                                axs[0].plot(original_signal, color='blue', label='Original'); axs[0].set_title('Original Signal'); axs[0].legend(loc='upper right'); axs[0].grid(True, linestyle='--', alpha=0.6)
                                axs[1].plot(reconstructed_signal, color='orange', label='Reconstructed'); axs[1].set_title('Reconstructed Signal'); axs[1].legend(loc='upper right'); axs[1].grid(True, linestyle='--', alpha=0.6)
                                axs[2].plot(original_signal, label='Original', color='blue', alpha=0.9); axs[2].plot(reconstructed_signal, label='Reconstructed', color='red', linestyle='--', alpha=0.8); axs[2].set_title('Overlay'); axs[2].set_xlabel('Time Step'); axs[2].legend(loc='upper right'); axs[2].grid(True, linestyle='--', alpha=0.6)
                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                            if _save_dir:
                                plot_filename = f'reconstruction_{"positive" if plot_target_label == 1 else "negative"}_sample_{plot_target_ordinal + 1}.png'
                                plot_path = os.path.join(_save_dir, plot_filename)
                                plt.savefig(plot_path)
                                logging.info(f"Saved reconstruction plot to '{plot_path}'")
                            else:
                                logging.warning("Save directory (--save_dir) not specified, cannot save reconstruction plot.")

                            plt.close(fig) # Always close figure to free memory
                            plotted = True
                        except ImportError:
                            logging.warning(
                                "Matplotlib not found, skipping plot. Install with 'pip install matplotlib'.")
                            plotted = True  # 避免重复警告
                    sample_type_counter += num_targets_in_batch

                # 收集所有样本的误差和标签用于最终评估
                all_scores.extend(scores)
                all_labels.extend(labels.cpu().numpy().flatten())

        # --- 新增：检查是否成功绘图 ---
        if plot_target_label is not None and not plotted and self.rank == 0:
            logging.warning(
                f"Could not find the {plot_target_ordinal + 1}-th {'positive' if plot_target_label == 1 else 'negative'} sample. "
                f"The dataset may contain fewer than this number of samples of that type.")

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 处理测试集只包含一个类别的边缘情况
        if len(np.unique(all_labels)) < 2:
            model_to_eval.train()
            logging.warning(
                f"Evaluation set contains only one class ({np.unique(all_labels)}), cannot compute a meaningful ROC curve. "
                "Returning 0.0 accuracy."
            )
            return 0.0

        if threshold is None:
            # 自动找到区分正常和异常样本的最佳阈值
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_idx]
        else:
            best_threshold = threshold
            logging.info(f"Using manually specified threshold: {best_threshold}")

        # 基于最佳阈值进行预测
        predictions = (all_scores >= best_threshold).astype(int)
        # 改为使用 F1-Score 作为主要的评估指标
        f1 = f1_score(all_labels, predictions)

        # ==================== 计算并打印详细指标 ====================
        num_positives = np.sum(all_labels == 1)
        num_negatives = np.sum(all_labels == 0)

        # 计算每个类别被正确预测的数量
        correct_positives = np.sum((predictions == 1) & (all_labels == 1))
        correct_negatives = np.sum((predictions == 0) & (all_labels == 0))

        # 计算每个类别的准确率
        acc_positives = correct_positives / num_positives if num_positives > 0 else 0.0  # 也称为召回率 (Recall) 或 TPR
        acc_negatives = correct_negatives / num_negatives if num_negatives > 0 else 0.0  # 也称为特异性 (Specificity) 或 TNR

        # 只在主进程上打印日志，避免分布式训练时重复打印
        if self.rank == 0:
            logging.info(
                f"  [Eval Stats] Negatives(0): {num_negatives} samples, Acc: {acc_negatives:.4f} | "
                f"Positives(1): {num_positives} samples, Acc: {acc_positives:.4f} | "
                f"Threshold: {best_threshold:.6f} | "
                f"F1 Score: {f1:.6f}"
            )
        # =================================================================

        # 在退出前确保模型切换回训练模式
        model_to_eval.train()
        if self.training_phase == 2 and hasattr(self, 'memory_head'):
            self.memory_head.train()

        return f1


class ClassifierONNX:
    """
    仅做推理/验证，不包含训练逻辑
    """

    def __init__(self, onnx_path: str, use_gpu: bool = False):
        """
        onnx_path : 保存的 .onnx 文件
        use_gpu   : True 则优先使用 CUDAExecutionProvider
        """
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name  # 取第一个输入名
        self.features_generator = FeaturesGeneratorCNN()

        # 下面两行只是为了保持接口一致，外部代码如果使用到可以继续调用
        self.seq_len = self.features_generator.seq_len
        self.onnx_path = onnx_path

        logging.info(f'ONNX model loaded: {onnx_path}, providers={self.session.get_providers()}')

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def infer(self, x, batch_size: int = 16):
        """
        x 可以是 numpy array、list，或任何 InferenceDataset 支持的形式
        返回: probs(np.ndarray), feats(np.ndarray or None)
        """
        dataset = InferenceDataset(
            x,
            transform=self.features_generator.transform_sample,
            seq_len=self.features_generator.seq_len
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        probs_out, feats_out = [], []
        has_feature_output = len(self.session.get_outputs()) > 1  # 按照导出时是否包含特征判断

        for batch_x in loader:
            batch_x = batch_x.numpy().astype(np.float32)  # ORT 只接受 numpy
            ort_outs = self.session.run(None, {self.input_name: batch_x})

            logits = ort_outs[0]
            probs = self._sigmoid(logits).flatten()
            probs_out.extend(probs)

            if has_feature_output:
                feats_out.extend(ort_outs[1])

        probs_out = np.asarray(probs_out, dtype=np.float32)
        feats_out = np.asarray(feats_out, dtype=np.float32) if has_feature_output else None
        return probs_out, feats_out

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def evaluate(self, test_path: str, batch_size: int = 1, n_total_samples: int = -4096):
        """
        test_path : HDF5Dataset 的路径
        """
        dataset = HDF5Dataset(test_path, self.features_generator.transform_sample)
        loader = DataLoader(dataset, batch_size=batch_size)

        all_probs, all_labels = [], []
        _cnt = 0
        for inputs, labels in loader:
            if _cnt > n_total_samples > 0:
                break
            inputs = inputs.numpy().astype(np.float32)
            ort_outs = self.session.run(None, {self.input_name: inputs})
            probs = self._sigmoid(ort_outs[0]).flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            _cnt += batch_size

        preds = [1 if p > 0.5 else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds)
        return f1
