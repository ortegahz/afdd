# FILE: classifier.py

import logging
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import f1_score, roc_curve
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_qat_fx, convert_fx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close  # PyTorch ≥1.12 推荐
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from cores.features_generator import (FeaturesGeneratorXGB, FeaturesGeneratorCNN, InferenceDataset,
                                      HDF5Dataset, HDF5SequentialSliceDataset, HDF5SPDataset)
from cores.loss import HardExampleMiningFocalLoss, F
from cores.nets import NetAFD, NetAFDAE_UNet, NetAFDAE_Mem, NetAFDAE_UNet_Mem, NetAFDAE_Mem_Flow
from utils.macros import MIN_VAL_TH


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
        self.local_rank = args.rank
        self.num_epochs = 8192
        self.lr = 1e-3
        self.ae_model_type = getattr(args, 'ae_model_type', 'mem-flow-ae')
        model = None
        if self.ae_model_type == 'unet':
            model = NetAFDAE_UNet().to(self.local_rank)
            self.use_mem_ae = False
        elif self.ae_model_type == 'mem-ae':
            model = NetAFDAE_Mem(latent_dim=128, mem_dim=2048).to(self.local_rank)
            self.use_mem_ae = True
            self.sparsity_weight = 1e-4
        elif self.ae_model_type == 'unet-mem':
            model = NetAFDAE_UNet_Mem(latent_dim=128, mem_dim=2048).to(self.local_rank)
            self.use_mem_ae = True
            self.sparsity_weight = 1e-4
        elif self.ae_model_type == 'mem-flow-ae':
            model = NetAFDAE_Mem_Flow(latent_dim=128, mem_dim=2048).to(self.local_rank)
            self.use_mem_ae = True  # It's also a memory AE
            self.sparsity_weight = 1e-3
            self.flow_loss_weight = 1e-4  # 1e-4  # Weight for the flow model's NLL loss
        else:
            raise ValueError(f"Unsupported AE model type: {self.ae_model_type}")

        self.model = model
        # For flow models, it might be better to use separate optimizers, but one is fine for a start.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-7)

        if self.local_rank == 0:
            logging.info(f"Initialized AE model of type: '{self.ae_model_type}'")
            if self.use_mem_ae:
                logging.info(f"Sparsity loss weight for MemAE: {self.sparsity_weight}")
            if 'flow' in self.ae_model_type:
                logging.info(f"Flow loss weight: {self.flow_loss_weight}")

        if args.path_ckpt is not None:
            self._load_checkpoint_v0(args.path_ckpt)

        if ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # self.criterion = nn.MSELoss().to(self.local_rank)  # Reconstruction loss
        self.criterion = nn.L1Loss().to(self.local_rank)
        # self.criterion = WeightedReconstructionLoss(alpha=8.0).to(self.local_rank)
        self.features_generator = FeaturesGeneratorCNN()
        self.rank = args.rank
        self.save_dir = args.save_dir

    def _load_checkpoint_v0(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{self.local_rank}')
        # Handle DDP-saved models
        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        # strict=False allows loading only encoder part if a classifier model is provided
        self.model.load_state_dict(state_dict, strict=False)
        logging.info(f'Loaded checkpoint from {checkpoint_path}')

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

    def train(self, data, loss_ckp=True):
        if self.save_dir is not None and not os.path.exists(self.save_dir) and self.rank == 0:
            os.makedirs(self.save_dir)

        # dataset = HDF5Dataset(data['train_path'], self.features_generator.transform_sample_ae)
        dataset = HDF5SPDataset(
            data['train_path'],
            self.features_generator.transform_sample_ae,
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
            epoch_recon_loss, epoch_sparsity_loss, epoch_flow_loss = 0.0, 0.0, 0.0
            num_batches = 0
            for inputs, labels in loader:  # Labels are used to filter for normal data
                # Filter for normal data (label == 0)
                normal_indices = (labels.view(-1) == 0)
                if not torch.any(normal_indices):
                    continue

                inputs_normal = inputs[normal_indices].to(self.local_rank)

                # Unpack model outputs
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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                if self.use_mem_ae:
                    epoch_sparsity_loss += sparsity_loss.item()
                if self.ae_model_type == 'mem-flow-ae':
                    epoch_flow_loss += flow_loss.item()
                num_batches += 1

            self.scheduler.step()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0
            avg_sparsity_loss = epoch_sparsity_loss / num_batches if num_batches > 0 else 0
            avg_flow_loss = epoch_flow_loss / num_batches if num_batches > 0 else 0

            val_accuracy = self.evaluate(data['test_path']) if not loss_ckp else 1 - avg_epoch_loss
            epoch_duration = time.time() - epoch_start_time

            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (
                    f'Epoch [{epoch + 1}/{self.num_epochs}], '
                    f'LR: {current_lr:.2e}, '
                    f'Time: {epoch_duration:.2f}s, '
                    f'Validation Acc: {val_accuracy:.4f}'
                )
                if self.ae_model_type == 'mem-flow-ae':
                    # Correctly calculate weighted losses for logging
                    weighted_recon_loss = avg_recon_loss  # Assuming weight is 1.0
                    weighted_sparsity_loss = avg_sparsity_loss * self.sparsity_weight
                    weighted_flow_loss = avg_flow_loss * self.flow_loss_weight
                    # For verification, their sum should be close to avg_epoch_loss
                    calculated_total_loss = weighted_recon_loss + weighted_sparsity_loss + weighted_flow_loss

                    log_msg += (
                        f' | Total Loss: {avg_epoch_loss:.8f} (Calc: {calculated_total_loss:.8f})\n'
                        f'        Components (Raw)      -> Recon: {avg_recon_loss:.8f}, Sparsity: {avg_sparsity_loss:.8f}, Flow: {avg_flow_loss:.8f}\n'
                        f'        Components (Weighted) -> Recon: {weighted_recon_loss:.8f}, Sparsity: {weighted_sparsity_loss:.8f}, Flow: {weighted_flow_loss:.8f}'
                    )
                elif self.use_mem_ae:
                    log_msg += (
                        f' | Total Loss: {avg_epoch_loss:.8f} '
                        f'(Recon: {avg_recon_loss:.8f} + Sparsity: {avg_sparsity_loss:.8f})'
                    )
                else:
                    log_msg += (
                        f' | Train Loss: {avg_epoch_loss:.8f}'
                    )
                logging.info(log_msg)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
                    _path_save = os.path.join(self.save_dir, f'ae_best_e{epoch}_acc{best_accuracy:.4f}.pt')
                    torch.save(model_to_save.state_dict(), _path_save)
                    logging.info(
                        f'Saved new best AE model with validation accuracy: {best_accuracy:.4f} to {_path_save}')

    def infer(self, x, batch_size=16):
        """
        Performs inference using the AutoEncoder model and returns the reconstruction error and latent vector for each sample.

        Args:
            x: Input data, can be a numpy array, list, or any format supported by InferenceDataset.
            batch_size (int): The batch size for inference.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - A 1D numpy array with the reconstruction score (MSE) for each sample.
                - A 2D numpy array with the latent vector for each sample.
        """
        # 确保模型处于评估模式
        model_to_infer = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_infer.eval()

        # Use the same 'ae' transform as in training/evaluation
        dataset = InferenceDataset(
            x,
            transform=self.features_generator.transform_sample_ae,
            seq_len=self.features_generator.seq_len
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_errors = []
        all_latents = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.local_rank)
                reconstructions, latents, *_ = model_to_infer(batch_x)

                # Calculate mean squared error for each sample in the batch.
                # Shape of batch_x & reconstructions: [batch, 1, seq_len]
                # We average over dims 1 and 2 to get a single scalar error score per sample.
                errors = torch.mean((batch_x - reconstructions) ** 2, dim=(1, 2))
                all_errors.append(errors.cpu().numpy())
                all_latents.append(latents.cpu().numpy())

        # 推理结束后，将模型恢复到训练模式
        model_to_infer.train()

        return np.concatenate(all_errors), np.concatenate(all_latents)

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
            transform=self.features_generator.transform_sample_ae,
            seq_len=self.features_generator.seq_len,
            step=self.features_generator.seq_len  # step=seq_len 表示无重叠切片
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

        all_errors, all_labels = [], []
        plotted = False
        sample_type_counter = 0  # 计数找到的目标类型样本数量

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)
                reconstructions, *_ = model_to_eval(inputs)

                errors = torch.mean((inputs - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()

                # --- 新增：查找并绘制指定次序的样本 ---
                _save_dir = "/home/manu/tmp"
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

                            original_signal = inputs[idx_in_batch].cpu().numpy().flatten()
                            reconstructed_signal = reconstructions[idx_in_batch].cpu().numpy().flatten()
                            error_for_sample = errors[idx_in_batch]
                            pointwise_squared_error = (original_signal - reconstructed_signal) ** 2
                            max_pointwise_error = np.mean(pointwise_squared_error)

                            fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
                            title = (
                                f'Reconstruction of {plot_target_ordinal + 1}-th {"Positive" if plot_target_label == 1 else "Negative"} Sample\n'
                                f'MSE: {error_for_sample:.6f} | Mean Point-wise Sq. Error: {max_pointwise_error:.6f}'
                            )
                            fig.suptitle(title, fontsize=16)

                            axs[0].plot(original_signal, color='blue', label='Original')
                            axs[0].set_title('Original Signal');
                            axs[0].legend(loc='upper right');
                            axs[0].grid(True, linestyle='--', alpha=0.6)

                            axs[1].plot(reconstructed_signal, color='orange', label='Reconstructed')
                            axs[1].set_title('Reconstructed Signal');
                            axs[1].legend(loc='upper right');
                            axs[1].grid(True, linestyle='--', alpha=0.6)

                            axs[2].plot(original_signal, label='Original', color='blue', alpha=0.9)
                            axs[2].plot(reconstructed_signal, label='Reconstructed', color='red', linestyle='--',
                                        alpha=0.8)
                            axs[2].set_title('Overlay');
                            axs[2].set_xlabel('Time Step');
                            axs[2].legend(loc='upper right');
                            axs[2].grid(True, linestyle='--', alpha=0.6)

                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plot_filename = f'reconstruction_{"positive" if plot_target_label == 1 else "negative"}_sample_{plot_target_ordinal + 1}.png'
                            plot_filename = os.path.join(_save_dir, plot_filename)
                            plt.savefig(plot_filename)
                            logging.info(f"Saved reconstruction plot to '{plot_filename}'")
                            plt.close(fig)
                            plotted = True
                        except ImportError:
                            logging.warning(
                                "Matplotlib not found, skipping plot. Install with 'pip install matplotlib'.")
                            plotted = True  # 避免重复警告
                    sample_type_counter += num_targets_in_batch

                # 收集所有样本的误差和标签用于最终评估
                all_errors.extend(errors)
                all_labels.extend(labels.cpu().numpy().flatten())

        # --- 新增：检查是否成功绘图 ---
        if plot_target_label is not None and not plotted and self.rank == 0:
            logging.warning(
                f"Could not find the {plot_target_ordinal + 1}-th {'positive' if plot_target_label == 1 else 'negative'} sample. "
                f"The dataset may contain fewer than this number of samples of that type.")

        all_errors = np.array(all_errors)
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
            fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_idx]
        else:
            best_threshold = threshold
            logging.info(f"Using manually specified threshold: {best_threshold}")

        # 基于最佳阈值进行预测
        predictions = (all_errors >= best_threshold).astype(int)
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
