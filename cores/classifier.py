import logging
import os
import time

import numpy as np
import onnxruntime as ort
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import f1_score
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_qat_fx, convert_fx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from cores.features_generator import FeaturesGeneratorXGB, FeaturesGeneratorCNN, InferenceDataset
from cores.features_generator import HDF5Dataset
from cores.loss import *
from cores.nets import NetAFD


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
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False, sampler=train_sampler)
        best_accuracy = 0.0
        for epoch in range(self.num_epochs):
            if self.qat:
                if epoch == 3:  # 经验值，可调
                    self.model.apply(torch.ao.quantization.disable_observer)
                # if epoch == 5:
                #     self.model.apply(torch.ao.quantization.freeze_bn_stats)
            epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs, feats = self.model(inputs)
                loss, std_loss, cos_loss = self._loss_computation(outputs, feats, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # break
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

    def infer(self, x, batch_size=16):
        dataset = InferenceDataset(x, transform=self.features_generator.transform_sample,
                                   seq_len=self.features_generator.seq_len)
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        predictions, features = [], []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.local_rank)
                outputs, feats = self.model(batch_x)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                predictions.extend(batch_predictions)
                batch_feats = feats.flatten().cpu().numpy()
                features.extend(batch_feats)
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
