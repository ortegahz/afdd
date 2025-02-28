import logging
import os
import time

import numpy as np
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import f1_score
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
        self.local_rank = args.rank if not is_infer else 0
        self.num_epochs = 512
        self.lr = 1e-4
        self.model = NetAFD().to(self.local_rank)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        if is_infer:
            self._load_checkpoint_v0(args)
        elif args.path_ckpt is not None:
            self._load_checkpoint_v0(args.path_ckpt)
            # self._load_checkpoint_v1(ckpt)
        if ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.criterion = FocalLossV0().to(self.local_rank)
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
            # val_accuracy = self.evaluate(x_val, y_val)
            val_accuracy = self.evaluate(data['test_path'])
            epoch_duration = time.time() - epoch_start_time
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

    def evaluate(self, test_path, batch_size=1024):
        # dataset = SignalDataset(x, y, transform=self.features_generator.transform_sample,
        #                         seq_len=self.features_generator.seq_len)
        dataset = HDF5Dataset(test_path, self.features_generator.transform_sample)
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs, _ = self.model(inputs)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                all_predictions.extend(batch_predictions)
                all_labels.extend(labels.cpu().numpy())
        predictions = [1 if prob > 0.5 else 0 for prob in all_predictions]
        # result = accuracy_score(all_labels, predictions)
        result = f1_score(all_labels, predictions)
        return result
