import logging
import pickle

import numpy as np
import torch
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from cores.features_generator import FeaturesGeneratorXGB, FeaturesGeneratorCNN
from cores.loss import FocalLoss
from cores.nets import NetAFD

import os


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
    def __init__(self, rank=0, ddp=False):
        super().__init__()
        self.local_rank = 0  # can not see other gpus
        self.num_epochs = 1024
        self.lr = 1e-3
        self.model = NetAFD().to(self.local_rank)
        if ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.criterion = FocalLoss().to(self.local_rank)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.features_generator = FeaturesGeneratorCNN()
        self.rank = rank

    def train(self, x_train, y_train, x_val, y_val, path_save):
        dataset = self.features_generator.dataset_generate(x_train, y_train, device=f'cuda:{self.local_rank}')
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False, sampler=train_sampler)
        best_accuracy = 0.0
        for epoch in range(self.num_epochs):
            train_sampler.set_epoch(epoch)
            for inputs, labels in loader:
                inputs = inputs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            val_accuracy = self.evaluate(x_val, y_val)
            if self.rank == 0:  # Only log and save on rank 0
                logging.info(
                    f'Epoch [{epoch + 1}/{self.num_epochs}],'
                    f' Loss: {loss.item():.8f},'
                    f' Validation Accuracy: {val_accuracy:.4f},')

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), path_save)
                    # with open(path_save, 'wb') as f:
                    #     pickle.dump(self, f)
                    logging.info(f'Saved new best model with accuracy: {best_accuracy:.4f}')

    def infer(self, x, batch_size=16):
        x = self.features_generator.transform(x, device=f'cuda:{self.local_rank}')
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size].to(self.local_rank)
                outputs = self.model(batch_x)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                predictions.extend(batch_predictions)
        return np.array(predictions)

    def evaluate(self, x, y, batch_size=16):
        x = self.features_generator.transform(x, device=f'cuda:{self.local_rank}')
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size].to(self.local_rank)
                outputs = self.model(batch_x)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                predictions.extend(batch_predictions)
        predictions = [1 if prob > 0.5 else 0 for prob in predictions]
        accuracy = accuracy_score(y, predictions)
        return accuracy
