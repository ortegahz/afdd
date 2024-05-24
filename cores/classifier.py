import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import DataLoader

from cores.features_generator import FeaturesGeneratorXGB, FeaturesGeneratorCNN
from cores.nets import NetAFD
from utils.macros import DEVICE


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
    def __init__(self):
        super().__init__()
        self.model = NetAFD().to(DEVICE)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.features_generator = FeaturesGeneratorCNN()

    def train(self, x, y, num_epochs=1):
        dataset = self.features_generator.dataset_generate(x, y)
        loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
        for epoch in range(num_epochs):
            for inputs, labels in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def infer(self, x, batch_size=16):
        x = self.features_generator.transform(x)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                outputs = self.model(batch_x)
                batch_predictions = torch.sigmoid(outputs).flatten().cpu().numpy()
                predictions.extend(batch_predictions)
        return np.array(predictions)
