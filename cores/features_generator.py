import logging

import numpy as np
import pywt
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from utils.macros import DEVICE


class FeaturesGeneratorBase:
    def __init__(self):
        pass


class FeaturesGeneratorXGB(FeaturesGeneratorBase):
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.feature_methods = [self._features_dummy_generate]

    @staticmethod
    def _features_fft_generate(data, num_intervals=64):
        data_array = np.array(data[:, :256])
        fft_values = np.fft.fft(data_array)
        fft_magnitude = np.abs(fft_values)
        interval_length = fft_magnitude.shape[1] // num_intervals
        fft_magnitude_pick = fft_magnitude[:, :interval_length * num_intervals][:, np.newaxis, :]
        fft_magnitude_pick_reshape = fft_magnitude_pick.reshape(len(fft_magnitude), -1, interval_length)
        features = np.mean(fft_magnitude_pick_reshape, axis=2)
        feature_names = [f'fft_magnitude_{i}' for i in range(num_intervals)]
        return features, feature_names

    @staticmethod
    def _features_dummy_generate(data, num_intervals=64):
        data_array = np.array(data[:, :256])
        interval_length = data_array.shape[1] // num_intervals
        data_array_pick = data_array[:, :interval_length * num_intervals][:, np.newaxis, :]
        data_array_pick_reshape = data_array_pick.reshape(len(data_array), -1, interval_length)
        features = np.mean(data_array_pick_reshape ** 2, axis=2)
        feature_names = [f'dummy_{i}' for i in range(num_intervals)]
        return features, feature_names

    @staticmethod
    def _features_hf_generate(data, num_intervals=64):
        data_array = np.array(data[:, 256:])
        interval_length = data_array.shape[1] // num_intervals
        data_array_pick = data_array[:, :interval_length * num_intervals][:, np.newaxis, :]
        data_array_pick_reshape = data_array_pick.reshape(len(data_array), -1, interval_length)
        features = np.mean(data_array_pick_reshape ** 2, axis=2)
        feature_names = [f'hf_{i}' for i in range(num_intervals)]
        return features, feature_names

    @staticmethod
    def _features_wt_generate(data, wavelet_type='sym2', wavelet_max_level=4):
        data = np.array(data[:, :256])
        features = np.zeros((data.shape[0], wavelet_max_level))
        for index, signal in enumerate(data):
            coeffs = pywt.wavedec(signal, wavelet_type, level=wavelet_max_level)
            for level, coeff in enumerate(coeffs[1:], start=1):
                features[index, level - 1] = np.mean(coeff ** 2)
        feature_names = [f'wt_level_{i}_avg' for i in range(1, wavelet_max_level + 1)]
        return features, feature_names

    def generate(self, x, y=None):
        features_lst, feature_names_lst = list(), list()
        for feature_method in self.feature_methods:
            _features, _feature_names = feature_method(x)
            features_lst.append(_features)
            feature_names_lst.extend(_feature_names)
        features_final = np.concatenate(features_lst, axis=1)
        self.feature_names = feature_names_lst
        features = xgb.DMatrix(features_final, label=y, feature_names=self.feature_names)
        return features


class FeaturesGeneratorCNN(FeaturesGeneratorXGB):
    def __init__(self):
        super().__init__()
        self.transformer = StandardScaler()

    def transform(self, x):
        x = x[:, :256]
        x_scaled = (x - 2048) / 4096
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        x_tensor = x_tensor.unsqueeze(1)
        return x_tensor.to(DEVICE)

    def dataset_generate(self, x, y=None):
        x_tensor = self.transform(x)
        # y_tensor = torch.tensor(y, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)
        dataset = TensorDataset(x_tensor, y_tensor)
        return dataset
