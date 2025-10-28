import h5py
import numpy as np
import pywt
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from utils.macros import SAMPLE_RATE


class FeaturesGeneratorBase:
    def __init__(self):
        pass


class FeaturesGeneratorXGB(FeaturesGeneratorBase):
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.feature_methods = [self._features_fft_generate_v1]

    @staticmethod
    def _features_fft_generate_v1(data, sample_rate=SAMPLE_RATE):
        # Convert data to a NumPy array and trim it to cover one period of 50 Hz
        data_array = np.array(data[:, :int(sample_rate / 50)])
        fft_values = np.fft.fft(data_array)
        fft_magnitude = np.abs(fft_values)
        # Calculate corresponding frequencies, taking only positive frequencies
        freqs = np.fft.fftfreq(data_array.shape[1], d=1 / sample_rate)
        positive_freqs = freqs[:data_array.shape[1] // 2]
        fft_magnitude = fft_magnitude[:, :positive_freqs.size]  # Only positive part of the spectrum
        # The features are simply the magnitudes of these frequencies
        features = fft_magnitude
        # Generate feature names for each frequency bin
        feature_names = [f'fft_magnitude_{freq:.2f}Hz' for freq in positive_freqs]
        return features, feature_names

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
        self.seq_len = int(SAMPLE_RATE / 50)

    @staticmethod
    def transform_sample(x_sample):
        x_tensor = torch.tensor(x_sample, dtype=torch.float32)
        x_signal = x_tensor.clone()
        x_signal = (x_signal - 2048) / 4096
        x_signal = x_signal.unsqueeze(0)

        # Calculate padding necessary to make length a multiple of 32
        current_length = x_signal.shape[-1]
        padding_required = (32 - (current_length % 32)) % 32
        padding = (0, padding_required)  # (left_pad, right_pad)

        # Pad the signal
        x_signal = F.pad(x_signal, padding, "constant", 0)

        return x_signal

    @staticmethod
    def transform_sample_ae(x_sample):
        x_tensor = torch.tensor(x_sample, dtype=torch.float32)
        x_signal = x_tensor.clone()
        x_signal = (x_signal - 2048) / 4096
        x_signal = x_signal.unsqueeze(0)

        # Calculate padding necessary to make length a multiple of 32
        current_length = x_signal.shape[-1]
        padding_required = (32 - (current_length % 32)) % 32
        padding = (0, padding_required)  # (left_pad, right_pad)

        # Pad the signal
        x_signal = F.pad(x_signal, padding, "constant", 0)

        return x_signal

    # @staticmethod
    # def transform_sample(x_sample):
    #     """
    #     将样本转换为Tensor，进行Min-Max归一化，并进行填充。
    #     """
    #     x_tensor = torch.tensor(x_sample, dtype=torch.float32)
    #     x_signal = x_tensor.clone()
    #
    #     # --- Min-Max 归一化开始 ---
    #     # 1. 找到当前样本的最小值和最大值
    #     min_val = torch.min(x_signal)
    #     max_val = torch.max(x_signal)
    #
    #     # 2. 避免除以零的边缘情况 (如果信号是常数)
    #     if max_val - min_val > 0:
    #         # 应用 Min-Max 公式: (x - min) / (max - min)
    #         x_signal = (x_signal - min_val) / (max_val - min_val)
    #     else:
    #         # 如果所有值都相同，则信号是平坦的，可以将其设置为全零
    #         x_signal = torch.zeros_like(x_signal)
    #     # --- Min-Max 归一化结束 ---
    #
    #     # 增加一个批次维度 (batch dimension)
    #     x_signal = x_signal.unsqueeze(0)
    #
    #     # 计算使长度成为32的倍数所需的填充量
    #     current_length = x_signal.shape[-1]
    #     padding_required = (32 - (current_length % 32)) % 32
    #     # 定义填充: (左侧填充, 右侧填充)
    #     padding = (0, padding_required)
    #
    #     # 对信号进行填充
    #     x_signal = F.pad(x_signal, padding, "constant", 0)
    #
    #     return x_signal

    # @staticmethod
    # def transform_sample(x_sample, seq_len):
    #     x_tensor = torch.tensor(x_sample, dtype=torch.float32)
    #     x_signal = x_tensor.clone()
    #     x_signal[:seq_len] = (x_signal[:seq_len] - 2048) / 4096
    #     x_signal[seq_len:] = 0
    #     x_signal = x_signal.unsqueeze(0)
    #     return x_signal

    # @staticmethod
    # def transform_sample(x_sample):
    #     seq_len = len(x_sample)
    #     # Convert to tensor and normalize the first part of the signal
    #     x_tensor = torch.tensor(x_sample, dtype=torch.float32)
    #     x_signal = (x_tensor[:seq_len] - 2048) / 4096
    #
    #     # Perform FFT and get the magnitudes
    #     fft_values = torch.fft.fft(x_signal)
    #     fft_magnitude = torch.abs(fft_values)
    #
    #     # Use high-frequency components from the positive frequencies
    #     half_point = seq_len // 2
    #     quarter_point = seq_len // 4
    #     # high_freq_component = fft_magnitude[quarter_point:half_point]
    #     high_freq_component = fft_magnitude  # all
    #
    #     # Calculate padding necessary to make length a multiple of 32
    #     current_length = high_freq_component.shape[0]
    #     padding_required = (32 - (current_length % 32)) % 32
    #     padding = (0, padding_required)  # (left_pad, right_pad)
    #
    #     # Pad the signal
    #     high_freq_component = F.pad(high_freq_component, padding, "constant", 0)
    #
    #     # Reshape to add channel dimension for CNN input
    #     high_freq_component = high_freq_component.unsqueeze(0)
    #
    #     return high_freq_component

    # @staticmethod
    # def transform_sample(x_sample, seq_len, wavelet='db4', level=3):
    #     x_tensor = torch.tensor(x_sample, dtype=torch.float32)
    #     x_signal = x_tensor.clone()
    #     x_signal[:seq_len] = (x_signal[:seq_len] - 2048) / 4096
    #     x_time_np = x_signal[:seq_len].numpy()
    #     _th_arc = (x_time_np[seq_len - 1] - np.mean(x_time_np)) * 0.01
    #     _cnt_arc_norm = np.sum(np.abs(x_time_np - np.mean(x_time_np)) < _th_arc) / seq_len
    #     x_signal[0:16] = _cnt_arc_norm  # anti-pool op
    #     coeffs = pywt.wavedec(x_time_np, wavelet, level=level)
    #     wavelet_coefficients = torch.from_numpy(np.concatenate(coeffs))
    #     if wavelet_coefficients.numel() < len(x_signal[seq_len:]):
    #         x_signal[seq_len:seq_len + len(wavelet_coefficients)] = wavelet_coefficients
    #         x_signal[seq_len + len(wavelet_coefficients):] = 0
    #     else:
    #         x_signal[seq_len:] = wavelet_coefficients[:len(x_signal) - seq_len]
    #     x_signal = x_signal.unsqueeze(0)
    #     return x_signal

    def dataset_generate(self, x, y=None):
        dataset = SignalDataset(x, y, transform=self.transform_sample, seq_len=self.seq_len)
        return dataset


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform, seq_len):
        self.x = x
        self.y = y
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        ones = np.ones(len(x_sample))
        x_sample = np.concatenate((x_sample, ones))
        y_sample = self.y[idx] if self.y is not None else None
        x_signal = self.transform(x_sample, self.seq_len)
        if y_sample is not None:
            y_tensor = torch.tensor(y_sample, dtype=torch.float32).view(-1)
            return x_signal, y_tensor
        else:
            return x_signal


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file_path, transform):
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.labels = self.hdf5_file['labels']
        self.features = self.hdf5_file['features']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        self.hdf5_file.close()

    def __getitem__(self, idx):
        x_sample = self.features[idx]
        # ones = np.ones(len(x_sample))
        # x_sample = np.concatenate((x_sample, ones))
        y_sample = self.labels[idx]
        # x_signal = self.transform(x_sample, int(SAMPLE_RATE / 50))
        x_signal = self.transform(x_sample)
        y_tensor = torch.tensor(y_sample, dtype=torch.float32).view(-1)
        return x_signal, y_tensor


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, x, transform, seq_len):
        self.x = x
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        # x_signal = self.transform(x_sample, self.seq_len)
        x_signal = self.transform(x_sample)
        return x_signal
