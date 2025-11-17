import random

import h5py
import numpy as np
import pywt
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pyts.image import MarkovTransitionField
from torch.utils.data import TensorDataset

from utils.macros import SAMPLE_RATE, MIN_VAL_TH


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

        # 找到序列的最大值和最小值
        min_val = torch.min(x_tensor)
        max_val = torch.max(x_tensor)

        # 根据最大值和最小值将数据归一化到 [-1, 1]
        # 如果 max_val 等于 min_val，说明信号是恒定的，为避免除以零，将其归一化为0
        if (max_val - min_val) > 0:
            x_signal = 2 * (x_tensor - min_val) / (max_val - min_val) - 1
        else:
            x_signal = torch.zeros_like(x_tensor)

        # 增加一个维度 (channel/batch dimension)
        x_signal = x_signal.unsqueeze(0)

        # --- 后续的 padding 逻辑保持不变 ---

        # 计算使长度成为32的倍数所需的填充量
        current_length = x_signal.shape[-1]
        padding_required = (32 - (current_length % 32)) % 32
        padding = (0, padding_required)  # (左填充, 右填充)

        # 填充信号
        x_signal = F.pad(x_signal, padding, "constant", 0)

        return x_signal

    @staticmethod
    def transform_sample_ae_mtf(x_sample, image_size=64, n_bins=8):
        """
        Transforms a signal into a Markov Transition Field (MTF) image.
        1. Normalizes the signal to [-1, 1].
        2. Pads it to be a multiple of 32.
        3. Resizes the signal to `image_size`.
        4. Computes the MTF.
        """
        # --- 1. Normalization (same as transform_sample_ae) ---
        x_tensor = torch.tensor(x_sample, dtype=torch.float32)
        min_val = torch.min(x_tensor)
        max_val = torch.max(x_tensor)
        if (max_val - min_val) > 0:
            x_signal_1d = 2 * (x_tensor - min_val) / (max_val - min_val) - 1
        else:
            x_signal_1d = torch.zeros_like(x_tensor)

        # --- 2. Padding (same as transform_sample_ae) ---
        x_signal_1d = x_signal_1d.unsqueeze(0)  # add channel dim for padding
        current_length = x_signal_1d.shape[-1]
        padding_required = (32 - (current_length % 32)) % 32
        padding = (0, padding_required)
        x_signal_padded = F.pad(x_signal_1d, padding, "constant", 0)

        # --- 3. Resize signal for MTF ---
        # pyts MTF output size is (n_timestamps, n_timestamps). We resize the signal first.
        # Use unsqueeze to create a batch dimension for interpolate: (1, 1, seq_len)
        resized_signal_tensor = F.interpolate(x_signal_padded.unsqueeze(0), size=image_size, mode='linear',
                                              align_corners=False)
        # pyts expects a 2D numpy array (n_samples, n_timestamps)
        resized_signal_np = resized_signal_tensor.squeeze(0).numpy()

        # --- 4. MTF Transformation ---
        mtf = MarkovTransitionField(n_bins=n_bins, strategy='uniform')
        mtf_image = mtf.fit_transform(resized_signal_np)  # output shape (1, image_size, image_size)

        # Convert back to tensor, shape (1, image_size, image_size) for a single channel image
        return torch.from_numpy(mtf_image).float().squeeze(0).unsqueeze(0)

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


class HDF5SPDataset(torch.utils.data.Dataset):
    """
    HDF5 Signal Processing Dataset for dynamic slicing.
    Reads long sequences from an HDF5 file and yields random, fixed-length slices.
    """

    def __init__(self, hdf5_file_path, transform, seq_len, min_delta=MIN_VAL_TH * 2, samples_per_epoch=1024 * 64):
        self.hdf5_file_path = hdf5_file_path
        self.transform = transform
        self.seq_len = seq_len
        self.min_delta = min_delta
        self.samples_per_epoch = samples_per_epoch

        # H5 file handle and keys will be initialized in the first __getitem__ call
        # This is to ensure compatibility with multi-worker DataLoader
        self.hdf5_file = None
        self.group_keys = None

    def __len__(self):
        # This determines the number of samples per epoch a
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Initialize file handle if not already done (for multi-worker support)
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
            self.group_keys = list(self.hdf5_file.keys())

        while True:
            # 1. Randomly select a group (long sequence)
            random_group_name = random.choice(self.group_keys)
            group = self.hdf5_file[random_group_name]
            full_signal = group['signal']

            # Ensure the sequence is long enough to be sliced
            if len(full_signal) < self.seq_len:
                continue

            # 2. Randomly select a starting index for slicing
            start_idx = random.randint(0, len(full_signal) - self.seq_len)
            end_idx = start_idx + self.seq_len
            x_sample = full_signal[start_idx:end_idx]

            # 3. Validate the sample
            if np.max(x_sample) - np.min(x_sample) > self.min_delta:
                # If valid, process and return the sample and its label
                y_sample_slice = group['label_seq'][start_idx:end_idx]
                y_label = 1 if np.any(y_sample_slice > 0) else 0

                x_signal = self.transform(x_sample.astype(np.float32))
                y_tensor = torch.tensor(y_label, dtype=torch.float32).view(-1)
                return x_signal, y_tensor


class HDF5SequentialSliceDataset(torch.utils.data.Dataset):
    """
    用于训练和评估的HDF5数据集，采用确定性的顺序滑窗切片。
    它会遍历HDF5文件中的所有长序列，并按固定的步长生成所有可能的切片。
    只保留那些信号变化幅度超过阈值的切片，以过滤掉无效或平坦的信号段。
    """

    def __init__(self, hdf5_file_path, transform, seq_len, step, min_delta=MIN_VAL_TH * 2, only_normal=False):
        self.hdf5_file_path = hdf5_file_path
        self.transform = transform
        self.seq_len = seq_len
        self.step = step
        self.hdf5_file = None  # Defer file opening to __getitem__ for multiprocessing

        # 预计算所有有效切片的位置信息
        self.slices = []
        with h5py.File(self.hdf5_file_path, 'r') as f:
            self.group_keys = list(f.keys())
            for key in self.group_keys:
                signal_len = len(f[key]['signal'])
                # 从0开始，以step为步长，生成所有可能的起始点
                for start_idx in range(0, signal_len - self.seq_len + 1, self.step):
                    # 如果只需要正常样本，则预先检查标签
                    if only_normal:
                        y_sample_slice = f[key]['label_seq'][start_idx:start_idx + self.seq_len]
                        if np.any(y_sample_slice > 0):
                            continue  # 跳过异常样本

                    # 读取切片数据以进行检查
                    x_sample = f[key]['signal'][start_idx: start_idx + self.seq_len]

                    # 检查振幅差是否大于阈值，过滤无效样本
                    if np.max(x_sample) - np.min(x_sample) > min_delta:
                        # 只有当切片有效时，才将其信息添加到列表中
                        self.slices.append((key, start_idx))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        根据索引 (idx) 获取一个经过处理的数据样本及其标签。
        """
        # 为了支持PyTorch的DataLoader多进程加载，每个worker需要独立的文件句柄
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        # 1. 根据索引从预计算的切片列表中获取该切片的信息
        group_key, start_idx = self.slices[idx]

        # 2. 计算切片的结束位置
        end_idx = start_idx + self.seq_len

        # 3. 从HDF5文件中定位到对应的长序列
        group = self.hdf5_file[group_key]

        # 4. 提取该切片对应的信号数据和标签序列
        x_sample = group['signal'][start_idx:end_idx].astype(np.float32)
        y_sample_slice = group['label_seq'][start_idx:end_idx]

        # 5. 为整个切片确定一个单一的标签：
        #    如果切片内的任何一个点的标签值大于0（即为故障），则整个切片的标签为1，否则为0。
        y_label = 1 if np.any(y_sample_slice > 0) else 0

        # 6. 对提取出的信号样本应用预设的转换函数
        x_signal = self.transform(x_sample)

        # 7. 将计算出的标量标签转换为PyTorch张量
        y_tensor = torch.tensor(y_label, dtype=torch.float32).view(-1)

        return x_signal, y_tensor

        # 5. 为整个切片确定一个单一的标签：
        #    如果切片内的任何一个点的标签值大于0（即为故障），则整个切片的标签为1，否则为0。
        y_label = 1 if np.any(y_sample_slice > 0) else 0

        # ---- 以下是补全的部分 ----

        # 6. 对提取出的信号样本应用预设的转换函数 (self.transform)。
        #    这个函数通常负责数据归一化、填充和转换为PyTorch张量。
        x_signal = self.transform(x_sample)

        # 7. 将计算出的标量标签 (y_label) 转换为PyTorch张量。
        #    - dtype=torch.float32 是为了与模型输出和损失函数（如BCE）的类型匹配。
        #    - .view(-1) 将其形状从标量变为一维张量 (例如, 0 变为 tensor([0.]) )。
        y_tensor = torch.tensor(y_label, dtype=torch.float32).view(-1)

        # 8. 返回处理好的信号张量和标签张量，这是DataLoader期望的格式。
        return x_signal, y_tensor
