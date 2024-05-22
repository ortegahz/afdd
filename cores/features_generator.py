import numpy as np
import pywt
import xgboost as xgb


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
        data_array = np.array(data)
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
        data_array = np.array(data)
        interval_length = data_array.shape[1] // num_intervals
        data_array_pick = data_array[:, :interval_length * num_intervals][:, np.newaxis, :]
        data_array_pick_reshape = data_array_pick.reshape(len(data_array), -1, interval_length)
        features = np.mean(data_array_pick_reshape ** 2, axis=2)
        feature_names = [f'dummy_{i}' for i in range(num_intervals)]
        return features, feature_names

    @staticmethod
    def _features_wt_generate(data, wavelet_type='sym2', wavelet_max_level=4):
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
