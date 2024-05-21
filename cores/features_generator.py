import numpy as np
import xgboost as xgb


class FeaturesGeneratorBase:
    def __init__(self):
        pass


class FeaturesGeneratorXGB(FeaturesGeneratorBase):
    def __init__(self):
        super().__init__()
        self.feature_names = None

    @staticmethod
    def features_fft_generate(data, num_intervals=256):
        data_array = np.array(data)
        fft_values = np.fft.fft(data_array)
        fft_magnitude = np.abs(fft_values)
        interval_length = len(fft_magnitude[1]) // num_intervals
        fft_magnitude_pick = fft_magnitude[:, :interval_length * num_intervals][:, np.newaxis, :]
        fft_magnitude_pick_reshape = fft_magnitude_pick.reshape(len(fft_magnitude), -1, interval_length)
        features = np.mean(fft_magnitude_pick_reshape, axis=2)
        feature_names = [f'fft_magnitude_{i}' for i in range(num_intervals)]
        return features, feature_names

    def generate(self, x, y=None):
        _features, self.feature_names = self.features_fft_generate(x)
        features = xgb.DMatrix(_features, label=y, feature_names=self.feature_names)
        return features
