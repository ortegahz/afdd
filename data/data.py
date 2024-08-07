import glob
import logging
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.io
from PyEMD import EMD
from screeninfo import get_monitors

from utils.macros import *
from utils.structs import *
from utils.utils import load_bin


class DataBase:
    def __init__(self):
        self.db = dict()


class DataV0(DataBase):
    def __init__(self, dir_in):
        super().__init__()
        self.dir_in = dir_in

    def load(self):
        files_bin = glob.glob(os.path.join(self.dir_in, '*.BIN'))
        logging.info(files_bin)
        files_power = [file for file in files_bin if 'power' in file]
        logging.info(files_power)
        keys = [os.path.basename(file_power).strip().split('_power')[0] for file_power in files_power]
        logging.info(keys)
        self.db = {key: Signals() for key in keys}
        logging.info(self.db)
        for key in self.db.keys():
            self.db[key].seq_power = np.array(load_bin(os.path.join(self.dir_in, f'{key}_power.BIN')))
            self.db[key].seq_hf = np.array(load_bin(os.path.join(self.dir_in, f'{key}_highfreq.BIN')))
            self.db[key].len = min(len(self.db[key].seq_power), len(self.db[key].seq_hf))
            self.db[key].seq_state_arc = np.array([0] * self.db[key].len)
            self.db[key].seq_state_normal = np.array([0] * self.db[key].len)
        file_label = glob.glob(os.path.join(self.dir_in, '*.xlsx'))
        if not len(file_label) == 1:
            return
        df = pd.read_excel(file_label[0], engine='openpyxl')
        alarm_index = df.columns.get_loc("报警位置")
        normal_index = df.columns.get_loc("正常位置")
        for index, row in df.iterrows():
            key = row['文件名'].replace('_power', '')
            alarm_interval_0 = row[alarm_index].split('~') if pd.notna(row[alarm_index]) else None
            normal_interval_0 = row[normal_index].split('~') if pd.notna(row[normal_index]) else None
            if alarm_interval_0 is not None:
                _idx_s, _idx_e = int(alarm_interval_0[0]), int(alarm_interval_0[1])
                _idx_s, _idx_e = max(0, _idx_s), min(self.db[key].len, _idx_e)
                self.db[key].seq_state_arc[_idx_s:_idx_e] = STATE_INDICATE_VAL
            if normal_interval_0 is not None:
                self.db[key].seq_state_normal[int(normal_interval_0[0]):int(normal_interval_0[1])] = \
                    STATE_INDICATE_VAL / 2
            for i in range(1, len(df.columns) - max(alarm_index, normal_index)):
                next_alarm_col = alarm_index + i
                next_normal_col = normal_index + i
                if next_alarm_col < len(df.columns) and pd.notna(row[next_alarm_col]):
                    next_alarm_interval = row[next_alarm_col].split('~')
                    _idx_s, _idx_e = \
                        int(next_alarm_interval[0]), int(next_alarm_interval[1])
                    _idx_s, _idx_e = max(0, _idx_s), min(self.db[key].len, _idx_e)
                    self.db[key].seq_state_arc[_idx_s:_idx_e] = \
                        STATE_INDICATE_VAL
                if next_normal_col < len(df.columns) and pd.notna(row[next_normal_col]):
                    next_normal_interval = row[next_normal_col].split('~')
                    self.db[key].seq_state_normal[int(next_normal_interval[0]):int(next_normal_interval[1])] = \
                        STATE_INDICATE_VAL / 2

    def plot(self, idx_pick=0, pause_time_s=1024, dir_save=None, show=True):
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        for idx in range(len(self.db.keys())):
            if 0 <= idx_pick != idx:
                continue
            key = list(self.db.keys())[idx]
            seq_power = self.db[key].seq_power
            seq_hf = self.db[key].seq_hf
            seq_len = self.db[key].len
            seq_state_arc = self.db[key].seq_state_arc
            seq_state_normal = self.db[key].seq_state_normal
            time_stamps = np.array(range(seq_len))
            ax1.plot(time_stamps, np.array(seq_power).astype(float), label='power')
            ax1.plot(time_stamps, np.array(seq_state_arc).astype(float), label='state_arc')
            ax1.plot(time_stamps, np.array(seq_state_normal).astype(float), label='state_normal')
            ax1.set_xlim(0, seq_len)
            ax1.set_ylim(0, 4096)
            ax1.legend()
            # ax2.plot(time_stamps, np.array(seq_hf).astype(float), label='hf')
            # f, t, Zxx = signal.stft(np.array(seq_power).astype(float), self.db[key].sps, nperseg=1024)
            # ax2.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            # wavelet = 'cmor'
            # scales = np.arange(1, 8)
            # coefficients, frequencies = pywt.cwt(np.array(seq_power).astype(float), scales, wavelet, 1 / self.db[key].sps)
            # plt.pcolormesh(time_stamps, frequencies, np.abs(coefficients), shading='gouraud')
            # wavelet, max_level = 'db5', 4  # hu's work --> db5 with 4 levels
            wavelet = 'sym2'
            logging.info(pywt.wavelist())
            max_level = pywt.dwt_max_level(seq_len, pywt.Wavelet(wavelet).dec_len)
            level = max_level
            coeffs = pywt.wavedec(np.array(seq_power).astype(float), wavelet, level=max_level)
            plt.plot(coeffs[level], label=f'{wavelet} level {level} [{max_level}]')
            ax2.set_xlim(0, len(coeffs[level]))
            # val_lim = 2 ** 15
            # ax2.set_ylim(-val_lim, val_lim)
            ax2.legend()
            plt.title(f'{os.path.basename(self.dir_in)} {key}')
            plt.tight_layout()
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            if dir_save is not None:
                plt.savefig(os.path.join(dir_save, f'{key}'))
            if show:
                plt.show()
                plt.pause(pause_time_s)
            plt.clf()


class DataV1(DataBase):
    @dataclass
    class Signals:
        seq_adc: list = field(default_factory=list)
        seq_len: int = 0

    def __init__(self, dir_in):
        super().__init__()
        self.path_in = dir_in
        self.db['default'] = self.Signals()

    def load(self):
        data = pd.read_csv(self.path_in, header=3)
        signal = data.iloc[:, 1]
        self.db['default'].seq_adc = signal.tolist()
        self.db['default'].seq_len = len(self.db['default'].seq_adc)
        _seq_adc_set = set(self.db['default'].seq_adc)
        # logging.info(f'[{len(_seq_adc_set)}] {_seq_adc_set}')

    def plot(self, pause_time_s=1024, dir_save=None, show=False, save_name=None):
        plt.ion()
        key = 'default'
        seq_adc = self.db[key].seq_adc
        seq_len = self.db[key].seq_len
        time_stamps = np.array(range(seq_len))
        # seq_adc_filtered = [64.0 if '∞' in str(item) else item for item in seq_adc]
        seq_adc_np = np.array(seq_adc)
        # seq_adc_np[seq_adc_np == '∞'] = 64.0
        # seq_adc_np[seq_adc_np == '-∞'] = -64.0
        plt.plot(time_stamps, seq_adc_np.astype(float), label='seq_adc')
        plt.xticks(np.arange(0, seq_len, SAMPLE_RATE / 50))
        plt.xlim(0, seq_len)
        # plt.ylim(-8, 8)
        plt.legend()
        plt.tight_layout()
        plt.title(save_name)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if show:
            plt.show()
            plt.pause(pause_time_s)
        if dir_save is not None:
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            fig = plt.gcf()
            fig.set_size_inches(screen_width / fig.dpi, screen_height / fig.dpi)
            plt.savefig(os.path.join(dir_save, save_name + '.png'), dpi=fig.dpi)
        plt.close()


class DataV2(DataV1):
    def __init__(self, dir_in):
        super().__init__(dir_in)

    def load(self):
        _mat = scipy.io.loadmat(self.path_in)
        self.db['default'].seq_adc = _mat['A'].flatten().tolist()[::1]
        self.db['default'].seq_len = len(self.db['default'].seq_adc)


class DataV3(DataV2):
    def __init__(self, dir_in):
        super().__init__(dir_in)

    def load(self):
        _npy = np.load(self.path_in)
        # _npy = _npy * 2048 / 40 + 2048
        self.db['default'].seq_adc = _npy.tolist()
        self.db['default'].seq_len = len(self.db['default'].seq_adc)


class DataV4(DataV3):
    def __init__(self, addr):
        super().__init__(addr)

    def load(self):
        _key = 'default'
        _npy = np.load(self.path_in)
        _npy = _npy * 2048 / 40 + 2048
        self.db[_key].seq_power = _npy.tolist()
        self.db[_key].len = len(self.db['default'].seq_power)
        self.db[_key].seq_hf = np.array([1] * self.db[_key].len)
        self.db[_key].seq_state_arc = np.array([0] * self.db[_key].len)
        self.db[_key].seq_state_normal = np.array([0] * self.db[_key].len)

        _file_label = self.path_in.replace('.npy', '.txt')
        if not os.path.exists(_file_label):
            return
        with open(_file_label, 'r') as f:
            _lines = f.readlines()
        for _line in _lines:
            idx_s, idx_e = _line.strip().split(' ')
            idx_s, idx_e = max(int(idx_s), 0), min(int(idx_e), self.db[_key].len)
            self.db[_key].seq_state_arc[idx_s:idx_e] = STATE_INDICATE_VAL


class DataRT(DataBase):
    @dataclass
    class Signals:
        seq_power: list = field(default_factory=list)
        seq_power_ss: list = field(default_factory=list)
        seq_hf: list = field(default_factory=list)
        seq_filtered: list = field(default_factory=list)
        seq_filter_envelope: list = field(default_factory=list)
        seq_power_mean: list = field(default_factory=list)
        seq_wavelet: list = field(default_factory=list)
        seq_state_gt_arc: list = field(default_factory=list)
        seq_state_gt_normal: list = field(default_factory=list)
        seq_state_pred_arc: list = field(default_factory=list)
        seq_state_pred_classifier: list = field(default_factory=list)
        seq_wt_power_bg: list = field(default_factory=list)
        seq_wt_power_pioneer: list = field(default_factory=list)
        info_pred_peaks: list = field(default_factory=list)
        info_af_scores: list = field(default_factory=list)
        seq_len: int = 0

    def __init__(self, wavelet_max_level=4):
        super().__init__()
        self.db['rt'] = self.Signals()
        self.wavelet_max_level = wavelet_max_level

    def update(self, cur_power, cur_hf=0, cur_state_gt_arc=0, cur_state_gt_normal=0):
        self.db['rt'].seq_power.append(cur_power)
        self.db['rt'].seq_hf.append(cur_hf)
        self.db['rt'].seq_filtered.append(0)
        self.db['rt'].seq_filter_envelope.append(0)
        self.db['rt'].seq_power_mean.append(cur_power)
        self.db['rt'].seq_wavelet.append([0] * self.wavelet_max_level)
        self.db['rt'].seq_state_pred_arc.append(0)
        self.db['rt'].seq_state_pred_classifier.append(0)
        self.db['rt'].seq_state_gt_arc.append(cur_state_gt_arc)
        self.db['rt'].seq_state_gt_normal.append(cur_state_gt_normal)
        self.db['rt'].seq_wt_power_bg.append([0] * self.wavelet_max_level)
        self.db['rt'].seq_wt_power_pioneer.append([0] * self.wavelet_max_level)
        self.db['rt'].seq_len += 1

    def plot(self, pause_time_s=1024, dir_save=None, save_name=None, show=True):
        plt.ion()
        key = 'rt'
        seq_power = self.db[key].seq_power
        seq_hf = self.db[key].seq_hf
        seq_filtered = self.db[key].seq_filtered
        seq_filter_envelope = self.db[key].seq_filter_envelope
        seq_power_mean = self.db[key].seq_power_mean
        seq_len = self.db[key].seq_len
        seq_state_pred_arc = self.db[key].seq_state_pred_arc
        seq_state_pred_classifier = self.db[key].seq_state_pred_classifier
        seq_state_arc = self.db[key].seq_state_gt_arc
        seq_state_normal = self.db[key].seq_state_gt_normal
        info_pred_peaks = self.db[key].info_pred_peaks
        info_af_scores = self.db[key].info_af_scores
        time_stamps = np.array(range(seq_len))
        plt.subplot(self.wavelet_max_level + 1, 1, 1)
        plt.plot(time_stamps, np.array(seq_power).astype(float), label='power')
        plt.plot(time_stamps, np.array(seq_state_pred_classifier).astype(float), label='seq_state_pred_classifier')
        plt.plot(time_stamps, np.array(seq_state_pred_arc).astype(float), label='state_arc_pred', color='red')
        plt.plot(time_stamps, np.array(seq_state_arc).astype(float), label='state_arc')
        plt.plot(time_stamps, np.array(seq_state_normal).astype(float), label='state_normal')
        plt.plot(time_stamps, np.array(seq_power_mean).astype(float), label='power_mean')
        plt.plot(info_pred_peaks, np.array(seq_power).astype(float)[info_pred_peaks], 'x', label='peaks')
        for i, peak in enumerate(info_pred_peaks):
            plt.annotate(f'{info_af_scores[i]: .2f}',
                         (peak, seq_power[peak]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         arrowprops=dict(arrowstyle="->", color='black'))
        plt.xticks(np.arange(0, seq_len, SAMPLE_RATE / 50))
        plt.xlim(0, seq_len)
        plt.ylim(0, 4096)
        plt.legend()
        plt.subplot(self.wavelet_max_level + 1, 1, 2)
        plt.plot(time_stamps, np.array(seq_filtered).astype(float), label='seq_filtered')
        plt.plot(time_stamps, np.array(seq_hf).astype(float), label='seq_hf')
        plt.plot(time_stamps, np.array(seq_filter_envelope).astype(float), label='seq_filter_envelope')
        plt.xlim(0, seq_len)
        plt.ylim(0, 16)
        plt.legend()
        # wavelet, max_level = 'sym2', 4
        # coeffs = pywt.wavedec(np.array(seq_power).astype(float), wavelet, level=max_level)
        # for level, coeff in enumerate(coeffs[1:], start=1):
        #     plt.subplot(max_level + 1, 1, level + 1)
        #     plt.plot(coeffs[level], label=f'{wavelet} level {level} [{max_level}]')
        #     plt.xlim(0, len(coeffs[level]))
        #     plt.legend()
        seq_wavelet = np.array(self.db[key].seq_wavelet)
        seq_wt_power_bg = np.array(self.db[key].seq_wt_power_bg)
        seq_wt_power_pioneer = np.array(self.db[key].seq_wt_power_pioneer)
        for i in range(self.wavelet_max_level):
            plt.subplot(self.wavelet_max_level + 1, 1, i + 2)
            plt.plot(seq_wavelet[:, i], label=f'level {i + 1} [{self.wavelet_max_level}]')
            plt.plot(seq_wt_power_pioneer[:, i], label=f'seq_wt_power_pioneer')
            plt.plot(seq_wt_power_bg[:, i], label=f'seq_wt_power_bg')
            plt.xlim(0, seq_len)
            plt.legend()
        plt.tight_layout()
        plt.title(save_name)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if show:
            plt.show()
            plt.pause(pause_time_s)
        if dir_save is not None:
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            fig = plt.gcf()
            fig.set_size_inches(screen_width / fig.dpi, screen_height / fig.dpi)
            plt.savefig(os.path.join(dir_save, save_name), dpi=fig.dpi)
        plt.close()

    def plot_cwt(self, pause_time_s=1024, dir_save=None, save_name=None, show=True):
        plt.ion()
        key = 'rt'
        seq_power = self.db[key].seq_power
        seq_len = self.db[key].seq_len
        seq_state_pred_arc = self.db[key].seq_state_pred_arc
        seq_state_arc = self.db[key].seq_state_gt_arc
        seq_state_normal = self.db[key].seq_state_gt_normal
        time_stamps = np.array(range(seq_len))
        plt.subplot(self.wavelet_max_level + 1, 1, 1)
        plt.plot(time_stamps, np.array(seq_power).astype(float), label='power')
        plt.plot(time_stamps, np.array(seq_state_arc).astype(float), label='state_arc')
        plt.plot(time_stamps, np.array(seq_state_normal).astype(float), label='state_normal')
        plt.plot(time_stamps, np.array(seq_state_pred_arc).astype(float), label='state_arc_pred')
        plt.xlim(0, seq_len)
        plt.ylim(0, 4096)
        plt.legend()
        wavelet, scales, fs = 'shan', np.arange(1, 128), 256 * 50
        coefficients, frequencies = pywt.cwt(seq_power, scales, wavelet, 1 / fs)
        plt.subplot(self.wavelet_max_level + 1, 1, 2)
        plt.pcolormesh(time_stamps, frequencies, np.log1p(np.abs(coefficients)))
        plt.tight_layout()
        plt.title(save_name)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if show:
            plt.show()
            plt.pause(pause_time_s)
        if dir_save is not None:
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            fig = plt.gcf()
            fig.set_size_inches(screen_width / fig.dpi, screen_height / fig.dpi)
            plt.savefig(os.path.join(dir_save, save_name), dpi=fig.dpi)
        plt.close()

    def plot_emd(self, pause_time_s=1024, dir_save=None, save_name=None, show=True):
        plt.ion()
        key = 'rt'
        idx_s = 410000
        seq_power = self.db[key].seq_power[idx_s:idx_s + 2048]
        seq_len = len(seq_power)
        time_stamps = np.array(range(seq_len))
        emd = EMD()
        imfs = emd(np.array(seq_power).astype(float))
        plt.subplot(len(imfs) + 1, 1, 1)
        plt.plot(time_stamps, np.array(seq_power).astype(float), label='power')
        plt.xlim(0, seq_len)
        plt.ylim(0, 4096)
        plt.legend()
        for i, imf in enumerate(imfs):
            plt.subplot(len(imfs) + 1, 1, i + 2)
            plt.plot(time_stamps, imf, 'g')
            plt.title('imfs ' + str(i + 1))
            plt.ylim(-2048, 2048)
        plt.tight_layout()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if show:
            plt.show()
            plt.pause(pause_time_s)
        if dir_save is not None:
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            fig = plt.gcf()
            fig.set_size_inches(screen_width / fig.dpi, screen_height / fig.dpi)
            plt.savefig(os.path.join(dir_save, save_name), dpi=fig.dpi)
        plt.close()

    def reset(self):
        del self.db['rt']
        self.db['rt'] = self.Signals()


class DataPP(DataBase):
    @dataclass
    class Signals:
        seq_power: list = field(default_factory=list)
        seq_filtered: list = field(default_factory=list)
        seq_sub_sampled: list = field(default_factory=list)
        seq_len: int = 0

    def __init__(self):
        super().__init__()
        self.key = 'pp'
        self.db[self.key] = self.Signals()

    def update(self, cur_power):
        self.db[self.key].seq_power.append(cur_power)
        self.db[self.key].seq_filtered.append(0)
        self.db[self.key].seq_len += 1

    def sub_sample(self, step=8):
        self.db[self.key].seq_sub_sampled = self.db[self.key].seq_filtered[::step]

    def save_sub_sample(self, path):
        np.save(path, np.array(self.db[self.key].seq_sub_sampled).astype(float))

    def plot(self, pause_time_s=1024, dir_save=None, save_name=None, show=True):
        plt.ion()
        seq_power = self.db[self.key].seq_power
        seq_filtered = self.db[self.key].seq_filtered
        seq_sub_sampled = self.db[self.key].seq_sub_sampled
        seq_len = self.db[self.key].seq_len
        _sub_sample_rate = round(self.db[self.key].seq_len / len(seq_sub_sampled))
        time_stamps = np.array(range(seq_len))
        plt.plot(time_stamps, np.array(seq_power).astype(float), label='power')
        plt.plot(time_stamps, np.array(seq_filtered).astype(float), label='seq_filtered')
        time_stamps_sub = time_stamps[::_sub_sample_rate]
        plt.plot(time_stamps_sub, np.array(seq_sub_sampled).astype(float), label='time_stamps_sub')
        plt.xlim(0, seq_len)
        # plt.ylim(0, 4096)
        plt.legend()
        plt.tight_layout()
        plt.title(save_name)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if show:
            plt.show()
            plt.pause(pause_time_s)
        if dir_save is not None:
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            fig = plt.gcf()
            fig.set_size_inches(screen_width / fig.dpi, screen_height / fig.dpi)
            plt.savefig(os.path.join(dir_save, save_name), dpi=fig.dpi)
        plt.close()

    def reset(self):
        del self.db[self.key]
        self.db[self.key] = self.Signals()
