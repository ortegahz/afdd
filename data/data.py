import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import pywt

from utils.macros import *
from utils.utils import load_bin


class Signals:
    def __init__(self):
        self.sps = 256 * 50
        self.seq_power = None
        self.seq_hf = None
        self.len = -1
        self.seq_state_arc = None
        self.seq_state_normal = None


class DataBase:
    def __init__(self):
        self.db = dict()
        self.seq_len = 0


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
                self.db[key].seq_state_arc[int(alarm_interval_0[0]):int(alarm_interval_0[1])] = STATE_INDICATE_VAL
            if normal_interval_0 is not None:
                self.db[key].seq_state_normal[int(normal_interval_0[0]):int(normal_interval_0[1])] = \
                    STATE_INDICATE_VAL / 2
            for i in range(1, len(df.columns) - max(alarm_index, normal_index)):
                next_alarm_col = alarm_index + i
                next_normal_col = normal_index + i
                if next_alarm_col < len(df.columns) and pd.notna(row[next_alarm_col]):
                    next_alarm_interval = row[next_alarm_col].split('~')
                    self.db[key].seq_state_arc[int(next_alarm_interval[0]):int(next_alarm_interval[1])] = \
                        STATE_INDICATE_VAL
                if next_normal_col < len(df.columns) and pd.notna(row[next_normal_col]):
                    next_normal_interval = row[next_normal_col].split('~')
                    self.db[key].seq_state_normal[int(next_normal_interval[0]):int(next_normal_interval[1])] = \
                        STATE_INDICATE_VAL / 2

    def plot(self, idx_pick=0, pause_time_s=1024, dir_save=None, show=True):
        plt.ion()
        for idx in range(len(self.db.keys())):
            if idx_pick is not None and idx != idx_pick:
                continue
            key = list(self.db.keys())[idx]
            seq_power = self.db[key].seq_power
            seq_hf = self.db[key].seq_hf
            seq_len = self.db[key].len
            seq_state_arc = self.db[key].seq_state_arc
            seq_state_normal = self.db[key].seq_state_normal
            time_stamps = np.array(range(seq_len))
            fig, (ax1, ax2) = plt.subplots(2, 1)
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
            wavelet = 'sym20'
            logging.info(pywt.wavelist())
            max_level = pywt.dwt_max_level(seq_len, pywt.Wavelet(wavelet).dec_len)
            coeffs = pywt.wavedec(np.array(seq_power).astype(float), wavelet, level=max_level)
            level = max_level
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
