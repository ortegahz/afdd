import glob
import logging
import os

import pandas as pd

from cores.preprocessor import Preprocessor
from data.data import DataV1
from utils.utils import make_dirs

from pathlib import Path


class PreprocessorWrapperBase:
    def __init__(self, addr, dir_save=None, path_label=None):
        self.addr = addr
        self.dir_save = dir_save
        self.pause_time_s = 1024
        self.plot_show = True
        make_dirs(dir_save, reset=True)
        df = pd.read_excel(path_label)
        self.labels_dict = df.set_index('文件名')[['采样率', '单位']].T.to_dict() if path_label is not None else None

    def run(self):
        raise NotImplementedError


class PreprocessorWrapperV0(PreprocessorWrapperBase):
    """
    format: case/<case>.csv
    """

    def __init__(self, addr, dir_save=None, path_label=None):
        super().__init__(addr, dir_save, path_label)
        self.preprocessor = Preprocessor()

    def _process_single(self, path_data, save_dir):
        case_name, _ = os.path.splitext(os.path.basename(path_data))
        _amp = 1000 * 1000 if self.labels_dict[case_name]['单位'] == 'MS/s' else 1000
        _sample_rate_org = float(self.labels_dict[case_name]['采样率']) * _amp
        self.preprocessor.sample_rate_org = _sample_rate_org
        self.preprocessor.reset()
        self.db_offline = DataV1(path_data)
        self.db_offline.load()
        db_offline_single = self.db_offline.db['default']
        for idx in range(db_offline_single.seq_len):
            # logging.info(f'processing {idx}th / {db_offline_single.seq_len} ... ')
            cur_power = db_offline_single.seq_adc[idx]
            cur_power_valid = float(cur_power) if '∞' not in str(cur_power) else self.preprocessor.seq_power_last_valid
            self.preprocessor.db.update(cur_power=cur_power_valid)
            self.preprocessor.process()
            self.preprocessor.seq_power_last_valid = cur_power_valid
            # if idx > 1024:
            #     break
        self.preprocessor.db.sub_sample(step=self.preprocessor.sub_sample_rate)
        if self.plot_show:
            self.preprocessor.db.plot(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
                                      save_name=f'{case_name}.png', show=self.plot_show)
        _save_path = os.path.join(save_dir, case_name + '.npy')
        self.preprocessor.db.save_sub_sample(_save_path)
        # self.preprocessor.reset()

    def run(self):
        self._process_single(self.addr, self.dir_save)


class PreprocessorWrapperV1(PreprocessorWrapperV0):
    """
    format: case/<cases>.csv
    """

    def __init__(self, addr, dir_save, path_label=None):
        super().__init__(addr, dir_save, path_label)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.preprocessor = Preprocessor()

    def run(self):
        cases_path = glob.glob(os.path.join(self.addr, '**', '*.csv'), recursive=True)
        for i, case_path in enumerate(cases_path):
            logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
            _dir_save, _ = os.path.split(case_path.replace(self.addr, self.dir_save))
            make_dirs(_dir_save, reset=False)
            self._process_single(case_path, _dir_save)
