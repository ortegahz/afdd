import glob
import logging
import os

from cores.preprocessor import Preprocessor
from data.data import DataV1
from utils.utils import make_dirs


class PreprocessorWrapperBase:
    def __init__(self, addr, dir_save=None):
        self.addr = addr
        self.dir_save = dir_save
        self.pause_time_s = 1024
        self.plot_show = True
        make_dirs(dir_save, reset=True)

    def run(self):
        raise NotImplementedError


class PreprocessorWrapperV0(PreprocessorWrapperBase):
    """
    format: case/<case>.csv
    """

    def __init__(self, addr, dir_save=None):
        super().__init__(addr, dir_save)
        self.preprocessor = Preprocessor()

    def _process_single(self, path_data, save_dir):
        case_name, _ = os.path.basename(path_data).split('.')
        self.db_offline = DataV1(path_data)
        self.db_offline.load()
        db_offline_single = self.db_offline.db['default']
        for idx in range(db_offline_single.seq_len):
            logging.info(f'processing {idx}th / {db_offline_single.seq_len} ... ')
            cur_power = db_offline_single.seq_adc[idx]
            cur_power_valid = float(cur_power) if '∞' not in str(cur_power) else self.preprocessor.seq_power_last_valid
            self.preprocessor.db.update(cur_power=cur_power_valid)
            self.preprocessor.process()
            self.preprocessor.seq_power_last_valid = cur_power_valid
            # if idx > 1024 * 1024 / 2:
            #     break
        self.preprocessor.db.sub_sample(step=self.preprocessor.sub_sample_rate)
        if self.plot_show:
            self.preprocessor.db.plot(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
                                      save_name=f'{case_name}.png', show=self.plot_show)
        _save_path = os.path.join(save_dir, case_name + '.npy')
        self.preprocessor.db.save_sub_sample(_save_path)
        self.preprocessor.reset()

    def run(self):
        self._process_single(self.addr, self.dir_save)


class PreprocessorWrapperV1(PreprocessorWrapperV0):
    """
    format: case/<cases>.csv
    """

    def __init__(self, addr, dir_save):
        super().__init__(addr, dir_save)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.preprocessor = Preprocessor()

    def run(self):
        cases_path = glob.glob(os.path.join(self.addr, '*'))
        for i, case_path in enumerate(cases_path):
            logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
            self._process_single(case_path, self.dir_save)
