import argparse
import logging

import h5py
from tqdm import tqdm

from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CSV to HDF5 format.')
    parser.add_argument('--csv_path', default='/home/manu/tmp/afd_pm')
    parser.add_argument('--hdf5_path', default='/home/manu/tmp/afd_pm_h5py')
    return parser.parse_args()


def csv_to_hdf5(csv_file_path, hdf5_file_path):
    logging.info(f'Converting {csv_file_path} to {hdf5_file_path}...')

    with open(csv_file_path, 'r') as csv_file:
        lines = csv_file.readlines()

    num_samples = len(lines)
    num_features = len(lines[0].strip().split(',')) - 1

    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        labels = hdf5_file.create_dataset('labels', (num_samples,), dtype='i')
        features = hdf5_file.create_dataset('features', (num_samples, num_features), dtype='f')

        for i, line in enumerate(tqdm(lines, desc="Processing lines")):
            values = line.strip().split(',')
            labels[i] = 0 if float(values[0]) == -1 else 1
            features[i, :] = list(map(float, values[1:]))

    logging.info(f'Finished converting {num_samples} samples with {num_features} features each.')


def main():
    set_logging()
    args = parse_args()
    csv_to_hdf5(args.csv_path, args.hdf5_path)


if __name__ == '__main__':
    main()
