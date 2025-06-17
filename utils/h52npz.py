# h5_to_npz.py

import torch.nn.functional as F


class MyTransform:
    @staticmethod
    def __call__(x_sample):
        x_tensor = torch.tensor(x_sample, dtype=torch.float32)
        x_signal = (x_tensor - 2048) / 4096

        # →  (C=1, L)
        x_signal = x_signal.unsqueeze(0)

        # pad to multiple of 32
        L = x_signal.shape[-1]
        pad_right = (32 - L % 32) % 32
        x_signal = F.pad(x_signal, (0, pad_right), value=0)

        # → (C=1, L_pad) , 但校准时还需要 batch 维
        x_signal = x_signal.unsqueeze(0)  # 变成 (1, 1, L_pad)
        x_signal = x_signal.unsqueeze(0)  # 变成 (1, 1, 1, L_pad)
        return x_signal


# h5_to_calib_npz.py
import h5py, numpy as np, argparse, torch


def convert(h5_path, feat_ds, out_npz, input_name, max_samples=None):
    tfm = MyTransform()
    with h5py.File(h5_path, 'r') as f:
        feats = f[feat_ds]
        N = len(feats) if max_samples is None else min(len(feats), max_samples)
        samples = []
        for i in range(N):
            x = feats[i]  # ndarray
            x_t = tfm(x)  # torch.Size([1,1,L_pad])
            samples.append(x_t.numpy())  # 转 numpy
            if (i + 1) % 100 == 0:
                print(f'processed {i + 1}/{N}')
    arr = np.concatenate(samples, axis=0)  # (N,1,L_pad)
    print('final shape:', arr.shape, arr.dtype)
    np.savez(out_npz, **{input_name: arr})
    print('✓ saved', out_npz)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='/home/manu/tmp/test_data_subset.h5')
    ap.add_argument('--feat-ds', default='features',
                    help='dataset name inside h5')
    ap.add_argument('-o', '--out', default='/home/manu/tmp/calib.npz')
    ap.add_argument('--input-name', default='inputs',
                    help='must equal model input name')
    ap.add_argument('-n', '--max-samples', type=int, default=8,
                    help='optional, only take first N samples')
    args = ap.parse_args()
    convert(args.h5, args.feat_ds, args.out,
            args.input_name, args.max_samples)
