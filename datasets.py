import h5py
from paddle.io import Dataset
import paddle
import numpy as np


class TrainDataset(paddle.io.Dataset):

    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx]
            hr = f['hr'][idx]
            # lr = f['lr'][str(idx)]
            # hr = f['hr'][str(idx)]
            # print(paddle.shape(lr),paddle.shape(hr)) 
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(paddle.io.Dataset):

    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # lr = f['lr'][str(idx)].value
            # hr = f['hr'][str(idx)].value
            lr = f['lr'][str(idx)][()]
            hr = f['hr'][str(idx)][()]
            
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
