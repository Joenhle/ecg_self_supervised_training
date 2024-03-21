import torch
import numpy as np
from torch.utils.data import Dataset
from config import PreTrainConfig
from tsaug import Quantize, AddNoise, Convolve, Pool, Drift, Dropout, Reverse

def signal_handler(sig, frame):
    pass

class PretrainDataset(Dataset):
    name = 'physionet'

    def __init__(self, index_path):
        index_arr = []
        for line in open(index_path):
            temp = line.strip().split()
            index_arr.append(temp)
        self.index_arr = index_arr
    
    def __len__(self):
        return len(self.index_arr)

    def __getitem__(self, index):
        data = self.index_arr[index]
        x = np.load(data[0]).astype(np.float32) 
        if PreTrainConfig.data_standardization and np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
        x = np.squeeze(x)
        x = torch.tensor(x, dtype=torch.float32)
        #todo 这里是否需要重采样?
        return x.unsqueeze(0)

class PreTrainContrastDataset(Dataset):

    name = 'physionet_contrast'

    class DataAugumentor(object):
        def __init__(self):
            self.transform = (
                Quantize(n_levels=10) @ 0.5
                + AddNoise(scale=0.3) @ 0.5
                + Convolve(size=11) @ 0.5
                + Pool(size=4) @ 0.5
                + Drift(max_drift=0.7, n_drift_points=5) @ 0.5
                + Dropout(p=0.9, size=(1, 10), per_channel=True) @ 0.5
                + Reverse() @ 0.5
            )
            
        def __call__(self, x):
            return self.transform.augment(x)
    
    def __init__(self, index_path, base_transform = DataAugumentor(), n_view = 2):
        index_arr = []
        for line in open(index_path):
            temp = line.strip().split()
            index_arr.append(temp)
        self.index_arr = index_arr
        self.base_transform = base_transform
        self.n_view = n_view
    
    def __len__(self):
        return len(self.index_arr)

    def __getitem__(self, index):
        data = self.index_arr[index]
        x = np.load(data[0]).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32)
        x = [torch.from_numpy(self.base_transform(x.numpy())) for _ in range(self.n_view)]
        return x