import torch
import numpy as np
from torch.utils.data import Dataset
from config import PreTrainConfig

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
