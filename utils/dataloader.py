import ujson as json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from tqdm import tqdm

class SensorDataset(Dataset):
    def __init__(self, fname: str, seq_len: int, stride: int, transform=lambda x: x):
        self.transform = transform
        print(f'Loading {fname}...')
        self.data = []
        with open(fname, 'r') as f:
            #instances = f.readlines()
            for line in tqdm(f):
                instance = json.loads(line)
                self.data += self.split_instance(instance['X'], instance['y'], seq_len, stride)
        print(f'There are {len(self.data)} data points.')

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def split_instance(X, y, seq_len: int, stride: int):
        '''
        X (n_channel, T)
        '''
        X = np.asarray(X)
        n = X.shape[1]
        split_index = list(range(seq_len, n, stride))
        if n <= seq_len or split_index[-1] != n: split_index = split_index + [n]
        return [(X[:, i-seq_len:i], y) for i in split_index] if n >= seq_len else [] #[(X, y)]

class SampleTransform(object):
    def __call__(self, sample):
        X, y = sample
        x0, x1 = X[0][0:-1], X[0][1:]
        X[0] = np.r_[0, x1-x0]
        #X = X[1:] # ignore time difference
        return (X.astype(np.float32), y)
    
def get_dataloader(fname: str, seq_len: int, stride: int, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    '''
    # 50 samples per second
    get_dataloader('tst.json', 5*50, 50, 32)
    '''
    dataset = SensorDataset(fname, seq_len, stride, SampleTransform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def rm_mode_index(dataset: Dataset, mode: int=3) -> List[int]:
    indices = []
    for idx in range(len(dataset)):
        if dataset[idx][1] != mode:
            indices.append(idx)
    return indices


