import ujson as json
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from typing import Optional, List, Dict, Callable, Tuple
import os

class SensorDataset(Dataset):
    def __init__(self, fname: str, seq_len: int, stride: int,
                 transform: Optional[Callable] = lambda x: x, removed_classes: Optional[List]|int = [], label_mapping: Optional[Dict] = {}):
        self.transform = transform
        if isinstance(removed_classes, int): removed_classes = [removed_classes]
        if len(removed_classes) > 0: assert len(label_mapping) > 0, "label_mapping cannot be empty if classes are removed."

        print(f'Loading {fname}...')
        self.data = []
        with open(fname, 'r') as f:
            #instances = f.readlines()
            for line in tqdm(f):
                instance = json.loads(line)
                if instance['y'] not in removed_classes:
                    y = label_mapping.get(instance['y'], instance['y'])
                    self.data += self.split_instance(instance['X'], y, seq_len, stride)
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

def get_dataloader(fname: str, seq_len: int, stride: int, batch_size: int, shuffle: bool, n_class: int, num_workers: Optional[int] = 0, 
                   removed_classes: Optional[List]|int = []) -> DataLoader:
    '''
    # 50 samples per second
    get_dataloader('tst.json', 5*50, 50, 32)
    '''
    if isinstance(removed_classes, int): removed_classes = [removed_classes]
    label_mapping, _ = class_relabel(n_class, removed_classes)
    dataset = SensorDataset(fname, seq_len, stride, SampleTransform(), removed_classes, label_mapping)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_train_dataloader(src_dir:str, trg_dir:str, seq_len: int, stride: int, batch_size: int, n_class: int, num_workers: Optional[int] = 0,
                         removed_classes: Optional[List]|int = []) -> DataLoader:
    if isinstance(removed_classes, int): removed_classes = [removed_classes]
    label_mapping, _ = class_relabel(n_class, removed_classes)
    src_dataset = SensorDataset(os.path.join(src_dir, "trn.json"), seq_len, stride, SampleTransform(), removed_classes, label_mapping)
    trg_dataset = SensorDataset(os.path.join(trg_dir, "trn.json"), seq_len, stride, SampleTransform(), removed_classes, label_mapping)
    max_dataset_sz = max(len(src_dataset), len(trg_dataset))
    src_dataloader = DataLoader(src_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=RandomSampler(src_dataset, replacement=True, num_samples=max_dataset_sz))
    trg_dataloader = DataLoader(trg_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=RandomSampler(trg_dataset, replacement=True, num_samples=max_dataset_sz))
    return src_dataloader, trg_dataloader

def class_relabel(n_class: int, removed_classes: List[int]) -> Tuple[Dict, int]:
    '''
    relabelling if several classes are removed.
    '''
    left_classes = set(range(n_class)) - set(removed_classes)
    return {c:i for i, c in enumerate(left_classes)}, len(left_classes)



