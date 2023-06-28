import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
import ujson as json
from tqdm import tqdm
from pandas import Series
from typing import List, Set, Tuple, Union, Optional
from toolz import partial
from argparse import ArgumentParser

np.random.seed(0)
#  python utils/dataset_preparation.py --raw_data_dir /data/xiucheng/oppo-transport/harbin --data_dir data_harbin
parser = ArgumentParser()
parser.add_argument("--raw_data_dir", default='/home/guihaokun/boathit/data_raw/oppo-transport/beijing')
parser.add_argument("--data_dir", default='/home/guihaokun/boathit/data/beijing')
args = parser.parse_args()

columns = [
    'mode', 'seq_id', 
    'time', 
    #'light',
    'acc_X', 'acc_Y', 'acc_Z', 
    'gravity_X', 'gravity_Y', 'gravity_Z',
    'lin_acc_X', 'lin_acc_Y', 'lin_acc_Z',
    'gyro_X', 'gyro_Y', 'gyro_Z',
    'mag_X', 'mag_Y', 'mag_Z',
    #'rot_vec_X', 'rot_vec_Y', 'rot_vec_Z',
]

data_columns = columns[2:]

def get_mode_id(path: Union[str, List[str]], usecols: List[str]) -> Series:
    '''
    Count the number of instances in each mode.
    '''
    df = dd.read_csv(path, usecols=usecols, dtype={'seq_id': 'str', 'mode': 'int32'})
    mode_id = df.groupby(['mode']).apply(lambda x: x['seq_id'].unique(), meta=pd.Series(dtype='str'))
    return mode_id.compute()

def split_list(mode_id: Series, ratio = [0.5, 0.3]) -> Tuple[List]:
    '''
    Split each mode with ratio into train/validation/test sets.
    '''
    trn, val, tst = [], [], []
    for mode in mode_id.index:
        ids = mode_id[mode]
        np.random.shuffle(ids)
        ids = list(ids)
        n = len(ids)
        n_trn, n_val = int(np.floor(n*ratio[0])), int(np.floor(n*ratio[1]))
        trn = trn + ids[0:n_trn]
        val = val + ids[n_trn:n_trn+n_val]
        tst = tst + ids[n_trn+n_val:]

    return trn, val, tst

def save_datasets(read_path: str, write_path: str, ds: Tuple[Set], usecols=columns, data_columns=data_columns) -> None:
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    for name in ['trn.json', 'val.json', 'tst.json']:
        if os.path.exists(os.path.join(write_path, name)):
            os.remove(os.path.join(write_path, name))
    
    def save_json(sdf, name):
        with open(os.path.join(write_path, name), 'a') as f:
            X = sdf[data_columns].dropna().to_numpy().T.tolist()
            y = int(sdf['mode'].iloc[0])
            f.write(json.dumps({'X': X, 'y': y}) + '\n')

    def iterate_csv_f(trn, val, tst, fname):
        df = pd.read_csv(fname, usecols=usecols, dtype={'seq_id': 'str', 'mode': 'int32', 'time': 'float64'})
        df.set_index('seq_id', inplace=True)
        print(f'processing {fname}...')
        for seq_id in tqdm(df.index.unique()):
            sdf = df.loc[seq_id].sort_values(by=['time'])           
            if seq_id in trn:
                save_json(sdf, 'trn.json')
            elif seq_id in val:
                save_json(sdf, 'val.json')
            elif seq_id in tst:
                save_json(sdf, 'tst.json')
    
    csv_fs = filter(lambda x: x.endswith('.csv'), os.listdir(read_path))
    csv_fs = map(lambda x: os.path.join(read_path, x), csv_fs)
    for name in csv_fs:
        iterate_csv_f(*ds, name)

## deprecated!
def save_datasets_slow(read_path: str, write_path: str, ds: Tuple[List], usecols=columns, data_columns=data_columns) -> None:
    '''
    save_datasets_slow(os.path.join(data_dir, 'beijing', '*.csv'), save_dir, (trn, val, tst))
    '''
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    
    trn, val, tst = ds
    df = dd.read_csv(read_path, usecols=usecols, dtype={'seq_id': 'str', 'mode': 'int32', 'time': 'float64'})
    df = df.set_index('seq_id')

    def save_json(ids: List, name: str):
        with open(os.path.join(write_path, name), 'w') as f:
            print(f'saving {name}...')
            f.write('[\n')
            for seq_id in tqdm(ids):
                sdf = df.loc[seq_id].compute().sort_values(by=['time'])
                X = sdf[data_columns].to_numpy().tolist()
                y = int(sdf['mode'].iloc[0])
                f.write(json.dumps({'X': X, 'y': y}) + '\n')
            f.write(']')
    
    save_json(trn, 'trn.json')
    save_json(val, 'val.json')
    save_json(tst, 'tst.json')



if __name__ == '__main__':
    mode_id = get_mode_id(os.path.join(args.raw_data_dir, '*.csv'), ['mode', 'seq_id'])
    print(mode_id)
    trn, val, tst = split_list(mode_id, [0.7, 0.2])
    print(f'#train:{len(trn)}, #validation: {len(val)}, #test: {len(tst)}.')
    print(f'trn: {trn}')
    print(f'val: {val}')
    print(f'tst: {tst}')
    save_datasets(args.raw_data_dir, args.data_dir, (set(trn), set(val), set(tst)))
