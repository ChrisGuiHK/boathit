import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from models import FCNNaive, GatedConv, MultiScaleFCN, LinearClassifier
from models import LitTSVanilla as LitTSClassifier
from utils import rm_mode_index, SensorDataset, SampleTransform
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

## setting for RTX 3090
torch.set_float32_matmul_precision('medium')

parser = ArgumentParser()
parser.add_argument("--accelerator", default='gpu')
parser.add_argument("--devices", default=1, type=int)
parser.add_argument("--data_src_dir", default="data/beijing")
parser.add_argument("--data_trg_dir", default="data/chongqing")
parser.add_argument("--max_epochs", default=80, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--train_stride", default=2*50, type=int)
parser.add_argument("--test_stride", default=2*50, type=int)
parser.add_argument("--hidden_size", default=320, type=int)
parser.add_argument("--L", default=32*50, type=int) # seq_len or window size
parser.add_argument("--N", default=16, type=int) # num_channel
parser.add_argument("--n_class", default=5, type=int)
parser.add_argument("--seed", default=701, type=int)
parser.add_argument("--log_name", default='vanilla', type=str)
args = parser.parse_args()

pl.seed_everything(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## preparing dataset
train_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'trn.json'), args.L, args.train_stride, SampleTransform())
valid_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'val.json'), args.L, args.test_stride, SampleTransform())
valid_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'val.json'), args.L, args.test_stride, SampleTransform())
## get indices without mode 3(airplane)
valid_src_index = rm_mode_index(valid_src_dataset, 3)
valid_trg_index = rm_mode_index(valid_trg_dataset, 3)
## preparing data
train_src_loader = DataLoader(train_src_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
valid_src_loader = DataLoader(valid_src_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=torch.utils.data.SubsetRandomSampler(valid_src_index))
valid_trg_loader = DataLoader(valid_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=torch.utils.data.SubsetRandomSampler(valid_trg_index))

## fcn
mF = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
mG = LinearClassifier(args.hidden_size*2, args.n_class)
tsc = LitTSClassifier(mF, mG, args.n_class)
## training and validation
logger = TensorBoardLogger('lightning_logs/', name=args.log_name)
checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True)
trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices], max_epochs=args.max_epochs, gradient_clip_val=1.0, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(tsc, train_src_loader, valid_trg_loader)

## testing with the best model
test_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())
test_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())
## get indices without mode 3(airplane)
test_src_index = rm_mode_index(test_src_dataset, 3)
test_trg_index = rm_mode_index(test_trg_dataset, 3)
test_src_loader = DataLoader(test_src_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=torch.utils.data.SubsetRandomSampler(test_src_index))
test_trg_loader = DataLoader(test_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=torch.utils.data.SubsetRandomSampler(test_trg_index))
trainer.test(tsc, [test_src_loader, test_trg_loader], ckpt_path='best')





