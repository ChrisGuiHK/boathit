import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os, torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models import FCNNaive, GatedConv, LinearClassifier
from models import LitTSVanilla as LitTSClassifier
from utils import get_dataloader
from typing import Any
from argparse import ArgumentParser

## setting for RTX 3090
torch.set_float32_matmul_precision('medium')
    
parser = ArgumentParser()
parser.add_argument("--accelerator", default='gpu')
parser.add_argument("--devices", default=0, type=int)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--max_epochs", default=80, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--train_stride", default=2*50, type=int)
parser.add_argument("--test_stride", default=2*50, type=int)
parser.add_argument("--hidden_size", default=320, type=int)
parser.add_argument("--L", default=32*50, type=int) # seq_len or window size
parser.add_argument("--N", default=12, type=int) # num_channel
parser.add_argument("--n_class", default=5, type=int)
args = parser.parse_args()

mF = FCNNaive((args.N, args.L), args.n_class, hidden_size=args.hidden_size, kernel_sizes=[5, 3, 3])
mG = LinearClassifier(args.hidden_size*2, args.n_class)
#tsc = LitTSClassfier(fcn, args.n_class)
## training and validation
tsc = LitTSClassifier.load_from_checkpoint(
    checkpoint_path='lightning_logs/version_0/checkpoints/epoch=38-step=94848.ckpt',
    hparams_file="lightning_logs/version_0/hparams.yaml",
    map_location=None,
    mF=mF,
    mG=mG,
    n_class=args.n_class
)
trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices])


## testing with the best model
test_loader = get_dataloader(os.path.join(args.data_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, False, 8)
trainer.test(tsc, test_loader)