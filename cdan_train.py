import argparse
import os
import torch
from models import MultiScaleFCN, LinearClassifier, FeatureHead
from models import ConditionalDomainAdversial, ConditionalDomainDiscriminator
from visualize import visualize
from utils import feature_extract, get_dataloader
from utils import SensorDataset, SampleTransform, rm_mode_index
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args: argparse.Namespace):
    if args.seed is not None:
        pl.seed_everything(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision('medium')
    
    ## preparing dataset
    train_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'trn.json'), args.L, args.train_stride, SampleTransform())
    train_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'trn.json'), args.L, args.train_stride, SampleTransform())
    valid_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'val.json'), args.L, args.test_stride, SampleTransform())
    test_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())
    test_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())
    ## get dataset max size
    dataset_max_size = max(len(train_src_dataset), len(train_trg_dataset))
    ## get indices without mode 3(airplane)
    valid_trg_index = rm_mode_index(valid_trg_dataset, 3)
    test_trg_index = rm_mode_index(test_trg_dataset, 3)
    ## preparing data
    train_src_loader = DataLoader(train_src_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_trg_loader = DataLoader(train_trg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=RandomSampler(train_trg_dataset, replacement=True, num_samples=dataset_max_size))
    valid_trg_loader = DataLoader(valid_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=SubsetRandomSampler(valid_trg_index))
    test_src_loader = DataLoader(test_src_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8)
    test_trg_loader = DataLoader(test_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=SubsetRandomSampler(test_trg_index))
    test_trg_all_loader = DataLoader(test_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8)

    ## fcn
    backbone = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
    classifer = LinearClassifier(args.hidden_size*2, args.n_class)
    feature_head = FeatureHead(args.hidden_size*2, args.feature_dim)
    domainDiscriminator = ConditionalDomainDiscriminator(args.feature_dim*args.n_class, 256)
    cdan = ConditionalDomainAdversial(backbone, classifer, domainDiscriminator, feature_head, args.n_class, args.trade_off)

    ## training and validation
    logger = TensorBoardLogger('lightning_logs/', name=args.log_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices], max_epochs=args.max_epochs, gradient_clip_val=1.0, logger=logger, callbacks=[checkpoint_callback])
    iterables = {'src': train_src_loader, 'trg': train_trg_loader}
    train_dataloader = CombinedLoader(iterables, 'max_size_cycle')
    trainer.fit(cdan, train_dataloader, valid_trg_loader)

    ## testing with the best model
    trainer.test(cdan, [test_src_loader, test_trg_loader, test_trg_all_loader], ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--data_src_dir", default="data/beijing")
    parser.add_argument("--data_trg_dir", default="data/chongqing")
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--train_stride", default=2*50, type=int)
    parser.add_argument("--test_stride", default=2*50, type=int)
    parser.add_argument("--hidden_size", default=320, type=int)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--L", default=32*50, type=int) # seq_len or window size
    parser.add_argument("--N", default=16, type=int) # num_channel
    parser.add_argument("--n_class", default=.5, type=int)
    parser.add_argument("--trade_off", default=2., type=float)
    parser.add_argument("--seed", default=701, type=int)
    parser.add_argument("--log_name", default='cdan', type=str)
    args = parser.parse_args()
    main(args)