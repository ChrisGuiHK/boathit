import argparse
import os
import torch
from models import MultiScaleFCN, LinearClassifier
from models import DomainAdversial, DomainDiscriminator
from utils import feature_extract, get_dataloader
from utils import SensorDataset, SampleTransform, rm_mode_index
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from visualize import visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

def main(args: argparse.Namespace):
    backbone = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
    classifer = LinearClassifier(args.hidden_size*2, args.n_class)
    domainDiscriminator = DomainDiscriminator(args.hidden_size*2, 512, 1024)
    dann = DomainAdversial.load_from_checkpoint(
        checkpoint_path='lightning_logs/dann_2classes/version_0/checkpoints/epoch=18-step=6479.ckpt',
        hparams_file="lightning_logs/dann_2classes/version_0/hparams.yaml",
        map_location=None,
        backbone=backbone, 
        classifer=classifer, 
        discriminator=domainDiscriminator, 
        n_class=args.n_class, 
        trade_off=args.trade_off
    )
    test_src_dataset = SensorDataset(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())
    test_trg_dataset = SensorDataset(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, SampleTransform())

    test_src_index = rm_mode_index(test_src_dataset, [2, 3, 4])
    test_trg_index = rm_mode_index(test_trg_dataset, 3)

    src_loader = DataLoader(test_src_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=SubsetRandomSampler(test_src_index))
    trg_loader = DataLoader(test_trg_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=8, sampler=SubsetRandomSampler(test_trg_index))

    src_features, src_labels = feature_extract(dann.backbone, src_loader)
    trg_features, trg_labels = feature_extract(dann.backbone, trg_loader)
    visualize(src_features, trg_features, src_labels, trg_labels, os.path.join(args.visualize_dir, 'dann_2classes_tsne.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=0, type=int)
    parser.add_argument("--data_src_dir", default="data/beijing")
    parser.add_argument("--data_trg_dir", default="data/chongqing")
    parser.add_argument("--visualize_dir", default="fig")
    parser.add_argument("--max_epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--train_stride", default=2*50, type=int)
    parser.add_argument("--test_stride", default=2*50, type=int)
    parser.add_argument("--hidden_size", default=320, type=int)
    parser.add_argument("--L", default=32*50, type=int) # seq_len or window size
    parser.add_argument("--N", default=16, type=int) # num_channel
    parser.add_argument("--n_class", default=5, type=int)
    parser.add_argument("--trade_off", default=1.0, type=float)
    parser.add_argument("--seed", default=701, type=int)
    parser.add_argument("--log_name", default='dann', type=str)
    args = parser.parse_args()
    main(args)