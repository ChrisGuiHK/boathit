import argparse
import os
import torch
from models import MultiScaleFCN, LinearClassifier
from models import DomainAdversial, DomainDiscriminator
from visualize import visualize
from utils import feature_extract, get_dataloader, get_train_dataloader, class_relabel
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
    _, n_class = class_relabel(args.n_class, args.removed_classes)
    test_src_dataloader = get_dataloader(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        n_class=args.n_class, num_workers=args.num_workers, removed_classes=args.removed_classes)
    test_trg_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        n_class=args.n_class, num_workers=args.num_workers, removed_classes=args.removed_classes)

    ## fcn
    backbone = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
    classifer = LinearClassifier(args.hidden_size*2, n_class)
    domainDiscriminator = DomainDiscriminator(args.hidden_size*2, 512, 1024)

    if args.mode == "train":
        train_src_dataloader, train_trg_dataloader = get_train_dataloader(args.data_src_dir, args.data_trg_dir, args.L, args.train_stride, args.batch_size, args.n_class, args.num_workers, args.removed_classes)
        valid_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'val.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False, 
                                    n_class=args.n_class, num_workers=args.num_workers, removed_classes=args.removed_classes)
        
        dann = DomainAdversial(backbone, classifer, domainDiscriminator, n_class, args.trade_off)

        ## training and validation
        logger = TensorBoardLogger('lightning_logs/', name=args.log_name)
        checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices], max_epochs=args.max_epochs, gradient_clip_val=1.0, logger=logger, callbacks=[checkpoint_callback])
        iterables = {'src': train_src_dataloader, 'trg': train_trg_dataloader}
        train_dataloader = CombinedLoader(iterables, 'max_size_cycle')
        trainer.fit(dann, train_dataloader, valid_dataloader)

        ## testing with the best model
        trainer.test(dann, [test_src_dataloader, test_trg_dataloader], ckpt_path='best')
        
    elif args.mode == "analysis":
        if args.best_ckpt is None:
            raise ValueError("Please specify the best checkpoint path")
        
        dann = DomainAdversial.load_from_checkpoint(
            checkpoint_path=f'lightning_logs/{args.log_name}/version_{args.version}/checkpoints/{args.best_ckpt}',
            hparams_file=f"lightning_logs/{args.log_name}/version_{args.version}/hparams.yaml",
            map_location=None,
            backbone=backbone, 
            classifer=classifer, 
            discriminator=domainDiscriminator, 
            n_class=n_class, 
            trade_off=args.trade_off
        )
        src_features, src_labels = feature_extract(dann.backbone, test_src_dataloader)
        trg_features, trg_labels = feature_extract(dann.backbone, test_trg_dataloader)
        visualize(src_features, trg_features, src_labels, trg_labels, os.path.join('fig', f'{args.fig_name}_tsne.png'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--trade_off", default=1., type=float)
    parser.add_argument("--seed", default=701, type=int)
    parser.add_argument("--log_name", default='dann', type=str)
    parser.add_argument("--removed_classes", default=[], choices=[0, 1, 2, 3, 4], nargs='*', type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--version", default=0, type=int)
    parser.add_argument("--best_ckpt", default=None, type=str)
    parser.add_argument("--fig_name", default="dann", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    args = parser.parse_args()
    main(args)