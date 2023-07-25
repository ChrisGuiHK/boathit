import torch
import pytorch_lightning as pl
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from models import MultiScaleFCN, LinearClassifier
from models import LitTSVanilla as LitTSClassifier
from utils import class_relabel, get_dataloader, feature_extract
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from visualize import visualize

def main(args: argparse.Namespace):
    ## setting for RTX 3090
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## dataset
    label_mapping, n_class = class_relabel(args.n_class, args.removed_classes)
    test_src_dataloader = get_dataloader(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=args.removed_classes)
    test_trg_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=[*args.removed_classes, 3])

    ## fcn
    mF = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
    mG = LinearClassifier(args.hidden_size*2, n_class)

    if args.mode == "train":
        train_dataloader = get_dataloader(os.path.join(args.data_src_dir, 'trn.json'), args.L, args.test_stride, args.batch_size, shuffle=True,
                                        label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=args.removed_classes)
        valid_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'val.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False, 
                                    label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=[*args.removed_classes, 3])
        
        tsc = LitTSClassifier(mF, mG, n_class, args.hidden_size*2, args.devices)
        ## training and validation
        logger = TensorBoardLogger('lightning_logs/', name=args.log_name)
        checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=-1, save_last=True, every_n_epochs=5)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices], max_epochs=args.max_epochs, gradient_clip_val=1.0, logger=logger, callbacks=[checkpoint_callback])
        trainer.fit(tsc, train_dataloader, valid_dataloader)

        ## testing with the best model
        trainer.test(tsc, [test_src_dataloader, test_trg_dataloader], ckpt_path='best')

    elif args.mode == "analysis":
        tsc = LitTSClassifier.load_from_checkpoint(
            checkpoint_path=f'lightning_logs/{args.log_name}/version_{args.version}/checkpoints/{args.best_ckpt}',
            hparams_file=f"lightning_logs/{args.log_name}/version_{args.version}/hparams.yaml",
            map_location=None,
            mF=mF,
            mG=mG,
            n_class=n_class,
            feature_dim=args.hidden_size*2,
            device=args.devices,
        )
        src_features, src_labels = feature_extract(tsc.mF, test_src_dataloader, args.devices)
        trg_features, trg_labels = feature_extract(tsc.mF, test_trg_dataloader, args.devices)
        visualize(src_features, trg_features, src_labels, trg_labels, os.path.join('fig', f'{args.fig_name}_tsne.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--data_src_dir", default="data/beijing")
    parser.add_argument("--data_trg_dir", default="data/chongqing")
    parser.add_argument("--max_epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--train_stride", default=2*50, type=int)
    parser.add_argument("--test_stride", default=2*50, type=int)
    parser.add_argument("--hidden_size", default=320, type=int)
    parser.add_argument("--L", default=32*50, type=int) # seq_len or window size
    parser.add_argument("--N", default=16, type=int) # num_channel
    parser.add_argument("--n_class", default=5, type=int)
    parser.add_argument("--seed", default=701, type=int)
    parser.add_argument("--log_name", default='vanilla', type=str)
    parser.add_argument("--removed_classes", default=[], choices=[0, 1, 2, 3, 4], nargs='*', type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--version", default=0, type=int)
    parser.add_argument("--best_ckpt", default=None, type=str)
    parser.add_argument("--fig_name", default="vanilla", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    args = parser.parse_args()
    main(args)



