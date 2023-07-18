import argparse
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import feature_extract, get_dataloader, get_train_dataloader, class_relabel
from models import MultiScaleFCN, LinearClassifier
from models import LitTSVanilla as LitTSClassifier
from models import ImportanceWeightAdversarial, DomainDiscriminator
from visualize import visualize

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args: argparse.Namespace):
    if args.seed is not None:
        pl.seed_everything(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision('medium')

    ## preparing dataset
    label_mapping, n_class = class_relabel(args.n_class, args.removed_classes)
    test_src_dataloader = get_dataloader(os.path.join(args.data_src_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=args.removed_classes)
    test_trg_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'tst.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False,
                                        label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=[*args.removed_classes, 3])
    
    ## preparing model
    if args.pretrained:
    
        tsc = LitTSClassifier.load_from_checkpoint(
            checkpoint_path=f'lightning_logs/{args.pretrained_model}/version_{args.pretrained_version}/checkpoints/{args.pretrained_ckpt}',
            hparams_file=f"lightning_logs/{args.pretrained_model}/version_{args.pretrained_version}/hparams.yaml",
            map_location=None,
            mF=MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11]),
            mG=LinearClassifier(args.hidden_size*2, n_class),
            n_class=n_class
        )
        backbone = tsc.mF
        classifier = tsc.mG
    else:
        backbone = MultiScaleFCN((args.N, args.L), hidden_size=args.hidden_size, kernel_sizes=[1, 3, 5, 7, 11])
        classifier = LinearClassifier(args.hidden_size*2, n_class)

    domain_adv_D = DomainDiscriminator(args.hidden_size*2, [512], output_dim=1)
    domain_adv_D0 = DomainDiscriminator(args.hidden_size*2, [512, 512])
    
    if args.mode == "train":
        train_src_dataloader, train_trg_dataloader = get_train_dataloader(args.data_src_dir, args.data_trg_dir, args.L, args.train_stride, args.batch_size, label_mapping, args.num_workers, args.removed_classes)
        valid_dataloader = get_dataloader(os.path.join(args.data_trg_dir, 'val.json'), args.L, args.test_stride, 2*args.batch_size, shuffle=False, 
                                    label_mapping=label_mapping, num_workers=args.num_workers, removed_classes=[*args.removed_classes, 3])
        iwan = ImportanceWeightAdversarial(backbone, classifier, domain_adv_D, domain_adv_D0, n_class, args.trade_off, args.gamma, list(filter(lambda x: x not in [2, 3, 4], range(args.n_class))), pretrained=args.pretrained)

        # logger
        logger = TensorBoardLogger('lightning_logs/', name=args.log_name)
        checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.devices], max_epochs=args.max_epochs, gradient_clip_val=1.0, logger=logger, callbacks=[checkpoint_callback])
        iterables = {'src': train_src_dataloader, 'trg': train_trg_dataloader}
        train_dataloader = CombinedLoader(iterables, 'max_size_cycle')
        trainer.fit(iwan, train_dataloader, valid_dataloader)

        ## testing with the best model
        trainer.test(iwan, [test_src_dataloader, test_trg_dataloader], ckpt_path='best')
    
    elif args.mode == "analysis":
        if args.best_ckpt is None:
            raise ValueError("Please specify the best checkpoint path")
        
        iwan = ImportanceWeightAdversarial.load_from_checkpoint(
            checkpoint_path=f'lightning_logs/{args.log_name}/version_{args.version}/checkpoints/{args.best_ckpt}',
            hparams_file=f"lightning_logs/{args.log_name}/version_{args.version}/hparams.yaml",
            map_location=None,
            backbone=backbone, 
            classifer=classifier, 
            domain_adv_D=domain_adv_D,
            domain_adv_D0=domain_adv_D0,
            n_class=n_class, 
            trade_off=args.trade_off,
            gamma=args.gamma,
            partial_classes_index=list(filter(lambda x: x not in [2, 3, 4], range(args.n_class)))
        )
        src_features, src_labels = feature_extract(iwan.backbone, test_src_dataloader, args.devices)
        trg_features, trg_labels = feature_extract(iwan.backbone, test_trg_dataloader, args.devices)
        visualize(src_features, trg_features, src_labels, trg_labels, os.path.join('fig', f'{args.fig_name}_tsne.png'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--trade_off", default=.5, type=float)
    parser.add_argument("--lamda", default=1., type=float)
    parser.add_argument("--gamma", default=.005, type=float)
    parser.add_argument("--seed", default=701, type=int)
    parser.add_argument("--log_name", default='iwan', type=str)
    parser.add_argument("--removed_classes", default=[], choices=[0, 1, 2, 3, 4], nargs='*', type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--version", default=0, type=int)
    parser.add_argument("--best_ckpt", default=None, type=str)
    parser.add_argument("--fig_name", default="iwan", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pretrained", default=True, action='store_true')
    parser.add_argument("--pretrained_model", default="vanilla", type=str)
    parser.add_argument("--pretrained_ckpt", default="last.ckpt", type=str)
    parser.add_argument("--pretrained_version", default=0, type=int)
    args = parser.parse_args()
    main(args)