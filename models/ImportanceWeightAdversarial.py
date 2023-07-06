from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import copy
from alignment import DomainAdversarialLoss
from models import WarmStartGradientReverseLayer
from alignment import get_partial_classes_weight

class ImportanceWeightAdversarial(pl.LightningModule):
    def __init__(self, backbone, classifier, domain_adv_D, domain_adv_D0, n_class, trade_off, lamda, gamma):
        super(ImportanceWeightAdversarial, self).__init__()
        self.src_backbone = backbone
        self.trg_backbone = copy.deepcopy(backbone)
        for params in self.src_backbone.parameters():
            params.requires_grad = False
        for params in self.trg_backbone.parameters():
            params.requires_grad = True
        self.classifier = classifier
        self.domain_adv_D = domain_adv_D
        self.domain_adv_D0 = domain_adv_D0
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
        self.trade_off = trade_off
        self.lamda = lamda
        self.gamma = gamma
        self.discriminator_loss = DomainAdversarialLoss(self.domain_adv_D)
        self.domain_adv_D0_loss = DomainAdversarialLoss(self.domain_adv_D0, grl=WarmStartGradientReverseLayer(alpha=.5, lo=0., hi=3., max_iters=2000, auto_step=True))
    
    def training_step(self, batch, batch_idx):
        # get optimizer

        src_x, src_y = batch['src']
        trg_x, _ = batch['trg']

        # weight discriminator forward
        src_f = self.src_backbone(src_x)
        trg_f = self.trg_backbone(trg_x)
        w = torch.ones(src_x.shape[0]).to(self.device) - self.domain_adv_D(src_f).view(-1,)
        w = w / w.mean()
        w = w.detach()
        src_g = self.classifier(src_f)

        # loss
        loss_discriminator = self.discriminator_loss(src_f.detach(), trg_f.detach(), sigmoid=True)
        loss_adv = self.domain_adv_D0_loss(src_f.detach(), trg_f, w_s=w)
        loss_entropy = entropy(src_g)
        loss = loss_adv + self.gamma * loss_entropy + self.trade_off * loss_discriminator

        partial_class_weight, non_partial_classes_weight = \
            get_partial_classes_weight(w, src_y)
        self.log('partial_class_weight', partial_class_weight, prog_bar=True)
        self.log('non_partial_classes_weight', non_partial_classes_weight, prog_bar=True)
        # log   
        self.log('train_loss_adv', loss_adv, prog_bar=True)
        self.log('train_loss_entropy', loss_entropy, prog_bar=True)
        self.log('train_loss_discriminator', loss_discriminator, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.trg_backbone(x)
        g = self.classifier(f)
        loss = F.nll_loss(torch.log(g), y)
        accu = self.accuracy(g, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accu, on_epoch=True, prog_bar=True)
        return accu
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.RMSprop([
            {'params': self.domain_adv_D.parameters(), 'lr':1e-4},
            {'params': self.trg_backbone.parameters(), 'lr':1e-3},
            {'params': self.domain_adv_D0.parameters(), 'lr':1e-4},
        ])
        return optimizer

    
def entropy(p: torch.Tensor):
    entropy = -p * torch.log(p + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy.mean()
