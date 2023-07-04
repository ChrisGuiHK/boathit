import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from alignment import ConditionalDomainAdversialLoss
from typing import Any

class ConditionalDomainAdversial(pl.LightningModule):
    def __init__(self, backbone, classifer, discriminator, feature_head, n_class, trade_off, pretrained=False):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifer
        self.discriminator = discriminator
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
        self.trade_off = trade_off
        self.feature_head = feature_head
        self.loss = ConditionalDomainAdversialLoss(self.discriminator)
        self.pretrained = pretrained

    def training_step(self, batch, batch_idx):
        src_x, src_y = batch['src']
        trg_x, trg_y = batch['trg']

        # forward
        x = torch.cat((src_x, trg_x), dim=0)
        f = self.backbone(x)
        y = self.classifer(f)
        y_s, y_t = torch.chunk(y, 2, dim=0)
        f = self.feature_head(f)

        # loss
        loss_cls = F.nll_loss(y_s, src_y)
        loss_adv = self.loss(f, y)
        loss = loss_cls + self.trade_off * loss_adv

        # log   
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss_cls', loss_cls, prog_bar=True)
        self.log('train_loss_adv', loss_adv, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.backbone(x)
        g = self.classifer(f)
        loss = F.nll_loss(g, y)
        accu = self.accuracy(g, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accu, on_epoch=True, prog_bar=True)
        return accu
        

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        f = self.backbone(x)
        g = self.classifer(f)
        accu = self.accuracy(g, y)
        self.log('test_accuracy', accu, on_epoch=True, prog_bar=True)


    def configure_optimizers(self) -> Any:
        if not self.pretrained:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam([
                { 'params': self.backbone.parameters(), 'lr': 1e-5}, 
                { 'params': self.classifer.parameters(), 'lr': 1e-4},
                { 'params': self.feature_head.parameters(), 'lr': 1e-4},
                { 'params': self.feature_head.parameters(), 'lr': 1e-4}
            ])
        return optimizer