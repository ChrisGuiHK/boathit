from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import List, Optional
from alignment import DomainAdversarialLoss
from models.ImportanceWeightModule import get_partial_classes_weight, get_importance_weight, entropy


class ImportanceWeightAdversarial(pl.LightningModule):
    def __init__(self, backbone, classifier, domain_adv_D, domain_adv_D0, n_class, trade_off, gamma, partial_classes_index: Optional[List[int]]=None, pretrained: Optional[bool]=False):
        super(ImportanceWeightAdversarial, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.domain_adv_D = domain_adv_D
        self.domain_adv_D0 = domain_adv_D0
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
        self.discriminator_accu = torchmetrics.Accuracy(task="binary")
        self.trade_off = trade_off
        self.gamma = gamma
        self.domain_adv_D_loss = DomainAdversarialLoss(self.domain_adv_D, sigmoid=True)
        self.domain_adv_D0_loss = DomainAdversarialLoss(self.domain_adv_D0)
        self.partial_classes_index = partial_classes_index
        self.pretrained = pretrained
    
    def training_step(self, batch, batch_idx):
        # get optimizer

        src_x, src_y = batch['src']
        trg_x, _ = batch['trg']

        # weight discriminator forward
        src_f = self.backbone(src_x)
        trg_f = self.backbone(trg_x)
        src_g = self.classifier(src_f)
        trg_g = self.classifier(trg_f)

        # loss
        loss_cls = F.nll_loss(src_g, src_y)
        loss_adv_D = self.domain_adv_D_loss(src_f.detach(), trg_f.detach())
        with torch.no_grad():
            w_s = get_importance_weight(self.domain_adv_D, src_f)
        loss_adv_D0 = self.domain_adv_D0_loss(src_f, trg_f, w_s=w_s)
        loss_entropy = entropy(torch.exp(trg_g), reduction='mean')
        loss = loss_cls + 1.5 * self.trade_off * loss_adv_D + self.trade_off * loss_adv_D0 + self.gamma * loss_entropy
        partial_class_weight, non_partial_classes_weight = \
            get_partial_classes_weight(w_s, src_y, self.partial_classes_index)
        
        f = torch.cat([src_f, trg_f], dim=0)
        discriminator_predict = self.domain_adv_D(f).squeeze()
        labels = torch.cat([torch.ones(src_g.shape[0], dtype=torch.long), torch.zeros(trg_g.shape[0], dtype=torch.long)], dim=0).to(discriminator_predict.device)
        
        # log  
        self.log('partial_class_weight', partial_class_weight, prog_bar=True)
        self.log('non_partial_classes_weight', non_partial_classes_weight, prog_bar=True)
        self.log('discriminator_accu', self.discriminator_accu(discriminator_predict, labels), prog_bar=True)

        self.log('train_loss_cls', loss_cls, prog_bar=True)
        self.log('train_loss_adv_D0', loss_adv_D0, prog_bar=True)
        self.log('train_loss_entropy', loss_entropy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.backbone(x)
        g = self.classifier(f)
        loss = F.nll_loss(g, y)
        accu = self.accuracy(g, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accu, on_epoch=True, prog_bar=True)
        return accu
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        f = self.backbone(x)
        g = self.classifier(f)
        accu = self.accuracy(g, y)
        self.log('test_accuracy', accu, on_epoch=True, prog_bar=True)

    
    def configure_optimizers(self) -> Any:
        if self.pretrained:
            optimizer = torch.optim.RMSprop([
                {'params': self.domain_adv_D.parameters(), 'lr':1e-3},
                {'params': self.domain_adv_D0.parameters(), 'lr':1e-3},
                {'params': self.classifier.parameters(), 'lr':1e-5},
                {'params': self.backbone.parameters(), 'lr':1e-5},
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': self.domain_adv_D.parameters(), 'lr':1e-3},
                {'params': self.domain_adv_D0.parameters(), 'lr':1e-3},
                {'params': self.classifier.parameters(), 'lr':1e-3},
                {'params': self.backbone.parameters(), 'lr':1e-3},
            ])
        return optimizer

