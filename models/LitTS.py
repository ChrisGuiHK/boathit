import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Any

class LitTSVanilla(pl.LightningModule):
    def __init__(self, mF, mG, n_class):
        super().__init__()
        self.mF = mF
        self.mG = mG
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)

    def training_step(self, batch, batch_idx):
        x, y = batch
        f = self.mF(x)
        g = self.mG(f)
        loss = F.nll_loss(g, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.mF(x)
        g = self.mG(f)
        loss = F.nll_loss(g, y)
        accu = self.accuracy(g, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accu, on_epoch=True, prog_bar=True)
        return accu
        

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        f = self.mF(x)
        g = self.mG(f)
        accu = self.accuracy(g, y)
        self.log('test_accuracy', accu, on_epoch=True, prog_bar=True)


    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
