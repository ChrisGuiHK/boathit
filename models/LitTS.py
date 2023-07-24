import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Any
from models.ImportanceWeightModule import entropy

class LitTSVanilla(pl.LightningModule):
    def __init__(self, mF, mG, n_class):
        super().__init__()
        self.mF = mF
        self.mG = mG
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)

    def training_step(self, batch, batch_idx):
        tau = 0.5

        x, y = batch
        x_l, x_r = torch.chunk(x, 2, dim=2) # [batch_size, Channel, Length // 2]

        N = x_l.shape[0]
        out_left, out_right = self.mF(x_l), self.mF(x_r) # [batch_size, feature]
        out = torch.cat([out_left, out_right], dim=0) # [2 * batch_size, feature]

        # calculate sim matrix
        norm = out / torch.linalg.norm(out, dim=1, keepdim=True)
        sim_matrix = torch.mm(norm, norm.T) # [2 * batch_size, 2 * batch_size]

        exponential = torch.exp(sim_matrix / tau)
    
        # This binary mask zeros out terms where k=i.
        mask = (torch.ones_like(exponential, device=self.device) - torch.eye(2 * N, device=self.device)).bool()
    
        # We apply the binary mask.
        exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
        denom = torch.sum(exponential, dim=1, keepdim=True) # [2*N, 1]

        left_norm = out_left / torch.linalg.norm(out_left, keepdims=True, dim=1)
        right_norm = out_right / torch.linalg.norm(out_right, keepdims=True, dim=1)
        pos_pair = torch.sum(left_norm * right_norm, dim=1, keepdim=True) # [N, 1]

        pos_pair = torch.cat([pos_pair, pos_pair], dim=0) # [2*N, 1]
        numerator = torch.exp(pos_pair / tau)
        simclr_loss = torch.mean(-torch.log(numerator / denom))

        g_l = self.mG(out_left)
        g_r = self.mG(out_right)

        cls_loss = F.nll_loss(g_l, y) + F.nll_loss(g_r, y)
        self.log('simclr_loss', simclr_loss, prog_bar=True)
        self.log('cls_loss', cls_loss, prog_bar=True)
        loss = 0.5 * simclr_loss + cls_loss

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
