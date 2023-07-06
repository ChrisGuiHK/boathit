import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models import WarmStartGradientReverseLayer

class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean', grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        if grl is None:
            self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1500, auto_step=True)
        else:
            self.grl = grl

        

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, w_s: Optional[torch.Tensor] = None, sigmoid: bool=False) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_label_s = torch.ones(f_s.size(0)).to(f_s.device)
        d_label_t = torch.zeros(f_t.size(0)).to(f_t.device)
        d_label = torch.cat((d_label_s, d_label_t), dim=0)

        if w_s is None:
            w_s = torch.ones((f_s.size(0), )).to(f_s.device)
        w_t = torch.ones((f_t.size(0), )).to(f_t.device)

        if sigmoid:
            d_s, d_t = torch.chunk(d.squeeze(), 2)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_label_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_label_t), reduction=self.reduction)
            )
        else:
            loss = F.cross_entropy(d, d_label.long(), reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)