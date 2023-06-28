import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models import WarmStartGradientReverseLayer

class DomainAdversialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean', grl=None):
        super(DomainAdversialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        if grl is None:
            self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=3., max_iters=1500, auto_step=True)
        else:
            self.grl = grl

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_label_s = torch.ones(f_s.size(0)).to(f_s.device)
        d_label_t = torch.zeros(f_t.size(0)).to(f_t.device)
        d_label = torch.cat((d_label_s, d_label_t), dim=0)

        return F.cross_entropy(d, d_label.long(), reduction=self.reduction)