import torch
import torch.nn as nn
import torch.nn.functional as F
from models import WarmStartGradientReverseLayer

class ConditionalDomainAdversialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, grl: nn.Module = None, reduction: str = 'mean'):
        super(ConditionalDomainAdversialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = grl if grl is not None else WarmStartGradientReverseLayer(alpha=5.0, lo=0., hi=.1, max_iters=2000, auto_step=True)
        self.reduction = reduction
        self.map = LinearMap()
    
    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        assert batch_size % 2 == 0, "batch size must be even"
        size = batch_size // 2
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        d_label = torch.cat([
                torch.ones(size).to(f.device),
                torch.zeros(size).to(f.device),
            ])
        return F.cross_entropy(d, d_label.long(), reduction=self.reduction)



class LinearMap(nn.Module):
    def __init__(self):
        super(LinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)