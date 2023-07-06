import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class WeightAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator, lamda):
        super(WeightAdversarialLoss, self).__init__()
        self.domainDiscriminator = domain_discriminator
        self.lamda = lamda

    def forward(self, src_f, trg_f):
        feature = torch.cat([src_f, trg_f], dim=0)
        domain = self.domainDiscriminator(feature)
        domain_src, domain_trg = torch.chunk(domain, 2)
        wdist = domain_src.mean() - domain_trg.mean()
        gp = gradient_penalty(self.domainDiscriminator, src_f, trg_f)
        
        return -wdist + self.lamda * gp



def gradient_penalty(critic: Callable, src_f: torch.Tensor, trg_f: torch.Tensor):
    batch_size = src_f.shape[0]
    alpha = torch.rand(batch_size, 1).to(src_f.device)
    differences = trg_f - src_f
    interpolates = src_f + (alpha * differences)
    interpolates = torch.cat([interpolates, src_f, trg_f]).requires_grad_()
    preds = critic(interpolates)
    gradients = torch.autograd.grad(
        preds, interpolates, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True
    )[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty



def get_partial_classes_weight(weights: torch.Tensor, labels: torch.Tensor):
    """
    Get class weight averaged on the partial classes and non-partial classes respectively.

    Args:
        weights (tensor): instance weight in shape :math:`(N, 1)`
        labels (tensor): ground truth labels in shape :math:`(N, 1)`

    .. warning::
        This function is just for debugging, since in real-world dataset, we have no access to the index of \
        partial classes and this function will throw an error when `partial_classes_index` is None.
    """

    weights = weights.squeeze()
    is_partial = torch.Tensor([label in [2, 3, 4] for label in labels]).to(weights.device)
    if is_partial.sum() > 0:
        partial_classes_weight = (weights * is_partial).sum() / is_partial.sum()
    else:
        partial_classes_weight = torch.tensor(0)

    not_partial = 1. - is_partial
    if not_partial.sum() > 0:
        not_partial_classes_weight = (weights * not_partial).sum() / not_partial.sum()
    else:
        not_partial_classes_weight = torch.tensor(0)
    return partial_classes_weight, not_partial_classes_weight