from typing import Optional, List
import torch
import torch.nn as nn

def get_importance_weight(discriminator, feature):
    """
    Get importance weights for each instance.

    Args:
        feature (tensor): feature from source domain, in shape :math:`(N, F)`

    Returns:
        instance weight in shape :math:`(N, )`
    """
    weight = 1. - discriminator(feature.detach())
    weight = weight / weight.mean()
    weight = weight.squeeze().detach()
    return weight

def get_partial_classes_weight(weights: torch.Tensor, labels: torch.Tensor, partial_classes_index: List[int] = None):
    """
    Get class weight averaged on the partial classes and non-partial classes respectively.

    Args:
        weights (tensor): instance weight in shape :math:`(N, 1)`
        labels (tensor): ground truth labels in shape :math:`(N, 1)`

    .. warning::
        This function is just for debugging, since in real-world dataset, we have no access to the index of \
        partial classes and this function will throw an error when `partial_classes_index` is None.
    """
    assert partial_classes_index is not None

    weights = weights.squeeze()
    is_partial = torch.Tensor([label in partial_classes_index for label in labels]).to(weights.device)
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

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H