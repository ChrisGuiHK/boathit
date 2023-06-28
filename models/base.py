import torch
import torch.nn as nn
from torch import Tensor
from models import MultiScaleConvBlock

# class LinearClassifier(nn.Module):
#     def __init__(self, n_f: int, n_class: int) -> None:
#         super(LinearClassifier, self).__init__()
#         self.linear = nn.Linear(n_f, n_class)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.linear(x)
#         return torch.log_softmax(x, dim=1)
    
LinearClassifier = lambda n_f, n_class: nn.Sequential(
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(n_f, n_class),
    nn.LogSoftmax(dim=1)
)

FFNClassifier = lambda n_f, n_class: nn.Sequential(
    nn.Linear(n_f, 512),
    nn.Tanh(),
    nn.Linear(512, n_class),
    nn.LogSoftmax(dim=1)
)

FeatureHead = lambda in_channel, out_channel: nn.Sequential(
    MultiScaleConvBlock(in_channel, out_channel, kernel_sizes=[1, 3, 5, 7]),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
)