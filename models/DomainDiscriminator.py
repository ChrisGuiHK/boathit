import torch 
import torch.nn as nn
from typing import List, Dict
from .FCN import MultiScaleConvBlock

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            MultiScaleConvBlock(in_channel, out_channel, kernel_sizes=[1, 3, 5, 7], p=0.1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channel, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class ConditionalDomainDiscriminator(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int) -> None:
        super(ConditionalDomainDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
