import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int | List[int], output_dim: int=2, sigmoid: bool = True, batch_norm: Optional[bool] = True):
        super(DomainDiscriminator, self).__init__()
        if sigmoid:
            sigmoid_layer = nn.Sigmoid()
        else:
            sigmoid_layer = nn.Identity()
        
        if isinstance(hidden_size, int): hidden_size = [hidden_size]
        kernel_sizes = [in_feature, *hidden_size, output_dim]

        self.layers = nn.Sequential()
        if batch_norm:
            for i in range(len(kernel_sizes)-2):
                self.layers.add_module(
                    f'layer_{i}',
                    nn.Sequential(
                        nn.Linear(kernel_sizes[i], kernel_sizes[i+1]),
                        nn.BatchNorm1d(kernel_sizes[i+1]),
                        nn.ReLU(),
                    )
                )
        else:
            for i in range(len(kernel_sizes)-2):
                self.layers.add_module(
                    f'layer_{i}',
                    nn.Sequential(
                        nn.Linear(kernel_sizes[i], kernel_sizes[i+1]),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
                )
        self.layers.add_module(f'layer_{len(kernel_sizes)-2}', nn.Linear(kernel_sizes[-2], kernel_sizes[-1]))
        self.layers.add_module(f'layer_{len(kernel_sizes)-1}', sigmoid_layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
