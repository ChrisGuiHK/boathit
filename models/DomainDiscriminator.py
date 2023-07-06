import torch 
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, output_dim: int=2, sigmoid: bool = False):
        super(DomainDiscriminator, self).__init__()
        if sigmoid:
            sigmoid_layer = nn.Sigmoid()
        else:
            sigmoid_layer = nn.Identity()
        
        self.layers = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            sigmoid_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
