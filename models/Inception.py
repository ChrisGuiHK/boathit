import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable
from torch import Tensor

class ConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(ConvUnit, self).__init__()
        self.padding = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.padding(x))


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], bottleneck_dim: int) -> None:
        super(InceptionModule, self).__init__()
        K = len(kernel_sizes)
        assert out_channels % (K + 1) == 0, "The number of kernels does not match the number of out channels"
        self.ms = nn.ModuleList([ConvUnit(bottleneck_dim, out_channels//(K+1), kernel_size) for kernel_size in kernel_sizes])
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_dim, 1)
        self.bottleneck_mp = ConvUnit(in_channels, out_channels//(K+1), bottleneck_dim)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = x.clone()
        y = self.bottleneck(y)
        y = torch.cat([m(y) for m in self.ms], dim=1)

        x = self.maxpool(x)
        x = self.bottleneck_mp(x)

        x = torch.cat([y, x], dim=1)
        x = self.bn(x)
        return x
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], bottleneck_dim: int) -> None:
        super(InceptionBlock, self).__init__()
        self.inception = InceptionModule(in_channels, out_channels, kernel_sizes, bottleneck_dim)
        self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.inception(x) + self.shortcut(x))


class InceptionTime(nn.Module):
    def __init__(self, channels: List[int], kernel_sizes: List[int], bottleneck_dim: List[int]) -> None:
        super(InceptionTime, self).__init__()
        assert len(channels) == len(bottleneck_dim) + 1, "The number of channels must be equal to the number of bottleneck dimensions + 1"
        block_nums = len(channels) - 1
        self.blocks = nn.ModuleList([InceptionBlock(channels[i], channels[i+1], kernel_sizes, bottleneck_dim[i]) for i in range(block_nums)])
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.downsample0 = nn.AvgPool1d(2)
        self.instancenorm0 = nn.InstanceNorm1d(channels[0], affine=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample0(x)
        x = self.instancenorm0(x)

        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        return x.squeeze(-1)