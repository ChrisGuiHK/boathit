import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Callable

class FCNNaive(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],
                 hidden_size: Optional[int] = 320, kernel_sizes: Optional[List] = [3, 3, 3]):
        '''
        input_shape (N, L): N is the number of channels and L is the time series length.
        '''
        super(FCNNaive, self).__init__()

        N, L = input_shape
        assert L % 2 == 0, "The length of input should be an even number."

        self.downsample0 = nn.AvgPool1d(2)
        self.instancenorm0 = nn.InstanceNorm1d(N, affine=True)
        
        kernel_size = kernel_sizes[0]
        h1 = hidden_size
        self.padding1 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv1 = nn.Conv1d(in_channels=N, out_channels=h1, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(num_features=h1)
        self.dropout1 = nn.Dropout1d(p=0.1)

        kernel_size = kernel_sizes[1]
        h2 = h1 * 2
        self.padding2 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv2 = nn.Conv1d(in_channels=h1, out_channels=h2, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(num_features=h2)
        self.dropout2 = nn.Dropout1d(p=0.1)

        kernel_size = kernel_sizes[2]
        h3 = h2
        self.padding3 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv3 = nn.Conv1d(in_channels=h2, out_channels=h3, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm1d(num_features=h3)
        self.dropout3 = nn.Dropout1d(p=0.1)

        self.averagepool = nn.AvgPool1d(kernel_size=L//32) # 32 is downsample multiple


    def forward(self, x: Tensor) -> Tensor:
        '''
        x (B, N, L)
        '''
        x = self.downsample0(x) # downsample 2
        x = self.instancenorm0(x)

        x = self.padding1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.max_pool1d(x, 4) # downsample 4
        x = self.dropout1(x)

        x = self.padding2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = F.max_pool1d(x, 4) # downsample 4
        x = self.dropout2(x)

        x = self.padding3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)

        x = self.averagepool(x)
        x = x.squeeze_(-1)
        return x
    
class ConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(ConvUnit, self).__init__()
        self.padding = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.padding(x))

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], 
                 pooling: Optional[Callable] = lambda x: x, p: Optional[float] = 0.1) -> None:
        super(MultiScaleConvBlock, self).__init__()
        K = len(kernel_sizes)
        assert out_channels % K == 0, "The number of kernels does not match the number of out channels"
        self.ms = nn.ModuleList([ConvUnit(in_channels, out_channels//K, kernel_size) for kernel_size in kernel_sizes])
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.dropout = nn.Dropout1d(p=p)
        self.pooling = pooling
    
    def forward(self, x: Tensor) -> Tensor:
        xs = [m(x) for m in self.ms]
        x = torch.cat(xs, dim=-2)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        return x

class MultiScaleFCN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],
                 hidden_size: Optional[int] = 320, kernel_sizes: Optional[List] = [1, 3, 5, 7]):
        '''
        input_shape (N, L): N is the number of channels and L is the time series length.
        '''
        super(MultiScaleFCN, self).__init__()

        N, L = input_shape
        K = len(kernel_sizes)
        assert L % 2 == 0, "The length of input should be an even number."
        assert hidden_size % K == 0, "The number of kernels does not match hidden size."

        self.downsample0 = nn.AvgPool1d(2)
        self.instancenorm0 = nn.InstanceNorm1d(N, affine=True)
        

        h1 = hidden_size
        # self.padding1_0 = nn.ConstantPad1d(((kernel_sizes[0]-1)//2, kernel_sizes[0]//2), 0)
        # self.conv1_0 = nn.Conv1d(in_channels=N, out_channels=h1//K, kernel_size=kernel_sizes[0])
        # self.padding1_1 = nn.ConstantPad1d(((kernel_sizes[1]-1)//2, kernel_sizes[1]//2), 0)
        # self.conv1_1 = nn.Conv1d(in_channels=N, out_channels=h1//K, kernel_size=kernel_sizes[1])
        # self.padding1_2 = nn.ConstantPad1d(((kernel_sizes[2]-1)//2, kernel_sizes[2]//2), 0)
        # self.conv1_2 = nn.Conv1d(in_channels=N, out_channels=h1//K, kernel_size=kernel_sizes[2])
        # self.padding1_3 = nn.ConstantPad1d(((kernel_sizes[3]-1)//2, kernel_sizes[3]//2), 0)
        # self.conv1_3 = nn.Conv1d(in_channels=N, out_channels=h1//K, kernel_size=kernel_sizes[3])
        # self.bn1 = nn.BatchNorm1d(num_features=h1)
        # self.dropout1 = nn.Dropout1d(p=0.1)
        self.msconvblock1 = MultiScaleConvBlock(in_channels=N, out_channels=h1, kernel_sizes=kernel_sizes, 
                                                pooling=lambda x: F.max_pool1d(x, 4), p=0.1)
        

        h2 = h1 * 2
        # self.padding2_0 = nn.ConstantPad1d(((kernel_sizes[0]-1)//2, kernel_sizes[0]//2), 0)
        # self.conv2_0 = nn.Conv1d(in_channels=h1, out_channels=h2//K, kernel_size=kernel_sizes[0])
        # self.padding2_1 = nn.ConstantPad1d(((kernel_sizes[1]-1)//2, kernel_sizes[1]//2), 0)
        # self.conv2_1 = nn.Conv1d(in_channels=h1, out_channels=h2//K, kernel_size=kernel_sizes[1])
        # self.padding2_2 = nn.ConstantPad1d(((kernel_sizes[2]-1)//2, kernel_sizes[2]//2), 0)
        # self.conv2_2 = nn.Conv1d(in_channels=h1, out_channels=h2//K, kernel_size=kernel_sizes[2])
        # self.padding2_3 = nn.ConstantPad1d(((kernel_sizes[3]-1)//2, kernel_sizes[3]//2), 0)
        # self.conv2_3 = nn.Conv1d(in_channels=h1, out_channels=h2//K, kernel_size=kernel_sizes[3])
        # self.bn2 = nn.BatchNorm1d(num_features=h2)
        # self.dropout2 = nn.Dropout1d(p=0.1)
        self.msconvblock2 = MultiScaleConvBlock(in_channels=h1, out_channels=h2, kernel_sizes=kernel_sizes,
                                                pooling=lambda x: F.max_pool1d(x, 4), p=0.1)
        

        h3 = h2
        # self.padding3_0 = nn.ConstantPad1d(((kernel_sizes[0]-1)//2, kernel_sizes[0]//2), 0)
        # self.conv3_0 = nn.Conv1d(in_channels=h2, out_channels=h3//K, kernel_size=kernel_sizes[0])
        # self.padding3_1 = nn.ConstantPad1d(((kernel_sizes[1]-1)//2, kernel_sizes[1]//2), 0)
        # self.conv3_1 = nn.Conv1d(in_channels=h2, out_channels=h3//K, kernel_size=kernel_sizes[1])
        # self.padding3_2 = nn.ConstantPad1d(((kernel_sizes[2]-1)//2, kernel_sizes[2]//2), 0)
        # self.conv3_2 = nn.Conv1d(in_channels=h2, out_channels=h3//K, kernel_size=kernel_sizes[2])
        # self.padding3_3 = nn.ConstantPad1d(((kernel_sizes[3]-1)//2, kernel_sizes[3]//2), 0)
        # self.conv3_3 = nn.Conv1d(in_channels=h2, out_channels=h3//K, kernel_size=kernel_sizes[3])
        # self.bn3 = nn.BatchNorm1d(num_features=h3)
        # self.dropout3 = nn.Dropout1d(p=0.1)
        self.msconvblock3 = MultiScaleConvBlock(in_channels=h2, out_channels=h3, kernel_sizes=[1, 3, 5, 7],
                                                pooling=lambda x: x, p=0.1)
        

        self.averagepool = nn.AvgPool1d(kernel_size=L//32) # 32 is downsample multiple


    def forward(self, x: Tensor) -> Tensor:
        '''
        x (B, N, L)
        '''
        x = self.downsample0(x) # downsample 2
        x = self.instancenorm0(x)

        # x0 = self.conv1_0(self.padding1_0(x))
        # x1 = self.conv1_1(self.padding1_1(x))
        # x2 = self.conv1_2(self.padding1_2(x))
        # x3 = self.conv1_3(self.padding1_3(x))
        # x = torch.cat([x0, x1, x2, x3], dim=-2)
        # x = self.bn1(x)
        # x = F.gelu(x)
        # x = F.max_pool1d(x, 4) # downsample 4
        # x = self.dropout1(x)
        x = self.msconvblock1(x) # downsample 4
        
        # x0 = self.conv2_0(self.padding2_0(x))
        # x1 = self.conv2_1(self.padding2_1(x))
        # x2 = self.conv2_2(self.padding2_2(x))
        # x3 = self.conv2_3(self.padding2_3(x))
        # x = torch.cat([x0, x1, x2, x3], dim=-2)
        # x = self.bn2(x)
        # x = F.gelu(x)
        # x = F.max_pool1d(x, 4) # downsample 4
        # x = self.dropout2(x)
        x = self.msconvblock2(x) # downsample 4
        

        # x0 = self.conv3_0(self.padding3_0(x))
        # x1 = self.conv3_1(self.padding3_1(x))
        # x2 = self.conv3_2(self.padding3_2(x))
        # x3 = self.conv3_3(self.padding3_3(x))
        # x = torch.cat([x0, x1, x2, x3], dim=-2)
        # x = self.bn3(x)
        # x = F.gelu(x)
        # x = self.dropout3(x)
        x = self.msconvblock3(x)
        

        # x = self.averagepool(x)
        # x = x.squeeze_(-1)
        return x   

class GatedConv(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],
                 hidden_size: Optional[int] = 128, kernel_sizes: Optional[List] = [8, 5, 3]):
        '''
        input_shape (N, L): N is the number of channels and L is the time series length.
        '''
        super(GatedConv, self).__init__()

        N, L = input_shape
        assert L % 2 == 0, "The length of input should be an even number."

        self.downsample0 = nn.AvgPool1d(2)
        self.instancenorm0 = nn.InstanceNorm1d(N, affine=True)
        
        kernel_size = kernel_sizes[0]
        h1 = hidden_size
        self.padding1 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv1 = nn.Conv1d(in_channels=N, out_channels=h1*2, kernel_size=kernel_size)
        self.glu1 = nn.GLU(dim=-2)
        self.bn1 = nn.BatchNorm1d(num_features=h1)
        self.dropout1 = nn.Dropout1d(p=0.1)

        kernel_size = kernel_sizes[1]
        h2 = h1 * 2
        self.padding2 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv2 = nn.Conv1d(in_channels=h1, out_channels=h2*2, kernel_size=kernel_size)
        self.glu2 = nn.GLU(dim=-2)
        self.bn2 = nn.BatchNorm1d(num_features=h2)
        self.dropout2 = nn.Dropout1d(p=0.1)

        kernel_size = kernel_sizes[2]
        h3 = h2
        self.padding3 = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
        self.conv3 = nn.Conv1d(in_channels=h2, out_channels=h3*2, kernel_size=kernel_size)
        self.glu3 = nn.GLU(dim=-2)
        self.bn3 = nn.BatchNorm1d(num_features=h3)
        self.dropout3 = nn.Dropout1d(p=0.1)

        self.averagepool = nn.AvgPool1d(kernel_size=L//32) # 32 is downsample multiple


    def forward(self, x: Tensor) -> Tensor:
        '''
        x (B, N, L)
        '''
        x = self.downsample0(x) # downsample 2
        x = self.instancenorm0(x)

        x = self.padding1(x)
        x = self.glu1(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 4) # downsample 4
        x = self.dropout1(x)

        x = self.padding2(x)
        x = self.glu2(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 4) # downsample 4
        x = self.dropout2(x)

        x = self.padding3(x)
        x = self.glu3(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.averagepool(x)
        x = x.squeeze_(-1)
        return x