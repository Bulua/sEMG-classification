import imp
from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F


class ConvNorm(nn.Module):

    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding='valid',
        bias=True,
        norm_layer='batchnorm2d',
        norm_kwargs=None
    ):
        norm_kwargs = norm_kwargs or {}
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if norm_layer == 'batchnorm2d':
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Attention(nn.Module):

    def __init__(self, ):
        super(Attention, self).__init__()

    def forward(self, x):

        return x



class RowNet(nn.Module):
    '''
        以原始肌电信号为输入的网络
    '''
    def __init__(self, cfgs):
        super(RowNet, self).__init__()

    def forward(self, x):
        return x


if  __name__ == '__main__':

    test_net = 'ConvNorm'
    
    if test_net == 'ConvNorm':
        x = torch.ones((64, 1, 300, 5))
        bn, ic, h, w = x.shape
        m = ConvNorm(ic, 32, padding='valid')
        y = m(x)
        print(y.shape)
    elif test_net == '':
        pass

