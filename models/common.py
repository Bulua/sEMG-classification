import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.init as init


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


class LSTMNorm(nn.Module):

    def __init__(self, 
                 batch_size,
                 data_size,
                 input_size,
                 hidden_size,
                 num_layers,
                 bias=False,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 proj_size=0) -> None:
        super(LSTMNorm, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size
        )

        self.bn = nn.BatchNorm1d(data_size)
    
    def forward(self, x):
        device = x.device
        h0 = init.xavier_normal_(torch.empty(self.num_layers, self.batch_size, self.hidden_size)).to(device)
        c0 = init.xavier_normal_(torch.empty(self.num_layers, self.batch_size, self.hidden_size)).to(device)

        y, (hn, cn) = self.lstm(x, (h0, c0))
        y = self.bn(y)
        return y


class LinearNorm(nn.Module):

    def __init__(self, 
                 in_features,
                 out_features,
                 bias=False,
                 dropout=0.):
        super(LinearNorm, self).__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_features)
        if dropout != 0:
            self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        y = self.linear(x)
        y = self.bn(y)
        if hasattr(self, 'drop'):
            y = self.drop(y)
        return y


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

