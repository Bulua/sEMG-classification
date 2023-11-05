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
            bias=bias,
        )
        if norm_layer == 'batchnorm2d':
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv1dNorm(nn.Module):

    def __init__(self, in_ch, out_ch, k_size, p, s):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, 
                              out_channels=out_ch,
                              kernel_size=k_size,
                              padding='same' if p == -1 else 'valid',
                              stride=s)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        return y

class ResidualConv(nn.Module):

    def __init__(self, in_ch, out_ch, k_size, p, s):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_ch, 
                              out_channels=out_ch,
                              kernel_size=k_size,
                              padding='same' if p == -1 else 'valid',
                              stride=s)
        self.linear = nn.Conv1d(in_channels=out_ch, 
                              out_channels=out_ch // 2,
                              kernel_size=1,
                              padding='same' if p == -1 else 'valid',
                              stride=s)
        self.conv2 = nn.Conv1d(in_channels=out_ch // 2, 
                              out_channels=out_ch,
                              kernel_size=k_size,
                              padding='same' if p == -1 else 'valid',
                              stride=s)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        y = self.linear(x)
        y = self.conv2(y)
        y = self.act(y)
        return x + y


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
        self.bidirectional = bidirectional

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
        w = torch.empty(self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size)
        h0 = init.xavier_normal_(w).to(device)
        c0 = init.xavier_normal_(w).to(device)

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
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

