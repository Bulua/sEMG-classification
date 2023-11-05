import torch
import torch.nn as nn
import yaml

from torch.nn import MaxPool1d
from models.common import LSTMNorm, ConvNorm, LinearNorm, Attention, Conv1dNorm, ResidualConv


class FeatureSelectNet(nn.Module):

    def __init__(self, 
                input_shape, 
                hidden_size=32, 
                num_layers=1,
                bias=False,
                batch_first=True,
                bidirectional=False,
                dropout=0.,
                proj_size=0,
                classes=8):
                
        super(FeatureSelectNet, self).__init__()
        # 时域特征：['MAV', 'WMAV','SSC','ZC','WA','WL','RMS','STD','SSI','VAR','AAC','EAN']"
        # 32, 200, 3
        self.lstm = LSTMNorm(
            batch_size=input_shape[0],
            data_size=input_shape[1],
            input_size=input_shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout,
            proj_size=proj_size
        )
        in_features = hidden_size*input_shape[1]

        self.linear = LinearNorm(
            in_features=in_features,
            out_features=in_features//2,
            bias=True,
            dropout=dropout
        )
        self.out = nn.Linear(in_features//2, classes)

    def forward(self, x):
        y = self.lstm(x)
        y = torch.flatten(y, start_dim=1)
        y = nn.LeakyReLU()(y)
        y = self.linear(y)
        y = nn.LeakyReLU()(y)
        y = self.out(y)
        y = torch.softmax(y, dim=0)
        return y


class FeatureNet(nn.Module):

    def __init__(self, input_shape, classes) -> None:
        super(FeatureNet, self).__init__()

        self.conv1 = ConvNorm(input_shape[1], 
                              out_channels=32, 
                              kernel_size=3,
                              stride=1,
                              padding='same')
        self.conv2 = ConvNorm(in_channels=32, 
                              out_channels=64, 
                              kernel_size=3,
                              stride=1,
                              padding='same')
        self.conv3 = ConvNorm(in_channels=64, 
                              out_channels=128, 
                              kernel_size=3,
                              stride=1,
                              padding='same')
        self.att = Attention(dim=200, num_heads=4)
        self.linear1 = LinearNorm(in_features=input_shape[2]*128,
                                 out_features=256, 
                                 dropout=0.5)
        self.linear2 = LinearNorm(in_features=256,
                                 out_features=64, 
                                 dropout=0.5)
        self.linear3 = LinearNorm(in_features=64,
                                 out_features=32, 
                                 dropout=0.5)
        self.out = nn.Linear(32, classes)
    

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = nn.MaxPool2d(kernel_size=(1, 10))(y)
        y = torch.squeeze(y)
        y = self.att(y)
        y = torch.flatten(y, start_dim=1)
        
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        y = self.out(y)
        y = torch.softmax(y, dim=0)
        return y


class Net(nn.Module):

    def __init__(self, input_shape, net_cfg):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.model = self.load_model(net_cfg)

    def load_model(self, cfg):
        input_shape = self.input_shape
        model = nn.ModuleDict()
        
        encoder1 = nn.ModuleList()   # semg encoder
        encoder2 = nn.ModuleList()   # acc  encoder
        head = nn.ModuleList()       # predict head
        
        for key in cfg.keys():
            if key == 'e1':
                module_list = encoder1
            elif key == 'e2':
                module_list = encoder2
            elif key == 'head':
                module_list = head
            model[key] = self.load_layers(module_list, cfg[key])
        return model
    
    def load_layers(self, module_list, cfg):
        for m in cfg:
            for _ in range(m[1]):
                a = m[3]
                global m_
                if m[2] in ['Conv1dNorm', 'ResidualConv']:
                    m_ = globals()[m[2]](a[0], a[1], a[2], a[3], a[4])
                elif m[2] in ['MaxPool1d']:
                    m_ = nn.MaxPool1d(
                        kernel_size=a[0],
                        padding=a[1],
                        stride=a[2]
                    )
                elif m[2] in ['Linear']:
                    m_ = nn.Linear(
                        in_features=a[0],
                        out_features=a[1],
                        bias=True if a[2] == 1 else False
                    )
                module_list.append(m_)
        return module_list

    def forward(self, x):
        # shape: (32, 25+12)
        y = torch.unsqueeze(x, dim=1)
        # shape: (32, 1, 25+12)
        y1 = y[:, :, :25]
        y2 = y[:, :, 25:]

        for m in self.model['e1']:
            y1 = m(y1)
        for m in self.model['e2']:
            y2 = m(y2)
        y = torch.concatenate([y1, y2], dim=-1)
        for m in self.model['head']:
            y = torch.squeeze(m(y), dim=-1)
        # shape: (32, 128, 19)
        y = nn.Softmax(dim=1)(y)
        return y



if __name__ == '__main__':
    input = torch.ones((32, 37))
    with open('configs/net.yaml', 'r') as f:
        args = yaml.safe_load(f)

    model = Net(input.shape, args)
    input = input.to('cuda')
    model = model.to('cuda')

    output = model(input)
    print(output.shape)

