import torch
import torch.nn as nn

from models.common import LSTMNorm, ConvNorm, LinearNorm


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





if __name__ == '__main__':
    input = torch.ones((32, 200, 10))
    model = FeatureNet(
        dim=10,
        num_heads=2,
    )
    input = input.to('cuda')
    model = model.to('cuda')
    output = model(input)
    print(output.shape)

