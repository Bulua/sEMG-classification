import torch
import torch.nn as nn

from common import LSTMNorm, ConvNorm, LinearNorm


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
        y = self.linear(y)
        y = self.out(y)
        return y


if __name__ == '__main__':
    input = torch.ones((32, 300, 4))
    model = FeatureSelectNet(
        input_shape=input.shape,
        hidden_size=32,
        num_layers=4,
        bias=True,
        batch_first=True,
    )
    input = input.to('cuda')
    model = model.to('cuda')
    output = model(input)
    print(output.shape)

