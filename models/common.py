import imp
import torch
import torch.nn as nn
import torch.functional as F


class Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
