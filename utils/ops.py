import random
import contextlib
import time
import torch

import numpy as np

from torch import nn

class Profile(contextlib.ContextDecorator):

    def __init__(self, t=0.0) -> None:
        super().__init__()
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f'Elapsed time is {self.t} s'

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def initialize_weights(model):

    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentus = 0.03
        elif t is nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def init_seeds(randm_seed=0):
    random.seed(randm_seed)
    np.random.seed(randm_seed)
    torch.manual_seed(randm_seed)  # cpu
    torch.cuda.manual_seed(randm_seed)  # gpu
    torch.cuda.manual_seed_all(randm_seed)