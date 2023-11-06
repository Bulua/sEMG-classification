import random
import contextlib
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from torch import nn


def plot_save(x, y, path, figname='', figsize=(8, 15)):
    x = [str(i) for i in x]
    if figname == '':
        figname = random.randint(0, 100)

    plt.figure(figsize=figsize, dpi=80)

    plt.bar(x, y)
    min_y, max_y = np.min(y) - 0.05, np.max(y) + 0.02
    plt.xticks(range(len(x)), x, rotation=-45, ha='left')
    for i, h in enumerate(y):
        plt.text(i, h, s=round(h, 3), ha='center')

    plt.xlabel('feature')
    plt.ylabel('acc(%)')
    plt.ylim(min_y, max_y)

    plt.savefig(f'{path}/{figname}.jpg')
    plt.close()
    logging.info(f'fig save path:{path}/{figname}.jpg')


# x = range(10)
# y = range(10)
class LossHistory:
    
    def __init__(self, train_acc, train_loss, valid_acc, valid_loss, path):
        assert len(train_acc) == len(train_loss) == len(valid_acc) == len(valid_loss), \
                'train_acc, train_loss, valid_acc, valid_loss数据不等长'
        
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.valid_acc = valid_acc
        self.valid_loss = valid_loss
        
        self.epochs = len(train_acc)
        self.save_path = path
    
    def plot_and_save(self, save_name='', figsize=(8, 8), dpi=80):
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.plot(self.train_acc, '--*', label='train acc')
        plt.plot(self.train_loss, '-.', label='train loss')
        plt.plot(self.valid_acc, '--', label='valid acc')
        plt.plot(self.valid_loss, '-*', label='valid loss')
        
        plt.xlim(1, self.epochs)
        plt.xlabel('epochs')
        plt.ylabel('accuracy/loss')
        
        plt.legend()
        # plt.show()
        plt.savefig(join(self.save_path, save_name))
        print(f'train/valid acc/loss saved in {join(self.save_path, save_name)}')


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