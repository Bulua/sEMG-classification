import os
import torch
import pathlib
import numpy as np


params = torch.ones(1, dtype=torch.float64)
optim = torch.optim.Adam([params], lr=0.9)
# lr_decay_epoch = list(range(0, 100, 10))
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer=optim,
#             T_max=20,
#             last_epoch=-1)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5, last_epoch=-1)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95, last_epoch=-1)

for i in range(100):
    lr = lr_scheduler.get_last_lr()
    print(lr)
    lr_scheduler.step()


# assert 1 == 2, '1 != 2'
# print(torch.__version__)
# def normalization(data):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     return (data - mean) / std


# data1 = [[100, 2, 3],
#         [5, 10, 9]]
# data2 = [[100, 2, 3],
#         [5, 10, 9]]

# print(np.concatenate([data1, data2], axis=1).shape)
# current_file = os.path.abspath(__file__)
# current_directory = os.path.dirname(__file__)
# print(current_directory)