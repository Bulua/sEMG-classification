import os
import torch
import torch.nn as nn
import pathlib
import numpy as np
from os.path import join
from tqdm import tqdm
from utils.ops import Profile
from utils.path_util import DATA_PATH, PROJECT_PATH, create_dir
from models.models import FeatureSelectNet

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


feature_images = np.load('./runs/data/user1/feature_images.npy')
feature_image_labels = np.load('./runs/data/user1/feature_image_labels.npy')

all_features = np.array(['MAV', 'WMAV', 'SSC', 'WL', 'RMS','STD', 'SSI', 'VAR', 'AAC', 'MEAN'])
for i, f in enumerate(all_features):
    data = feature_images[:, i, 5:]
    data = np.reshape(data, (data.shape[0], -1))
    label = feature_image_labels

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f'{all_features[i]}: {accuracy_score(y_pred, y_test)}')
    
nn.Linear

# input = torch.ones((32, 4, 8))
# model = FeatureSelectNet(
#     input_shape=input.shape,
#     hidden_size=32,
#     num_layers=4,
#     bias=True,
#     batch_first=True,
# )
# input = input.to('cuda')
# model = model.to('cuda')
# output = model(input)
# print(output.shape)

# rnn = nn.LSTM(10, 20, 2, batch_first=True)
# input = torch.randn(3, 5, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

# linear = nn.Linear(100, 8)

# output = linear(torch.flatten(output, start_dim=1))
# print(output.shape)

# data = np.load(join(DATA_PATH, 'user1', 'feature_images.npy'))

# print(data.shape)

# p = Profile()

# with p:
#     for i in range(100000):
#         i = i
# print(p)


# params = torch.ones(1, dtype=torch.float64)
# optim = torch.optim.Adam([params], lr=0.9)
# lr_decay_epoch = list(range(0, 100, 10))
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer=optim,
#             T_max=20,
#             last_epoch=-1)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5, last_epoch=-1)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95, last_epoch=-1)

# for i in range(100):
#     lr = lr_scheduler.get_last_lr()
#     print(lr)
#     lr_scheduler.step()


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