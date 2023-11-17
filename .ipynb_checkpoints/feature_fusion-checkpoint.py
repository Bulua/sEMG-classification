import os
import arg
import torch

import numpy as np

from os.path import join
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.path_util import DATA_PATH
from data.data_fusion import DataFusion


def sliding_window(datas, labels, window=200, stride=50):
    tms = datas.shape[0]
    sample_rate = 1000
    window = int(window / 1000 * sample_rate)
    cnt = (tms - window) // stride + 1
    images, image_labels = [], []

    for i in range(cnt):
        image = datas[i*stride:i*stride+window, :]
        major_class = np.argmax(np.bincount(labels[i*stride:i*stride+window]))
        images.append(image)
        image_labels.append(major_class)
    return np.array(images), np.array(image_labels)


def main():
    subjects = ['user1']
    
    datas = []
    labels = []
    for s in subjects:
        data  = np.load(join(DATA_PATH, s, 'signal.npy'))
        label = np.load(join(DATA_PATH, s, 'labels.npy'))
        datas.append(data)
        labels.append(label)
    datas  = np.concatente(datas)
    labels = np.concatente(labels)
    
    df = DataFusion(datas)
    mode = 'kalman'
    fusion_datas = df.fusion(mode)
    
    
    
if __name__ == '__main__':
    main()