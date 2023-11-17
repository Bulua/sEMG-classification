import os
import torch
import argparse

import numpy as np

from os.path import join
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils.path_util import DATA_PATH
from data.data_fusion import DataFusion
from utils.features import *
from utils.dataset_util import split_dataset


def calculate_features(image, fs):
    features = []
    for f in fs:
        features.append(eval(f)(image))
    return np.array(features)


def sliding_window(datas, labels, window=200, stride=50):
    tms = datas.shape[0]
    sample_rate = 1000
    window = int(window / 1000 * sample_rate)
    cnt = (tms - window) // stride + 1
    images, image_labels = [], []

    for i in range(cnt):
        image = datas[i*stride:i*stride+window, :]
        major_class = np.argmax(np.bincount(labels[i*stride:i*stride+window]))

        semg_features = calculate_features(image[:, :5], ['WL', 'MEAN', 'SSC', 'WMAV', 'STD'])
        acc_features = calculate_features(image[:, 5:], ['WMAV', 'STD', 'SSI', 'MEAN'])

        image = np.concatenate([semg_features.flatten(), acc_features.flatten()])
        images.append(image)
        image_labels.append(major_class)
    return np.array(images), np.array(image_labels)


def do_kalman(subjects):
    datas = []
    labels = []
    for s in subjects:
        data  = np.load(join(DATA_PATH, s, 'signal.npy'))
        label = np.load(join(DATA_PATH, s, 'labels.npy'))
        datas.append(data)
        labels.append(label)
    datas  = np.concatenate(datas)
    labels = np.concatenate(labels)
    
    df = DataFusion(datas)
    
    fusion_datas = df.fusion('kalman')
    images, image_labels = sliding_window(fusion_datas, labels)
    split_rate = 0.3
    train_idx, test_idx = split_dataset(len(images), split_rate)

    X_train = images[train_idx]
    X_test = images[test_idx]
    y_train = image_labels[train_idx]
    y_test = image_labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def gcca(data_list, num_components):
    data_list = [data - np.mean(data, axis=0) for data in data_list]
    cov_list = [np.cov(data, rowvar=False) for data in data_list]

    total_cov = sum(cov_list)

    _, eigenvectors = eigh(total_cov, subset_by_index=(len(total_cov)-num_components, len(total_cov)-1))
    projection_matrices = [eigenvectors.T.dot(cov).dot(eigenvectors) for cov in cov_list]

    return projection_matrices, eigenvectors

def do_gcca(subjects):
    datas = []
    labels = []
    for s in subjects:
        data  = np.load(join(DATA_PATH, s, 'signal.npy'))
        label = np.load(join(DATA_PATH, s, 'labels.npy'))
        datas.append(data)
        labels.append(label)
    datas  = np.concatenate(datas)
    labels = np.concatenate(labels)

    semg, acc = datas[:, :5], datas[:, 5:]
    acc = np.pad(acc, ((0, 0), (0, semg.shape[1] - acc.shape[1])), mode='constant')

    print('gcca fusion start....')
    projection_matrices, eigenvectors = gcca([semg, acc], num_components=semg.shape[1])
    projected_data1 = semg.dot(projection_matrices[0])
    projected_data2 = acc.dot(projection_matrices[1])
    gcca_datas = np.hstack((projected_data1, projected_data2))
    gcca_datas = preprocessing.StandardScaler().fit_transform(gcca_datas)
    print('gcca fusion successfully!')

    images, image_labels = sliding_window(gcca_datas, labels)
    split_rate = 0.3
    train_idx, test_idx = split_dataset(len(images), split_rate)

    X_train = images[train_idx]
    X_test = images[test_idx]
    y_train = image_labels[train_idx]
    y_test = image_labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc


def do_cca(subjects):
    datas = []
    labels = []
    for s in subjects:
        data  = np.load(join(DATA_PATH, s, 'signal.npy'))
        label = np.load(join(DATA_PATH, s, 'labels.npy'))
        datas.append(data)
        labels.append(label)
    datas  = np.concatenate(datas)
    labels = np.concatenate(labels)

    semg, acc = datas[:, :5], datas[:, 5:]
    cca = CCA(n_components=3)
    print('cca fusion start....')
    cca.fit(semg, acc)
    X_c, Y_c = cca.transform(semg, acc)
    merged_data = np.hstack((X_c, Y_c))
    merged_data = preprocessing.StandardScaler().fit_transform(merged_data)
    print('cca fusion successfully!')

    images, image_labels = sliding_window(merged_data, labels)
    split_rate = 0.3
    train_idx, test_idx = split_dataset(len(images), split_rate)

    X_train = images[train_idx]
    X_test = images[test_idx]
    y_train = image_labels[train_idx]
    y_test = image_labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def do_serial(subjects):
    datas = []
    labels = []
    for s in subjects:
        data  = np.load(join(DATA_PATH, s, 'signal.npy'))
        label = np.load(join(DATA_PATH, s, 'labels.npy'))
        datas.append(data)
        labels.append(label)
    datas  = np.concatenate(datas)
    labels = np.concatenate(labels)
    datas = preprocessing.StandardScaler().fit_transform(datas)
    print('serial fusion start....')
    images, image_labels = sliding_window(datas, labels)
    print('serial fusion successfully!')
    split_rate = 0.3
    train_idx, test_idx = split_dataset(len(images), split_rate)

    X_train = images[train_idx]
    X_test = images[test_idx]
    y_train = image_labels[train_idx]
    y_test = image_labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def main():
    subjects = ['user1']
    modes = ['kalman', 'cca', 'serial', 'gcca']
    accs = {}

    for m in modes:
        if m == 'kalman':
            acc = do_kalman(subjects)
        elif m == 'cca':
            acc = do_cca(subjects)
        elif m == 'serial':
            acc = do_serial(subjects)
        elif m == 'gcca':
            acc = do_gcca(subjects)
        accs[m] = round(acc * 100, 2)
    
    # {'kalman': 83.35, 'cca': 63.52, 'serial': 86.11, 'gcca': 83.21}
    print(accs)
    
    
if __name__ == '__main__':
    main()