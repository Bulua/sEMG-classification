import argparse
import torch

import numpy as np

from os.path import join
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.path_util import DATA_PATH, SEMG_FEATURE_SELECT_PATH, ACC_FEATURE_SELECT_PATH
from utils.ops import init_seeds, plot_save
from data.semg_datasets import SemgFeatureDataset, ACCFeatureDataset



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据读取
    subject = 'user1'
    split_rate = 0.7

    feature_images_path = join(DATA_PATH, subject, 'feature_images.npy')
    feature_image_labels_path = join(DATA_PATH, subject, 'feature_image_labels.npy')

    feature_images = np.load(feature_images_path)
    feature_image_labels = np.load(feature_image_labels_path)

    # 所选sEMG特征
    # 可选
    all = ['MAV', 'WMAV', 'SSC', 'WL', 'RMS','STD', 'SSI', 'VAR', 'AAC', 'MEAN']
    # 单特征比较
    selected_f = [np.array(f) for f in all]
    accs = []
    
    for sf in tqdm(selected_f):
        # sfd = SemgFeatureDataset(feature_images, feature_image_labels, sf)
        sfd = ACCFeatureDataset(feature_images, feature_image_labels, sf)

        X_train, X_test, y_train, y_test = \
            train_test_split(sfd.features, sfd.labels, test_size=split_rate)

        knn = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_pred, y_test)

        accs.append(acc)

    plot_save(all, accs, path=ACC_FEATURE_SELECT_PATH, figname='one_feature')
    for f, a in zip(all, accs):
        print(f'{f}: {a}')
    

if __name__ == '__main__':
    init_seeds()
    main()