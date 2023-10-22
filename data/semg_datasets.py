import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from os.path import join
from .base import BaseDataset
from sklearn import preprocessing
from utils.path_util import PROJECT_PATH, DATA_PATH, create_dir
from utils.features import MAV, WMAV, SSC, ZC, WA, WL, RMS, STD, SSI, VAR, AAC, MEAN


class RowDataset(BaseDataset):

    def __init__(self, 
                dataset_path, 
                subjects, 
                classes, 
                window, 
                stride, 
                normalization, 
                denoise,
                save_action_detect_result, 
                save_processing_result,
                save_data,
                verbose):
        '''
        dataset_path: /autodl-tmp/resources/dataset/dzp
        subjects: ['user1', 'user2', ...]
        '''
        super(RowDataset, self).__init__(dataset_path, 
                                        subjects, 
                                        classes, 
                                        normalization, 
                                        denoise, 
                                        save_action_detect_result, 
                                        save_processing_result, 
                                        save_data,
                                        verbose)
        self.window = int(window / 1000 * self.sample_rate)
        self.stride = stride
        
        # 父类BaseDataset包含 self.signal, self.labels
        self.images, self.image_labels = self.sliding_window()

        if self.verbose:
            print('原始肌电图处理结果, 生成肌电图像: {}, 生成标签: {}'
                    .format(self.images.shape, self.image_labels.shape))
        if save_data:
            self.save_row(self.images, self.image_labels)

    def save_row(self, datas, labels):
        assert len(self.subjects) == 1 , '仅能保存一个人的原始数据,多人数据请设置save_data=False...'

        for subject in tqdm(self.subjects, 
                            desc='save row image data', 
                            total=len(self.subjects),
                            leave=True):
            p = create_dir(join(DATA_PATH, subject))

            if not os.path.exists(join(p, 'images.npy')):
                np.save(join(p, 'images.npy'), datas)
            if not os.path.exists(join(p, 'image_labels.npy')):   
                np.save(join(p, 'image_labels.npy'), labels)
        print('row image数据保存完毕!')

    def sliding_window(self):
        tms = self.signal.shape[0]
        cnt = (tms - self.window) // self.stride + 1
        images, image_labels = [], []

        for i in range(cnt):
            label = np.zeros(self.classes, dtype=np.int)
            image = self.signal[i*self.stride:i*self.stride+self.window, :]
            major_class = self.get_major_class(self.labels[i*self.stride:i*self.stride+self.window])
            label[major_class] = 1
            images.append(image)
            image_labels.append(label)
        return np.array(images), np.array(image_labels)

    def get_major_class(self, nums):
        return np.argmax(np.bincount(nums))
    
    @property
    def shape(self):
        return self.images.shape[1:]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self,):
        return len(self.images)


class TimeFeatureDataset(BaseDataset):

    def __init__(self, 
                dataset_path, 
                subjects, 
                classes,
                features, 
                window, 
                stride, 
                normalization, 
                denoise, 
                save_action_detect_result,
                save_processing_result,
                save_data,
                verbose):
        '''
            feature: 
                平均绝对值MAV, 加权平均绝对值WMAV, 斜率符号变化
                过零点率ZC, 威利森幅值WA, 波形长度WL, 均方根RMS
                标准差STD, 简单方形积分SSI, 方差VAR, 平均幅度改变AAC
                均值MEAN
        '''
        super(TimeFeatureDataset, self).__init__(dataset_path, 
                                                 subjects, 
                                                 classes, 
                                                 normalization, 
                                                 denoise, 
                                                 save_action_detect_result, 
                                                 save_processing_result,
                                                 save_data,
                                                 verbose)

        self.window = int(window / 1000 * self.sample_rate)
        self.stride = stride
        self.features = features
        # 父类BaseDataset包含 self.signal, self.labels
        self.feature_images, self.feature_image_labels = self.sliding_window()
        self.feature_images = np.expand_dims(self.feature_images, axis=1)

        if self.verbose:
            print('特征图处理结果, 生成特征图像: {}, 生成标签: {}'
                    .format(self.feature_images.shape, self.feature_image_labels.shape))
        if save_data:
            self.save_features(self.feature_images, self.feature_image_labels)

    def save_features(self, datas, labels):
        assert len(self.subjects) == 1 , '仅能保存一个人的时域数据,多人数据请设置save_data=False...'

        for subject in tqdm(self.subjects, 
                            desc='save time-features data', 
                            total=len(self.subjects),
                            leave=True):
            p = create_dir(join(DATA_PATH, subject))

            if not os.path.exists(join(p, 'feature_images.npy')):
                np.save(join(p, 'feature_images.npy'), datas)
            if not os.path.exists(join(p, 'feature_image_labels.npy')):   
                np.save(join(p, 'feature_image_labels.npy'), labels)
        print('time-features数据保存完毕!')

    def sliding_window(self):
        tms = self.signal.shape[0]
        cnt = (tms - self.window) // self.stride + 1
        feature_images, feature_image_labels = [], []

        for i in range(cnt):
            label = np.zeros(self.classes, dtype=np.int)
            image = self.signal[i*self.stride:i*self.stride+self.window, :]
            major_class = self.get_major_class(self.labels[i*self.stride:i*self.stride+self.window])
            label[major_class] = 1

            features = self.calculate_features(image.T).T
            features = preprocessing.StandardScaler().fit_transform(features)

            feature_images.append(features)
            feature_image_labels.append(label)
        return np.array(feature_images, dtype=np.float), np.array(feature_image_labels, dtype=np.float)

    def get_major_class(self, nums):
        return np.argmax(np.bincount(nums))
    
    def calculate_features(self, image):
        features = []
        for f in self.features:
            features.append(eval(f)(image))
        return np.array(features)
    
    def feature_norm(self, features):
        '''
        features: (12, 8)
        对每行的特征进行归一化
        '''
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / std
    
    def standardize(self, images):
        # （X-X的均值）/ X的标准差
        images = (images - np.mean(images, axis=0)[None, :, :]) / np.std(images, axis=0)[None, :, :]
        return images

    @property
    def shape(self):
        return self.feature_images.shape[1:]

    def __getitem__(self, index):
        feature_images = torch.from_numpy(self.feature_images).to(torch.float)
        feature_image_labels = torch.from_numpy(self.feature_image_labels).to(torch.float)
        return feature_images[index], feature_image_labels[index]

    def __len__(self,):
        return len(self.feature_images)
        

class TimeFeatureDataset(BaseDataset):

    def __init__(self, 
                dataset_path, 
                subjects, 
                classes,
                features, 
                window, 
                stride, 
                normalization, 
                denoise, 
                save_action_detect_result,
                save_processing_result,
                save_data,
                verbose):
        '''
            feature: 
                平均绝对值MAV, 加权平均绝对值WMAV, 斜率符号变化
                过零点率ZC, 威利森幅值WA, 波形长度WL, 均方根RMS
                标准差STD, 简单方形积分SSI, 方差VAR, 平均幅度改变AAC
                均值MEAN
        '''
        super(TimeFeatureDataset, self).__init__(dataset_path, 
                                                    subjects, 
                                                    classes, 
                                                    normalization, 
                                                    denoise, 
                                                    save_action_detect_result, 
                                                    save_processing_result,
                                                    save_data,
                                                    verbose)
        self.window = int(window / 1000 * self.sample_rate)
        self.stride = stride
        self.features = features
        # 父类BaseDataset包含 self.signal, self.labels
        self.feature_images, self.feature_image_labels = self.sliding_window()