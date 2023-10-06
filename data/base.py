import os
from os.path import join
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import Dataset
from utils.path_util import PROJECT_PATH, create_dir
from utils.regex import find_data_txt


class BaseDataset(Dataset):

    def __init__(self, 
                dataset_path, 
                subjects, 
                classes,
                normal, 
                denoise, 
                save_action_detect_result=False,
                save_processing_result=False,
                verbose=False):
        
        self.classes = classes
        self.dataset_path = dataset_path
        self.save_action_detect_result = save_action_detect_result
        self.save_processing_result = save_processing_result
        self.verbose = verbose

        self.subject2data = self.load_data(subjects, normal, denoise,)

        # 将所有数据统一为 信号：标签, (tms, 8) (tms,)
        self.signal, self.labels = self.semg2label()

        if verbose:
            print(subjects, '共产生{}的活动段肌电数据'.format(self.signal.shape))

    def semg2label(self):
        semg, label = [], []

        for subject in sorted(self.subject2data.keys()):
            subject_data = self.subject2data[subject]
            for gesture in sorted(subject_data.keys()):
                # shape: [tms, channel]
                signal = self.subject2data[subject][gesture]
                signal_action = self.action_detect(signal, subject, gesture)
                semg.append(signal_action)
                label.append(np.ones(signal_action.shape[0], dtype=np.int) * (int(gesture) - 1))

        return np.concatenate(semg), np.concatenate(label)
    
    def action_detect(self, signal, subject, gesture):
        semg = signal[:, :5]
        # tms, 1
        energe = np.sum(np.abs(semg), axis=1)
        energe = self.normalization(energe, 'min-max')
        threshould = np.mean(energe)
        segments = []
        segment = []

        for i, e in enumerate(energe):
            if e > threshould and len(segment) < 1:
                segment.append(i)
            elif e < threshould and len(segment) == 1:
                segment.append(i)
                segments.append(segment)
                segment = []

        tmps = [segments[0]]
        i = 1
        while i < len(segments):
            pred = tmps[-1]
            curr = segments[i]
            if curr[0] - pred[1] <= 500:
                tmps[-1][1] = curr[1]
            else:
                tmps.append(curr)
            i += 1

        # for i in range(1, len(segments)):
        #     pred = segments[i-1]
        #     curr = segments[i]
        #     if curr[0] - pred[1] < 300:
        #         tmps.append([pred[0], curr[1]])
        #     else:
        #         tmps.append(pred)
        segments = tmps

        # 保存结果图
        if self.save_action_detect_result:
            plt.figure(figsize=(25, 18), dpi=80)
            plt.plot(energe)

            for seg in segments:
                le, ri = seg
                h = np.max(energe[le:ri])
                line_segment = [(le, 0), (le, h), (ri, h), (ri, 0)]
                x, y = zip(*line_segment)

                plt.plot(x, y, linestyle='-', color='r')
            create_dir(join(PROJECT_PATH, 'runs/action_detect', subject))
            plt.savefig(join(PROJECT_PATH, 'runs/action_detect', subject, '{}.png'.format(gesture)))
            if verbose:
                print('活动信号检测进程, subject: {}, gesture: {}'.format(subject, gesture))
            plt.close()

        # 获取活动段信号
        action_signal = []
        for seg in segments:
            le, ri = seg
            action_signal.append(signal[le:ri])
        return np.concatenate(action_signal)

    def load_data(self, subjects, normal, denoise,):
        # subject: {label: data}
        subject2data = {}

        for subject in subjects if isinstance(subjects, list) else [subjects]:
            subject_path = join(self.dataset_path, subject)
            gestures = sorted(os.listdir(subject_path))
            # label: data
            label2data = {}

            for gesture in gestures:
                gesture_path = join(subject_path, gesture)
                file = find_data_txt(os.listdir(gesture_path))[0]
                data = np.loadtxt(join(gesture_path, file))[:, 2:]

                if normal:
                    func = self.normalization
                    data = self.exec(func, data, normal, subject, gesture)
                if denoise:
                    func = self.signal_denoise
                    data = self.exec(func, data, denoise, subject, gesture)

                label2data[str(gesture)] = data

            subject2data[subject] = label2data
        return subject2data
    
    def exec(self, func, data, model, subject, gesture):
        semg, acc = data[:, :5], data[:, 5:]
        acc = self.normalization(acc, 'min-max')
        if self.save_processing_result:
            plt.figure(figsize=(25, 18), dpi=80)
            plt.subplot(2, 2, 1)
            plt.plot(semg)
            plt.subplot(2, 2, 2)
            plt.plot(acc)

            semg = func(semg, model)
            acc = self.average_filtering(acc)

            plt.subplot(2, 2, 3)
            plt.plot(semg)
            plt.subplot(2, 2, 4)
            plt.plot(acc)

            create_dir(join(PROJECT_PATH, 'runs/{}'.format(func.__name__), subject))
            plt.savefig(join(PROJECT_PATH, 'runs/{}'.format(func.__name__), subject, '{}.png'.format(gesture)))
            if self.verbose:
                print('信号处理{}进程, subject: {}, gesture: {}'.format(func.__name__, subject, gesture))
            plt.close()
        else:
            semg = func(semg, model)
            acc = self.average_filtering(acc)
        
        data = np.concatenate([semg, acc], axis=1)
        return data
    
    def average_filtering(self, acc):
        window = 20
        stride = 1
        tms = acc.shape[0]
        for i in range(window, stride, tms):
            acc[i, :] = np.mean(acc[i-window:i, :], axis=0)
        return acc


    def normalization(self, data, normal):
        if normal == 'min-max':
            max = np.max(data, axis=0)
            min = np.min(data, axis=0)
            return (data - min) / (max - min)
        elif normal == 'z-zero':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return (data - mean) / std

    def signal_denoise(self, data, denoise):
        '''
            denoise: 降噪方法

        '''
        data = data.T
        if denoise == 'ButterWorth':
            denoised_signal = []
        
            for ch in data:
                denoised_signal.append(self.butter_bandpass_filter(self.notch_filter(ch)))
            return np.array(denoised_signal).T
        elif denoise == 'Wavelet':
            denoised_signal = []
        
            for ch in data:
                denoised_signal.append(self.wavelet_filter(ch))
            return np.array(denoised_signal).T
        return data.T
    
    # 4阶的巴特沃斯滤波器，20-300Hz滤波
    def butter_bandpass_filter(self, data, lowcut=20, highcut=300, fs=1024, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # 定义50 Hz 陷波滤波器函数
    def notch_filter(self, data, cutoff=50.0, Q=30.0, fs=1024):
        nyquist = 0.5 * fs
        freq = cutoff / nyquist
        b, a = iirnotch(freq, Q)
        y = filtfilt(b, a, data)
        return y

    # 小波降噪
    def wavelet_filter(self, data, wavelet='db4', threshold=2.8, level=1):
        coeffs = pywt.wavedec(data, wavelet, level)
        coeffs_thresholde = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
        denoised_channel = pywt.waverec([coeffs[0]] + coeffs_thresholde, wavelet)
        return denoised_channel

    def __getitem__(self, index):
        pass

    def __len__(self,):
        pass
