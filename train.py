import argparse

import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from data.datasets import RowDataset, TimeFeatureDataset
from utils.dataset_util import split_dataset


def train(train_dataloader, 
          valid_dataloader, 
          epochs,
          lr,
          ):
    pass


def main(args):
    dataset = None
    if args.features == 'row':
        dataset = RowDataset(args.data_path, 
                            args.subjects, 
                            args.classes,
                            args.window, 
                            args.stride, 
                            args.normalization,
                            args.denoise,
                            args.save_action_detect_result,
                            args.save_processing_result,
                            args.verbose)
    else:
        dataset = TimeFeatureDataset(
                            args.data_path,
                            args.subjects,
                            args.classes,
                            args.features,
                            args.window, 
                            args.stride, 
                            args.normalization,
                            args.denoise,
                            args.save_action_detect_result,
                            args.save_processing_result,
                            args.verbose)

    if args.cross_validation > 1:
        k_fold = KFold(n_splits=args.cross_validation)

        for train_idx, valid_idx in k_fold.split(dataset):
            train_set = Subset(dataset, train_idx)
            valid_set = Subset(dataset, valid_idx)
            # print(train_idx, valid_idx)
            print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

            train_dataloader = DataLoader(train_set, batch_size=args.batch_size)
            valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size)

            # 开始训练
            train(train_dataloader, valid_dataloader, args.epochs, args.lr)
    else:
        train_idx, valid_idx = split_dataset(len(dataset), split_rate=0.7)
        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)
        print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size)
        valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size)

        # 开始训练
        train(train_dataloader, valid_dataloader, args.epochs, args.lr)


def init_setting(randm_seed=0):
    np.random.seed(randm_seed)
    torch.manual_seed(randm_seed)  # cpu
    torch.cuda.manual_seed(randm_seed)  # gpu

    # np.set_printoptions(threshold=np.inf)


def parse_opt():
    init_setting()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='', help='网络模型')
    parser.add_argument('--subjects', type=list, 
                        default=['user1'], 
                        # default=['user1', 'user2', 'user3', 'user4', 'user5', 
                        #         'user6', 'user7', 'user8', 'user10'], 
                        help='用户')
    parser.add_argument('--classes', type=list, default=8, help='手势数量')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/resources/dataset/dzp', 
                        help='数据路径')
    parser.add_argument('--denoise', type=str, default='Wavelet', 
                        help='降噪方法, Wavelet、ButterWorth降噪, ...')
    parser.add_argument('--normalization', type=str, default='z-zero', help='数据标准化方法, min-max, z-zero')
    parser.add_argument('--active_signal_detect', type=str, default='', help='活动段检测方法, ...')
    parser.add_argument('--features', 
                        default=['MAV', 'WMAV','SSC','ZC','WA','WL',
                                'RMS','STD','SSI','VAR','AAC','MEAN'], 
                        help="特征提取方法, \
                        原始特征: row, \
                        时域特征：['MAV', 'WMAV','SSC','ZC','WA','WL','RMS','STD','SSI','VAR','AAC','EAN']")
    parser.add_argument('--window', type=int, default=200, help='滑动窗口长度')
    parser.add_argument('--stride', type=int, default=100, help='滑动窗口步长')

    parser.add_argument('--epochs', type=int, default=100, help='迭代次数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--optimzer', type=str, default='adam', help='优化器, 可选: adam, sgd')
    parser.add_argument('--cross_validation', default=5, help='是否采用交叉验证, <=1为不采用, >1采用')

    parser.add_argument('--save_action_detect_result', type=bool, default=True, help='是否保存活动段检测结果图')
    parser.add_argument('--save_processing_result', type=bool, default=False, help='是否保存信号处理结果图')
    parser.add_argument('--save_model', type=bool, default=True, help='是否保存最优模型')
    parser.add_argument('--save_train_result', type=bool, default=True, help='是否保存训练结果图')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', default=True, help='是否打印细节信息')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
