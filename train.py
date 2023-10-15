import argparse
import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from data.builder import DatasetBuilder
from utils.dataset_util import split_dataset


def prepare_trainer(model, params):
    optimizer_name = params['optim'].lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=params['lr'],
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
        )
    else:
        raise ValueError("未被支持的optimizer: {}".format(optimizer_name))
    
    lr_mode = params['lr_mode'].lower()
    if lr_mode == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=params['lr_decay_period'],
            gamma=params['lr_decay'],
            last_epoch=-1
        )
    elif lr_mode == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=params['epochs'],
            last_epoch=-1)
    elif lr_mode == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=params['lr_decay'],
            last_epoch=-1
        )
    else:
        raise ValueError("未被支持的lr_mode: {}".format(lr_mode))
    return optimizer, lr_scheduler

def train(net,
          optimizer,
          batch_size,
          train_dataloader, 
          valid_dataloader, 
          epochs,
          lr,
          device,
          ):
    
    pass


def load_trainer_param():
    params = {
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.001,
        'optimizer': 'adam',    # sgd、adam、adamw
        'lr_model': 'StepLR',   # ExponentialLR、StepLR、CosineAnnealingLR
        'weight_decay': 0.95,   
        'momentum': 0.9,
        'lr_decay_period': 20,  # 学习率衰减周期
    }
    return params


def main(args):
    # 构建数据集
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.builder()

    # 获取训练参数
    trainer_param = load_trainer_param()
    # 网络
    net = None
    # 优化器、学习率管理
    optimizer, lr_scheduler = prepare_trainer(net, trainer_param)

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
                        # default='row',
                        default=['MAV', 'WMAV','SSC','ZC','WA','WL', 'RMS','STD','SSI','VAR','AAC','MEAN'], 
                        help="特征提取方法, \
                        原始特征: row, \
                        时域特征：['MAV', 'WMAV','SSC','ZC','WA','WL','RMS','STD','SSI','VAR','AAC','EAN']")
    parser.add_argument('--window', type=int, default=200, help='滑动窗口长度')
    parser.add_argument('--stride', type=int, default=100, help='滑动窗口步长')
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
