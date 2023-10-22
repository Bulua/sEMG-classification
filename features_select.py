import argparse
import torch

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from commons.runing import train

from data.builder import DatasetBuilder
from models.models import FeatureSelectNet, FeatureNet
from utils.ops import init_seeds, initialize_weights
from utils.dataset_util import split_dataset
from utils.prepare import load_trainer_param, prepare_trainer



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建数据集
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.builder()

    # 获取训练参数
    trainer_param = load_trainer_param()
    # 网络
    # model = FeatureSelectNet(
    #     input_shape=(trainer_param['batch_size'], dataset.shape[0], dataset.shape[1]),
    #     hidden_size=128,
    #     num_layers=4,
    #     bias=True,
    #     dropout=0.,
    #     classes=8
    # ).to(device)
    # ch, h, w = dataset.shape
    model = FeatureNet(input_shape=(trainer_param['batch_size'], 1, 200, 10),
                       classes=8).to(device)
    model.apply(initialize_weights)

    # print(model)

    # 优化器、学习率管理
    optimizer, lr_scheduler, loss_func = prepare_trainer(model, trainer_param)
    

    if args.cross_validation > 1:
        k_fold = KFold(n_splits=args.cross_validation)

        for train_idx, valid_idx in k_fold.split(dataset):
            train_set = Subset(dataset, train_idx)
            valid_set = Subset(dataset, valid_idx)
            # print(train_idx, valid_idx)
            print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

            train_dataloader = DataLoader(train_set, 
                                          batch_size=trainer_param['batch_size'],
                                          shuffle=True, 
                                          drop_last=True)
            valid_dataloader = DataLoader(valid_set, 
                                          batch_size=trainer_param['batch_size'],
                                          shuffle=True, 
                                          drop_last=True)

            # 开始训练
            train(model=model, 
                  train_dataloader=train_dataloader, 
                  valid_dataloader=valid_dataloader,
                  loss_func=loss_func,
                  epochs=args.epochs, 
                  optimizer=optimizer, 
                  lr_scheduler=lr_scheduler,
                  device=device)
    else:
        train_idx, valid_idx = split_dataset(len(dataset), split_rate=0.7)

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)
        print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

        train_dataloader = DataLoader(train_set, 
                                        batch_size=trainer_param['batch_size'],
                                        shuffle=True, 
                                        drop_last=True)
        valid_dataloader = DataLoader(valid_set, 
                                        batch_size=trainer_param['batch_size'],
                                        shuffle=True, 
                                        drop_last=True)

        # 开始训练
        train(model=model, 
              train_dataloader=train_dataloader, 
              valid_dataloader=valid_dataloader,
              loss_func=loss_func,
              epochs=trainer_param['epochs'], 
              optimizer=optimizer, 
              lr_scheduler=lr_scheduler,
              device=device)


def parse_opt():
    init_seeds()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='', help='网络模型')
    parser.add_argument('--subjects', type=list, 
                        default=['user1'], 
                        # default=['user1', 'user2', 'user3', 'user4', 'user5', 
                                # 'user6', 'user7', 'user8', 'user10'], 
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
                        default=['MAV', 'WMAV','SSC', 'WL', 'RMS','STD','SSI','VAR','AAC','MEAN'], 
                        help="特征提取方法, \
                        原始特征: row, \
                        时域特征：['MAV', 'WMAV','SSC','ZC','WA','WL','RMS','STD','SSI','VAR','AAC','EAN']")
    parser.add_argument('--window', type=int, default=200, help='滑动窗口长度')
    parser.add_argument('--stride', type=int, default=100, help='滑动窗口步长')
    parser.add_argument('--cross_validation', default=1, help='是否采用交叉验证, <=1为不采用, >1采用')

    parser.add_argument('--save_data', type=bool, default=False, help='是否保存features')
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