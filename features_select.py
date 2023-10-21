import argparse
import torch

from torch.utils.data import Subset, DataLoader

from data.builder import DatasetBuilder
from models.feature_selector import FeatureSelectNet
from utils.ops import init_seeds, initialize_weights
from utils.dataset_util import split_dataset
from utils.prepare import load_trainer_param, prepare_trainer



def main(args):
    # 构建数据集
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.builder()

    # 获取训练参数
    trainer_param = load_trainer_param()
    # 网络
    net = FeatureSelectNet(
        input_shape=input_shape,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        dropout=0.5,
        classes=8
    )
    net.apply(initialize_weights)
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
    parser.add_argument('--features', 
                        # default='row',
                        default=['MAV', 'WMAV','SSC','ZC','WA','WL', 'RMS','STD','SSI','VAR','AAC','MEAN'], 
                        help="特征提取方法, \
                        原始特征: row, \
                        时域特征：['MAV', 'WMAV','SSC','ZC','WA','WL','RMS','STD','SSI','VAR','AAC','EAN']")
    parser.add_argument('--cross_validation', default=5, help='是否采用交叉验证, <=1为不采用, >1采用')

    parser.add_argument('--save_model', type=bool, default=True, help='是否保存最优模型')
    parser.add_argument('--save_train_result', type=bool, default=True, help='是否保存训练结果图')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', default=True, help='是否打印细节信息')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)