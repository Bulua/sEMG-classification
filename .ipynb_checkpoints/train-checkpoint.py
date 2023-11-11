import argparse
import random
import torch
import yaml
import numpy as np
from data.semg_datasets import ACCFeatureDataset, SemgFeatureDataset

from models.models import Net
from commons.runing import train

from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from data.builder import DatasetBuilder
from utils.dataset_util import split_dataset
from utils.ops import initialize_weights, init_seeds, Profile
from utils.prepare import load_trainer_param, prepare_trainer


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subject = args.subjects[0] if len(args.subjects) == 1 else '{}-{}'.format(args.subjects[0], args.subjects[-1])

    # 构建数据集
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.builder()
    semg_dataset = SemgFeatureDataset(
        images=dataset.feature_images,
        labels=dataset.feature_image_labels,
        selected_features=['WL', 'MEAN', 'SSC', 'WMAV', 'STD'],
        standard=True

    )
    acc_dataset = ACCFeatureDataset(
        images=dataset.feature_images,
        labels=dataset.feature_image_labels,
        selected_features=['WMAV', 'STD', 'SSI', 'MEAN'],
        standard=True
    )
    features = torch.from_numpy(
        np.concatenate([semg_dataset.features, acc_dataset.features], axis=-1),
    ).float()

    labels = torch.from_numpy(dataset.feature_image_labels).long()
    dataset = TensorDataset(features, labels)

    # 获取训练参数
    trainer_param = load_trainer_param()
    # 网络
    with open('configs/net.yaml', 'r') as f:
        net_args = yaml.safe_load(f)
    net = Net((trainer_param['batch_size'], 1, 25+12), net_args).to(device)

    net.apply(initialize_weights)
    # 优化器、学习率管理
    optimizer, lr_scheduler, loss_func = prepare_trainer(net, trainer_param)

    if args.cross_validation > 1:
        histories = []
        profiles = [Profile() for i in range(args.cross_validation)]
        k_fold = KFold(n_splits=args.cross_validation,
                       shuffle=True, 
                       random_state=0)
        
        for i, (train_idx, valid_idx) in enumerate(k_fold.split(dataset)):
            train_set = Subset(dataset, train_idx)
            valid_set = Subset(dataset, valid_idx)
            # print(train_idx, valid_idx)
            print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

            train_dataloader = DataLoader(train_set, batch_size=trainer_param['batch_size'])
            valid_dataloader = DataLoader(valid_set, batch_size=trainer_param['batch_size'])
            subject_ = '%s_cv%d_%d' % (subject, args.cross_validation, i)
            # 开始训练
            with profiles[i]:
                loss_acc_history = train(subject=subject_,
                                         model=net, 
                                         train_dataloader=train_dataloader, 
                                         valid_dataloader=valid_dataloader, 
                                         loss_func=loss_func,
                                         epochs=trainer_param['epochs'], 
                                         optimizer=optimizer,
                                         lr_scheduler=lr_scheduler,
                                         device=device)
            histories.append(loss_acc_history)
        valid_accs = [np.max(h.valid_acc) for h in histories]
        print(profiles)
        print(f'{args.cross_validation} 折交叉验证的平均准确率为: {round(np.mean(valid_accs) * 100, 2)}%')
    else:
        profile = Profile()
        train_idx, valid_idx = split_dataset(len(dataset), split_rate=0.7)
        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)
        print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

        train_dataloader = DataLoader(train_set, batch_size=trainer_param['batch_size'], drop_last=True)
        valid_dataloader = DataLoader(valid_set, batch_size=trainer_param['batch_size'], drop_last=True)

        # 开始训练
        with profile:
            loss_acc_history = train(subject=subject,
                                     model=net, 
                                     train_dataloader=train_dataloader, 
                                     valid_dataloader=valid_dataloader, 
                                     loss_func=loss_func,
                                     epochs=trainer_param['epochs'], 
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler,
                                     device=device)
        loss_acc_history.plot_and_save(save_name=f'{subject}.jpg', dpi=150)
        print(profile)
        print(f'随机验证的准确率为: {round(np.max(loss_acc_history.valid_acc) * 100, 2)}%')


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
                        default=['MAV', 'WMAV', 'SSC', 'WL', 'RMS','STD', 'SSI', 'VAR', 'AAC', 'MEAN'], 
                        help="特征提取方法, \
                        原始特征: row, \
                        时域特征：['MAV', 'WMAV', 'SSC', 'WL', 'RMS','STD', 'SSI', 'VAR', 'AAC', 'MEAN']")
    parser.add_argument('--window', type=int, default=200, help='滑动窗口长度')
    parser.add_argument('--stride', type=int, default=100, help='滑动窗口步长')
    parser.add_argument('--cross_validation', default=4, help='是否采用交叉验证, <=1为不采用, >1采用')

    parser.add_argument('--save_data', type=bool, default=False, help='是否保存features')
    parser.add_argument('--save_action_detect_result', type=bool, default=True, help='是否保存活动段检测结果图')
    parser.add_argument('--save_processing_result', type=bool, default=True, help='是否保存信号处理结果图')
    parser.add_argument('--save_model', type=bool, default=True, help='是否保存最优模型')
    parser.add_argument('--save_train_result', type=bool, default=True, help='是否保存训练结果图')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', default=True, help='是否打印细节信息')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
