import argparse
from os.path import join
import torch
import numpy as np
from data.semg_datasets import ACCFeatureDataset, SemgFeatureDataset

from models.models import Net
from commons.runing import evaluate

from torch.utils.data import Subset, DataLoader, TensorDataset
from utils.dataset_util import split_dataset
from utils.ops import init_seeds, Profile
from utils.path_util import DATA_PATH, MODELS_PATH
from utils.prepare import load_trainer_param, prepare_trainer



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    feature_images = []
    feature_image_labels = []
    for s in args.subjects:
        feature_images.append(np.load(join(DATA_PATH, s, 'feature_images.npy')))
        feature_image_labels.append(np.load(join(DATA_PATH, s, 'feature_image_labels.npy')))

    feature_images = np.concatenate(feature_images, axis=0)
    feature_image_labels = np.concatenate(feature_image_labels, axis=0)

    semg_dataset = SemgFeatureDataset(
        images=feature_images,
        labels=feature_image_labels,
        selected_features=['WL', 'MEAN', 'SSC', 'WMAV', 'STD'],
        standard=True
    )
    acc_dataset = ACCFeatureDataset(
        images=feature_images,
        labels=feature_image_labels,
        selected_features=['WMAV', 'STD', 'SSI', 'MEAN'],
        standard=True
    )
    features = torch.from_numpy(
        np.concatenate([semg_dataset.features, acc_dataset.features], axis=-1),
    ).float()

    labels = torch.from_numpy(feature_image_labels).long()
    dataset = TensorDataset(features, labels)
    
    train_idx, valid_idx = split_dataset(len(dataset), split_rate=0.7)
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    print('训练集大小: {}, 验证集大小: {}'.format(len(train_set), len(valid_set)))

    # 获取训练参数
    trainer_param = load_trainer_param()

    train_dataloader = DataLoader(train_set, batch_size=trainer_param['batch_size'], drop_last=True)
    valid_dataloader = DataLoader(valid_set, batch_size=trainer_param['batch_size'], drop_last=True)

    # 网络
    # 查看models下的pt文件名，选择当前所选user的pt文件
    net_pt = join(MODELS_PATH, args.model)
    net = torch.load(net_pt).to(device)

    # 优化器、学习率管理
    _, _, loss_func = prepare_trainer(net, trainer_param)
    net.eval()

    profile = Profile()
    with profile:
        valid_loss, valid_acc = evaluate(model=net,
                                        dataloader=valid_dataloader,
                                        loss_func=loss_func,
                                        epoch='evaluate epoch',
                                        device=device)
    print(f'valid acc: {round(valid_acc * 100, 2)} %, valid loss: {round(valid_loss, 4)}')
    print(profile)


def parse_opt():
    init_seeds()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='user1_97.34.pt', help='网络模型')
    parser.add_argument('--subjects', type=list, 
                        default=['user1'], 
                        # default=['user1', 'user2', 'user3', 'user4', 'user5', 
                                # 'user6', 'user7', 'user8', 'user10'], 
                        help='用户')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

    