import numpy as np

from sklearn.model_selection import KFold


def split_dataset(size, split_rate=0.7):
    '''
        size: 数据集大小
        split_rate: 分割比例 0.8, 0.4 ...
        return: 训练下标, 验证下标
    '''
    idx = list(range(size))
    np.random.shuffle(idx)
    train_size = int(size * split_rate)
    return idx[:train_size], idx[train_size:]
