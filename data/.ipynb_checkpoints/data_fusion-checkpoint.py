import numpy as np

from tqdm import tqdm
from filterpy.kalman import KalmanFilter


class DataFusion:
    
    def __init__(self, data):
        '''
        semg: (tms, 5)
        acc:  (tms, 3)
        '''
        self.data = data
    
    def fusion(self, mode='kalman'):
        if mode == 'kalman':
            return self.kalman_fusion()
        elif mode == '':
            pass
        return self.kalman_fusion()
    
    def kalman_fusion(self):
        data = self.data
        tms, ch = data.shape
        kf = KalmanFilter(dim_x=ch, dim_z=ch)
        kf.x = np.zeros(ch)   # 初始化状态估计值
        kf.P *= 1e-2                       # 初始化协方差矩阵
        # 定义系统动力学矩阵
        kf.F = np.eye(ch)   # 单位矩阵，表示状态不发生变化
        # 定义测量矩阵
        kf.H = np.eye(ch)   # 单位矩阵，表示测量结果中包含8个模态的信号
        # 定义过程噪声和测量噪声的协方差矩阵
        kf.Q *= 1e-5
        kf.R *= 0.01
        
        # 对每个采样点进行卡尔曼滤波
        filtered_states = []
        for measurement in tqdm(data, desc="kalman fusion", total=tms, leave=True):
            kf.predict()
            kf.update(measurement)
            filtered_states.append(np.copy(kf.x))
        filtered_states = np.array(filtered_states)
        return filtered_states