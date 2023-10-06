import numpy as np
'''
    feature: 
        平均绝对值MAV, 加权平均绝对值WMAV, 斜率符号变化
        过零点率ZC, 威利森幅值WA, 波形长度WL, 均方根RMS
        标准差STD, 简单方形积分SSI, 方差VAR, 平均幅度改变AAC
        均值MEAN
'''

def MAV(data):
    '''
        data.shape: (n, ch)
        return shape: (ch, )
    '''
    return np.mean(np.abs(data), axis=0)

def WMAV(data):
    n = data.shape[0]
    w = np.logical_and(0.25*n <= data, data <= 0.75*n)
    w = np.where(w, 1.0, 0.5)
    return np.mean(w * np.abs(data), axis=0)

def SSC(data, threshold=10e-7):
    delta = np.flip(np.diff(np.flip(data, axis=0), axis=0)[:-1], axis=0) * \
            np.diff(data, axis=0)[:-1]
    return np.sum(np.where(delta >= threshold, 1.0, 0.), axis=0)

def ZC(data, threshold=10e-7):
    abs_diff = np.abs(np.diff(data, axis=0))
    sign = np.diff(np.sign(data), axis=0)
    log = np.logical_and(sign != 0, abs_diff > threshold)
    return np.sum(log, axis=0)

def WA(data, threshold=10e-7):
    abs_diff = np.abs(np.diff(data, axis=0))
    return np.sum(np.where(abs_diff > threshold, 1.0, 0.), axis=0)

def WL(data):
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)

def RMS(data):
    return np.sqrt(np.mean(data**2, axis=0))

def STD(data):
    return np.std(data, axis=0)

def SSI(data):
    return np.sum(data**2, axis=0)

def VAR(data):
    return np.var(data, axis=0)

def AAC(data):
    return np.mean(np.diff(data, axis=0), axis=0)

def MEAN(data):
    return np.mean(data, axis=0)


if __name__ == '__main__':
    data = [[7,2,3,5],
            [1,8,4,3],
            [7,2,3,2]]
    data1 = np.array(data)
    data2 = np.array(data)
    # x = data1 * data2
    print(SSC(data1))