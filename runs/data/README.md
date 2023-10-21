目录结构：
data/
- user1
    - signal.npy    # 信号数据文件(tms, channel)
    - labels.npy    # 信号标签文件(tms, classes)
    - images.npy    # 信号图文件(n, window, channel)
    - image_labels.npy  # 信号图标签文件(n, classes)
    - feature_images.npy    # 特征文件(n, features, channel), features:['MAV', 'WMAV','SSC','ZC','WA','WL', 'RMS','STD','SSI','VAR','AAC','MEAN']
    - feature_image_labels.npy  # 特征标签文件(n, classes)