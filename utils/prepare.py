import torch


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