import torch


def load_trainer_param():
    params = {
        'epochs': 200,
        'batch_size': 32,
        'lr': 0.000001,
        # 'lr': 0.0001,
        'loss_f': 'CrossEntropyLoss',   # CrossEntropyLoss
        'optim': 'adamw',    # sgd、adam、adamw
        'lr_mode': 'StepLR',   # ExponentialLR、StepLR、CosineAnnealingLR
        'weight_decay': 0.90,   
        'momentum': 0.99,
        'lr_decay': 1e-8,
        'lr_decay_period': 100,  # 学习率衰减周期
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
    if lr_mode == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=params['lr_decay_period'],
            gamma=params['lr_decay'],
            last_epoch=-1
        )
    elif lr_mode == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=params['epochs'],
            last_epoch=-1)
    elif lr_mode == 'exponentiallr':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=params['lr_decay'],
            last_epoch=-1
        )
    else:
        raise ValueError("未被支持的lr_mode: {}".format(lr_mode))

    loss_f = params['loss_f'].lower()
    if loss_f == 'crossentropyloss':
        loss_func = torch.nn.CrossEntropyLoss()
    elif loss_f == 'bceloss':
        loss_func = torch.nn.BCELoss()
    else:
        raise ValueError("未被支持的loss_func: {}".format(loss_f))

    return optimizer, lr_scheduler, loss_func