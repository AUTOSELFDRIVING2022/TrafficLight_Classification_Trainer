import torch.optim as optim
from torch.optim import lr_scheduler

from adamp import AdamP
from adabelief_pytorch import AdaBelief

def get_optimizer(cfg, model):
    if cfg.optimizer.name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=float(cfg.optimizer.wd))
    elif cfg.optimizer.name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, betas=(0.9, 0.999), weight_decay=cfg.optimizer.wd)
    elif cfg.optimizer.name == 'adamp':
        optimizer = AdamP(model.parameters(), lr=cfg.optimizer.lr, betas=(0.9, 0.999), weight_decay=cfg.optimizer.wd)
    elif cfg.optimizer.name == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=cfg.optimizer.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple = True, rectify = False)
    elif cfg.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=0.9, weight_decay=cfg.optimizer.wd)
    else:
        raise NameError('Choose proper optimizer!!!')
    return optimizer

def get_scheduler(cfg, optimizer):
    _T_max         = int(30000/cfg.train_config.train_bs*cfg.train_config.epochs)+50
    _T_0           = 25
    
    if cfg.optimizer.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = _T_max, 
                                                   eta_min=cfg.optimizer.min_lr)
    elif cfg.optimizer.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = _T_0, 
                                                             eta_min=cfg.optimizer.min_lr)
    elif cfg.optimizer.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.optimizer.min_lr.min_lr,)
    elif cfg.optimizer.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    else:
        raise NameError('Choose proper scheduler!!!')
    return scheduler