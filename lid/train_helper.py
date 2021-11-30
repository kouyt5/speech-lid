from typing import Optional, Tuple
import torch
import logging

import torch
from lid.loggers.logger import Logger
from optim.novograd import Novograd
from optim.cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def init_optim_scheduler(optimizer: torch.optim.Optimizer,
                         lr_scheduler_name:str = None,
                         lr_scheduler_dict:Optional[dict]=None) -> Tuple[torch.optim.lr_scheduler._LRScheduler, str, str]:

    # 学习率调度器
    step = "step"  # 学习率调度按照step还是epoch进行调度
    monitor = None
    lr_schedulers = {
        "ReduceLROnPlateau": {"class": torch.optim.lr_scheduler.ReduceLROnPlateau, "step": "epoch", "monitor": "loss"},
        "CosineAnnealingWarmupRestarts": {"class": CosineAnnealingWarmupRestarts, "step": "step", "monitor": None},
        "None": None
    }
    if lr_scheduler_name not in lr_schedulers.keys() or lr_scheduler_name == "None":
        lr_scheduler = None
    else:
        lr_scheduler = lr_schedulers[lr_scheduler_name]["class"](optimizer, **lr_scheduler_dict)
        step = lr_schedulers[lr_scheduler_name]["step"]
    return lr_scheduler, step, monitor


def init_model(model:torch.nn.Module = None, ddp:bool = False, gpu:Optional[int]=None) -> torch.nn.Module:
    """初始化模型，根据策略送入到gpu或cpu中

    Args:
        model (torch.nn.Module, optional): 模型. Defaults to None.
        ddp (bool, optional): 是否是分布式. Defaults to False.
        gpu (Optional[int], optional): gpu编号. Defaults to None.

    Returns:
        torch.nn.Module: [description]
    """
    #  模型初始化
    if gpu is None:
        logging.warning(f"使用cpu训练，可能会导致训练时间很长...")  # 默认放到cpu上
    else:
        if not torch.cuda.is_available():
            logging.error(f"cuda不可用，使用cpu训练")
            raise Exception("cuda不可用,考虑使用cpu训练")
        model = model.to(torch.device(gpu))
        logging.debug(f"模型放到gpu{gpu}上训练")
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, gpu)
        
    return model


def init_optim(model: torch.nn.Module, 
               optim_name:str = "SGD",
                optim_dict:dict=None) -> torch.optim.Optimizer:
    """初始化优化器

    Args:
        model (torch.nn.Module): 模型
        optim_name (str, optional): 优化器名字，支持SGD Adam AdamW Adadelta NovoGrad.

    Returns:
        torch.optim.Optimizer: 优化器
    """
    # 优化器配置
    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "Adadelta": torch.optim.Adadelta,
        "NovoGrad": Novograd
    }
    if optim_name not in optimizers.keys():
        raise Exception(f"优化器{optim_name}不支持")
    return optimizers[optim_name](model.parameters(), **optim_dict)

def resume_from_checkpoint(checkpoint_path:str=None, gpu_id:Optional[int] = None,
                           model:torch.nn.Module = None,
                           optimizer:torch.optim.Optimizer=None,
                           scalar:torch.cuda.amp.GradScaler=None,
                           lr_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
                           logger:Logger=None) -> Tuple[torch.nn.Module, int, torch.optim.Optimizer,
                                                        torch.cuda.amp.GradScaler, torch.optim.lr_scheduler._LRScheduler,
                                                        Logger]:
    """从checkpoint中恢复模型训练状态

    Args:
        checkpoint_path (str, optional): checkpoint路径. Defaults to None.
        gpu_id (Optional[int], optional): gpu id如果为None 表示在cpu上. Defaults to None.
        model (torch.nn.Module, optional): 模型. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): 优化器. Defaults to None.
        scalar (torch.cuda.amp.GradScaler, optional): 混合精度训练. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器. Defaults to None.
        logger (Logger, optional): 日志工具. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, int, torch.optim.Optimizer, torch.cuda.amp.GradScaler, torch.optim.lr_scheduler._LRScheduler, Logger]: [description]
    """
    device = torch.device("cuda:"+str(gpu_id)) if isinstance(gpu_id, int) else torch.device("cpu")
    state_dicts = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dicts['model'])
    epoch = state_dicts['epoch']
    optimizer.load_state_dict(state_dicts['optimizer'])

    if scalar is not None:
        scalar.load_state_dict(state_dicts['scalar'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])
    logger.load_state_dict(state_dicts['logger'])
    return model, epoch, optimizer, scalar, lr_scheduler, logger