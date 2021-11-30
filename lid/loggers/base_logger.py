

from abc import abstractmethod
import torch.nn as nn


class BaseLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def get_resume_state(self):
        """获取储存的状态，用于恢复训练日志指标的状态
        """
        return None, None

    def resume_from(self, checkpoint:dict):
        """从checkpoint中恢复状态

        Args:
            checkpoint str: torch的checkpoint
        """
        pass
    
    @abstractmethod
    def log(self, *args, **kwargs):
        """日志接口
        """
        raise NotImplementedError()

    def save(self, path):
        """保存checkpoint到logger

        Args:
            path str: checkpoint的路径 
        """
        pass
    
    def watch_model(self, model:nn.Module, *args, **kwargs):
        """打印模型信息
        仅仅rank 0 进程watch

        Args:
            model (nn.Module): 一个pytorch模型
        """
        pass
    
    def get_checkpoint_by_name(self, name:str, path:str = None):
        """根据名字获取checkpoint路径，适配wandb

        Args:
            name (str): 文件名
        """
        return None