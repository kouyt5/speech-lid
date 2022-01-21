from typing import Any, Dict, List

from ccml.loggers.base_logger import BaseLogger
from tqdm import tqdm
import torch
import logging

from ccml.loggers.wandb_logger import WandbLogger


class Logger:
    def __init__(self, rank: int = -1, interval:int = 1) -> None:
        self.rank = rank
        self.loggers = []  # logger集合

        self.global_tqdm_elements = {}  # 打印到tqdm的元素
        self.interval = interval
        self.global_step = 0
        
    def add_logger(self, logger: BaseLogger):
        """添加一个logger

        Args:
            logger ([type]): wandb, tensorboard等
        """
        if logger is not None and isinstance(logger, BaseLogger):
            self.loggers.append(logger)
            logging.info("添加一个logger")
        else:
            logging.warning("添加logger失败")
    def attach_trainer(self, trainer):
        self.trainer = trainer

    def log(
        self,
        data: Dict[str, Any] = None,  # 打印的数据
        progress: bool = False,  # 是否打印到进度条
        stage:str = "train",  # train, val
        only_tbar: bool = False,  #是否仅仅打印到控制台
        *args, **kwargs
    ):
        """打印日志接口

        日志分为两种：
            1. 在tqdm中
            2. 在tqdm外
        在tqdm中，如果需要实时查看value，需要将progress置为true，并且传入tbar变量
        如果不在tqdm中，此时没有tbar变量，如果选择需要输出到progress，将在下一个tqdm循环中看到这个value
        """
        if data is None:
            return
        self.global_step += 1
        # 打印频率
        if stage == "train" and self.global_step % self.interval != 0:
            return
        if not only_tbar:
            for logger in self.loggers:
                # if isinstance(logger, WandbLogger):
                #     logger.log({stage:data}, *args, **kwargs)
                logger.log(data, *args, **kwargs)

        # 打印到tqdm,并且限制rank为0的master节点才打印
        if progress and self.rank <= 0:
            for key, value in data.items():
                # 判断value值类型
                convert_value = 0.0
                if isinstance(value, float) or isinstance(value, int):
                    convert_value = value
                if isinstance(value, torch.Tensor):
                    try:
                        convert_value = value.detach().item()
                    except Exception as e:
                        logging.warning("Tensor cant convert to a float or int. ", e)
                self.global_tqdm_elements[key] = convert_value
            # print
            if self.trainer.tbar is not None:
                self.trainer.tbar.set_postfix(self.global_tqdm_elements)

    def watch_model(self, model: torch.nn.Module, *args, **kwargs):
        """观察模型

        Args:
            model (torch.nn.Module): pytorch的模型
        """
        if not self.rank == 0:
            return
        for logger in self.loggers:
            logger.watch_model(model, *args, **kwargs)

    def get_checkpoint_by_name(self, name, path):
        checkpoint_path = None
        for logger in self.loggers:
            if logger.get_checkpoint_by_name(name, path) is not None:
                checkpoint_path = logger.get_checkpoint_by_name(name, path)
                break
        return checkpoint_path

    def load_state_dict(self, state_dict: str = None):
        """从checkpoint中恢复状态

        Args:
            state_dict (str, optional): 状态字典. Defaults to None.
        """
        for logger in self.loggers:
            logger.resume_from(state_dict)

    def state_dict(self):
        """logger的状态字典"""
        checkpoint = {}
        for logger in self.loggers:
            if logger.get_resume_state()[0] is not None:
                checkpoint[logger.get_resume_state()[0]] = logger.get_resume_state()[1]
        return checkpoint

    def remove_key(self, keys:List):
        for key in keys:
            if key in self.global_tqdm_elements.keys():
                self.global_tqdm_elements.pop(key)

if __name__ == "__main__":
    import random
    import time

    logger = Logger()
    with tqdm(enumerate(range(100)), total=100, desc="训练中") as tbar:
        for i, batch in tbar:
            logger.log(
                {"loss": random.randint(1, 4), "acc": random.randint(1, 2)},
                progress=True,
                tbar=tbar,
            )
            time.sleep(0.5)
            logger.log(
                {"loss2": random.randint(1, 4), "acc2": random.randint(1, 2)},
                progress=True,
                tbar=tbar,
            )
