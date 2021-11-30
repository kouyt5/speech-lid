from typing import Any, Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
import logging

from loggers.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir:str=None) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metrics_step_dict = {}
        logging.info("tensorboard init 成功！")
        
    def log(self, data:Dict[str, Any]
            ):
        """tensorboard 日志, 一个step只能调用一次

        Args:
            data 指标:数值
        """
        for key, value in data.items():
            step = self.get_global_step(key)
            self.writer.add_scalar(key, value, global_step=step)
    
    def get_global_step(self, name):
        """根据指标名字获取step数字

        Args:
            name 指标名字: 例如train/loss
        """
        if name not in self.metrics_step_dict.keys():
            self.metrics_step_dict[name] = 0
        # 用一次就加一
        self.metrics_step_dict[name] += 1
        return self.metrics_step_dict[name]
        
    def get_resume_state(self) -> Tuple[str, dict]:
        """获取储存的状态，用于恢复训练日志指标的状态
        """
        return "tensorboard", self.metrics_step_dict
    
    def resume_from(self, checkpoint:dict):
        """从checkpoint中恢复状态

        Args:
            checkpoint str: torch的checkpoint
        """
        if not isinstance(checkpoint, Dict):
            raise Exception("checkpoint must be a dict, please check")
        if "tensorboard" in checkpoint.keys():
            self.metrics_step_dict = checkpoint["tensorboard"]
        else:
            logging.warning("checkpoint 中无tensorboad的状态字典")
        logging.info("tensorboard resume from checkpoint, value={:s}".format(str(checkpoint)))
    
# if __name__=='__main__':
#     import time
#     import torch
#     logging.basicConfig(level=logging.DEBUG)
#     tb = TensorboardLogger(log_dir='outputs/tensorboard')
#     epoch = 0
#     checkpoint = torch.load('outputs/last.ckpt')
#     epoch = checkpoint['epoch']
#     tb.resume_from(checkpoint['tb_state'])
#     for i in range(epoch, 1000):
#         tb.log({'train/loss':i, 'val/loss': i/10})
#         time.sleep(0.1)
#         if i % 10 == 1:
#             tb.log({'epoch': 1})
#         if i % 100 == 1:
#             logging.info('save a file {:d}'.format(i))
#             torch.save({'epoch': i, 'tb_state':tb.get_resume_state()}, 'outputs/last.ckpt')
        