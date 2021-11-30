import logging
import os
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lid.loggers.base_logger import BaseLogger
from lid.loggers.logger import Logger
from lid.train_helper import init_model_optim
from train_helper import init_optim_scheduler, init_model, init_optim, resume_from_checkpoint


class Trainer:
    """
    训练器，用于pytorch的标准训练流程，相当于pytorch_lightning的Trainer
    """
    def __init__(self, 
                 total_epoch:int = 0,
                 world_size:int = 1,
                 local_rank:int = 0,
                 accumulate_grad:int = 1,  # 梯度累加
                 model:torch.nn.Module = None,
                 train_dataloader:DataLoader = None,
                 val_dataloader:DataLoader = None,
                 test_dataloader:DataLoader = None,
                 ddp: bool = False,  # 是否使用分布式
                 backend:str = 'gloo',
                 rank:int = 0, 
                 init_method:str = "env://",
                 master_addr:str = "localhost",
                 master_port:int = 11488,
                 use_amp:bool = False,  # 是否使用半精度训练
                 gpu_id:Optional[int] = None,  # GPU装置id，如果为None表示使用cpu训练
                 optim_name:str = "SGD", optim_dict:Optional[dict] = None,
                 lr_scheduler_name:str = None, lr_scheduler_dict:Optional[dict] = None,
                 checkpoint_path:str = None,
                 loggers:Optional[List[BaseLogger]] = None,
                 ) -> None:
        self.total_epoch = total_epoch
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_amp = use_amp
        
        assert model is None
        assert val_dataloader is None or train_dataloader is None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if ddp:
            if not torch.cuda.is_available():  # 如果cuda不可用，backend只能选择gloo
                backend = 'gloo'
                logging.warning(f"cuda不可用，无法使用nccl后端，默认gloo后端，使用cpu作分布式训练...")
            # 初始化ddp进程组
            if init_method == "env://":
                os.environ['MASTER_PORT'] = master_port
                os.environ['MASTER_ADDR'] = master_addr
            Trainer.init_ddp(backend=backend, rank=rank, init_method=init_method, world_size=world_size)
            logging.info(f"分布式初始化完成，world_size={dist.get_world_size()}, local_rank={dist.get_rank()}")
            
        # 初始化模型和优化器等
        self.model = init_model(model, ddp, gpu_id)
        self.optimizer = init_optim(self.model, optim_name, optim_dict)
        self.lr_scheduler = None
        self.sche_interval = None  #"step" or "epoch"
        self.sche_monitor = None  # "loss" or "acc" 由每一步返回值中的key决定
        if self.lr_scheduler_name is not None:  # 如果使用学习率调度器
            self.lr_scheduler, self.sche_interval, self.sche_monitor = init_optim_scheduler(
                self.optimizer, lr_scheduler_name, lr_scheduler_dict)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # 初始化日志工具，只有rank0的进程才能打印到控制台
        self.logger = Logger(rank=local_rank)
        for logger in loggers:
            self.logger.add_logger(logger)
        
        self.train_step = 0
        self.val_step = 0
        self.global_epoch = 0
        # 从checkpoint中resume当前训练状态
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                checkpoint_path = self.logger.get_checkpoint_by_name(checkpoint_path)
            if checkpoint_path is None:
                logging.error("checkpoint 无法在本地和logger中找到")
                raise Exception(f"resume失败，请检查checkpoint {checkpoint_path}")
            # model scaler optimizer epoch scheduler logger
            self.model, self.global_epoch, self.optimizer, self.scalar, self.lr_scheduler, self.logger = \
                resume_from_checkpoint(checkpoint_path, gpu_id, self.model, self.optimizer,
                                       self.scalar, self.lr_scheduler, self.logger)
        # watch model
        self.logger.watch_model(model=model)
        
        
        

        
    def train_loop(self):
        """
        训练循环
        """
        pass
    
    def eval_loop(self):
        """
        验证循环
        
        """
        pass
    
    def test_loop(self):
        """
        测试循环
        """
        pass
    
    def train(self):
        """
        整个流程的整合
        """
        for epoch in range(self.global_epoch, self.total_epoch):
            self.model.train()
            with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="训练中") as tbar:
                for i, batch in tbar:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        train_out = self.train_loop(batch)
                    self.scaler.scale(train_out['loss']).backward()
                    if i % self.accumulate_grad == self.accumulate_grad - 1:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20)
                        self.scaler.step(self.optimizer)
                        self.scalar.update()
                        self.optimizer.zero_grad()
                        
            
            self.model.eval()
            with tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc="测试中") as tbar:
                for i, batch in tbar:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        eval_out = self.eval_loop(batch)
            
            if self.sche_interval == "step":
                self.lr_scheduler.step(out[self.sche_monitor])
                    # after eval callback
            self.logger.log({'epoch': epoch}, progress=True)
    
    @staticmethod
    def init_ddp(backend:str = 'nccl', rank:int = 0, world_size:int = 1,
                 init_method:str = "env://"):
        """初始化分布式进程组
        参考: https://pytorch.org/docs/stable/distributed.html
            https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        Args:
            backend (str, optional): ddp后端，可选gloo，mpi，nccl(CUDA). Defaults to 'nccl'.
            rank (int, optional): rank master节点为0，不同机器可以相同. Defaults to 0.
            world_size (int, optional): 分布式训练节点大小. Defaults to 1.
            init_method str: 初始化方法，有"tcp://ip:port" "env://" "file://" 三种，建议env://,比较方便配置
                "env://", 需要指定rank world_size
                "tcp://ip:port", rank(必须是0对应tcp中的ip) world_size
                "file://", world_size
            
        """
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank,
                                init_method=init_method)
            


if __name__=='__main__':
    trainer = Trainer()
    trainer.train()