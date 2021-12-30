from abc import abstractmethod
from collections import OrderedDict
from typing import Any, List, Tuple
import torch
from torch.cuda import device
from torch.functional import Tensor
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from ccml.trainer import Trainer

# 去除继承自nn.Module，考虑加载多个DDP模型封装这个类的方法来实现DDP。
# 如果失败，再考虑其他
class CCMLModule:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hyper_parameters = {}  # 模型初始化的超参数
        self.model = None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.dataloader_param = {}

    def get_model(self) -> nn.Module:
        """获取到nn.Module模型

        Raises:
            NotImplementedError: 未实现错误

        Returns:
            nn.Module: 模型
        """
        return self.model

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def point_trainer(self, trainer: Trainer = None):
        """引用trainer对象，在AgentModel初始化后必须由Trainer引用

        Args:
            trainer (Trainer, optional): Trainer对象. Defaults to None.
        """
        self.trainer = trainer

    def save_hyper_parameters(self, hyper_parameters: dict):
        self.hyper_parameters = hyper_parameters

    def get_hyper_parameters(self):
        """获取模型超参数，用于模型的初始化

        Returns:
            [type]: 超参数字典
        """
        return self.hyper_parameters

    @abstractmethod
    def forward(self, *args, **kwargs):
        """代理模型的前向传播层

        Raises:
            NotImplementedError: 必须实现
        """
        raise NotImplementedError()

    def config_datasource(self):
        """配置数据集，定义Datalset
            例如:
                self.train_dataset = Dataset(...)
                self.val_dataset = Dataset(...)
                self.test_dataset = Dataset(...)
            也可以不设置，在trainer.fit的时候指定数据集，如果每个数据长度不同，
            需要实现collate_fn方法，以便进行序列对齐

        """
        pass

    @abstractmethod
    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        """初始化优化器和学习率调度器，在数据初始化完成后调用, 如果不使用学习率调度器，返回optimizer, None, None

        Returns:
            dict: 包含optimizer、scheduler（可选）、monitor（表示是什么决定调度）、interval（step或epoch）
        """
        total_steps = self.trainer.total_steps
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            cooldown=3,
            min_lr=1e-5,
        )
        return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}

    @abstractmethod
    def train_loop(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def val_loop(self, batch):
        raise NotImplementedError()

    def test_loop(self, batch):
        raise NotImplementedError()

    def train_loop_end(self, outputs: List[Any] = None):
        """当训练结束时候调用

        Args:
            outputs (List[Any], optional): 训练loop的结果集. Defaults to None.
        """
        pass

    def val_loop_end(self, outputs: List[Any] = None):
        """当验证结束时候调用

        Args:
            outputs (List[Any], optional): 验证loop的结果集. Defaults to None.
        """
        pass

    def test_loop_end(self, outputs: List[Any] = None):
        """当测试结束时候调用

        Args:
            outputs (List[Any], optional): 测试loop的结果集. Defaults to None.
        """
        pass

    @staticmethod
    def resume_from_checkpoint(checkpont: str, map_location: str) -> nn.Module:
        """从checkpoint恢复模型状态

        Args:
            checkpont (str): 模型路径
            map_location (str): 'cpu', 'cuda:0'等

        Returns:
            [type]: agentModel
        """
        device = torch.device(map_location)
        stat_dicts = torch.load(checkpont, device)
        hyper_parameters = {}
        if "hyper_parameters" in stat_dicts.keys():
            hyper_parameters = stat_dicts["hyper_parameters"]
        agent_model = CCMLModule(**hyper_parameters).to(device)
        agent_model.get_model().load_state_dict(state_dict=stat_dicts["model"])
        return agent_model
