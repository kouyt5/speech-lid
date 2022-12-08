import contextlib
import logging
import os
import time
from typing import Any, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from tqdm import tqdm
import torch.distributed as dist
from ccml.loggers.base_logger import BaseLogger
from ccml.loggers.logger import Logger
from ccml.utils.profile import register_cost_statistic
from ccml.utils.profile import _time_cost_recoder


class Trainer:
    """
    训练器，用于pytorch的标准训练流程，相当于pytorch_lightning的Trainer
    """

    def __init__(
        self,
        total_epoch: int = 0,
        world_size: int = 1,
        local_rank: int = -1,
        accumulate_grad: int = 1,  # 梯度累加
        eval_interval: int = 1,
        train_data_factor: float = 1.0,  # 训练放缩因子
        ddp: bool = False,  # 是否使用分布式
        backend: str = "gloo",
        init_method: str = "env://",
        master_addr: str = "localhost",
        master_port: str = "11488",
        use_amp: bool = False,  # 是否使用半精度训练
        gpu_id: Optional[int] = None,  # GPU装置id，如果为None表示使用cpu训练
        checkpoint_path: str = None,
        callbacks: List[Any] = [],
        resume_train_states: bool = True,  # 恢复时是否resume训练状态，包括优化器等等，测试时为False
        loggers: Optional[List[BaseLogger]] = [],
        log_interval: int = 1,
        use_swa: bool = False,  # 梯度平均
        swa_config: Tuple[float, float] = (0.1, 0.1),  # (加权系数，swa最后轮数比例)
    ) -> None:
        self.total_epoch = total_epoch
        self.eval_interval = eval_interval
        self.accumulate_grad = accumulate_grad
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_amp = use_amp
        self.gpu_id = gpu_id
        self.train_data_factor = train_data_factor
        self.ddp = ddp
        self.resume_train_states = resume_train_states
        self.checkpoint_path = checkpoint_path
        self.use_swa = use_swa
        self.swa_config = swa_config
        # assert train_dataset is None or val_dataset is None
        logging.debug(f"justlfy ddp cuda {torch.cuda.is_available()}")
        if ddp:
            if not torch.cuda.is_available():  # 如果cuda不可用，backend只能选择gloo
                backend = "gloo"
                logging.warning(f"cuda不可用，无法使用nccl后端，默认gloo后端，使用cpu作分布式训练...")
            # 初始化ddp进程组
            if init_method == "env://":
                os.environ["MASTER_PORT"] = master_port
                os.environ["MASTER_ADDR"] = master_addr
                logging.info(f"master addr {master_addr}")
            if init_method == "tcp://":
                init_method += master_addr + ":" + master_port
                logging.info(f"init_method={init_method}")
            logging.info(f"rank {self.local_rank} wait other process to join...")
            self.init_ddp(
                backend=backend,
                rank=self.local_rank,
                init_method=init_method,
                world_size=world_size,
            )
            logging.info(
                f"分布式初始化完成，world_size={dist.get_world_size()}, local_rank={dist.get_rank()}"
            )

        # trainer创造的参数
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self.train_batch_sampler = None
        self.val_batch_sampler = None
        self.test_batch_sampler = None
        self.ccml_module = None
        self.training = False  # 是否处于训练状态
        self.tbar = None  # 日志tqdm bar
        self.sche_interval = None  # "step" or "epoch"
        self.sche_monitor = None  # "loss" or "acc" 由每一步返回值中的key决定
        self.total_steps = 0
        self.current_epoch = 0
        self.current_step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.scheduler_param = None
        self.model = None
        self.callbacks = callbacks
        self.dataloader_params = {}
        if self.gpu_id is not None:
            assert torch.cuda.is_available()
        self.device = (
            torch.device("cuda:" + str(self.gpu_id))
            if self.gpu_id is not None
            else torch.device("cpu")
        )

        # 混合精度训练
        self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 初始化日志工具，只有rank0的进程才能打印到控制台
        if ddp:
            self.local_rank = dist.get_rank()
        self.logger = Logger(rank=self.local_rank, interval=log_interval)
        self.logger.attach_trainer(self)
        for logger in loggers:
            self.logger.add_logger(logger)

        # 给回调接口添加trainer对象
        for callback in callbacks:
            callback.add_trainer(self)

    def trainer_prepare(self):
        if self.ccml_module is None:
            raise Exception("模型为None")
        # 将模型放到对应的gpu或者cpu上
        self.model = self.ccml_module.get_model()
        # self.model = self.model.to(torch.device("cuda:"+str(self.gpu_id))) if self.gpu_id is not None else self.model
        # 初始化模型,加载到正确的gpu中和分布式封装
        self.model, self.swa_model = self.init_model(
            self.model, self.ddp, self.gpu_id, self.use_swa, self.swa_config[0]
        )
        logging.debug("model init done")
        self.training = (
            self.train_dataset is not None and self.val_dataset is not None
        )  # 是否在训练还是测试
        # 初始化数据加载器
        self.init_dataloader(ddp=self.ddp, **self.dataloader_params)
        if not self.training:  # 如果是非训练模式
            if self.resume_train_states:
                logging.warning(
                    f"resume_train_states is {self.resume_train_states}, under the test mode, it will be set to false"
                )
                self.resume_train_states = False

            # 分布式的模型能否加载到非分布式模型？按理说应该可以
            # cpu上面初始化的模型能否加载gpu上初始化的模型？
            self.model, self.current_epoch, _, _, _, _ = self.resume_from_checkpoint(
                self.checkpoint_path, self.resume_train_states, self.gpu_id, self.model
            )
            return  # 测试部分，只需要模型参数
        self.total_steps = (
            len(self.train_dataloader) / self.accumulate_grad
        ) * self.total_epoch  # 总的训练步数
        # 初始化优化器
        (
            self.optimizer,
            self.lr_scheduler,
            self.scheduler_param,
        ) = self.ccml_module.config_optim()
        # 学习率调度器参数
        if self.scheduler_param is not None:
            self.sche_interval = self.scheduler_param["interval"]
            self.sche_monitor = self.scheduler_param["monitor"]
        # 从checkpoint中resume当前训练状态
        if self.checkpoint_path is not None:
            if not os.path.exists(self.checkpoint_path):
                logging.error("checkpoint 无法在本地和logger中找到")
                raise Exception(f"resume失败，请检查checkpoint {self.checkpoint_path}")
            # model scaler optimizer epoch scheduler logger
            (
                self.model,
                self.current_epoch,
                self.optimizer,
                self.scalar,
                self.lr_scheduler,
                self.logger,
            ) = self.resume_from_checkpoint(
                self.checkpoint_path,
                self.resume_train_states,
                self.gpu_id,
                self.model,
                self.optimizer,
                self.scalar,
                self.lr_scheduler,
                self.logger,
            )
        self.current_step = self.current_epoch * int(
            len(self.train_dataloader) / self.accumulate_grad
        )
        # watch model
        self.logger.watch_model(model=self.model)

    def train_loop(self, batch: Any = None):
        """
        训练循环
        """
        return self.ccml_module.train_loop(batch)

    def before_train_loop(self, value):
        """
        训练开始回调
        """
        return self.ccml_module.before_train_loop(value)

    def train_loop_end(self, outputs: List[Any]):
        return self.ccml_module.train_loop_end(outputs)

    def eval_loop(self, batch: Any = None):
        """
        验证循环
        """
        return self.ccml_module.val_loop(batch)

    def eval_loop_end(self, outputs: List[Any]):
        return self.ccml_module.val_loop_end(outputs)

    def test_loop(self, batch: Any = None):
        """
        测试循环
        """
        return self.ccml_module.test_loop(batch)

    def test_loop_end(self, outputs: List[Any]):
        return self.ccml_module.test_loop_end(outputs)

    # TODO 必须在Trainer之前调用初始化数据集，因为必须在初始化优化器之前知道总的step有多少
    # dataloader 可以不变，dataset可以动态的改变
    def init_dataloader(
        self,
        ddp: bool = False,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        pin_memory: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        train_sampler: Sampler = None,
        val_sampler: Sampler = None,
        test_sampler: Sampler = None,
        train_batch_sampler: Sampler = None,
        val_batch_sampler: Sampler = None,
        test_batch_sampler: Sampler = None,
    ):
        """初始化数据加载器，对train_dataloader等做封装

        Args:
            train_datasets (Dataset, optional): 训练集. Defaults to None.
            val_datasets (Dataset, optional): 验证集. Defaults to None.
            test_datasets (Dataset, optional): 测试集. Defaults to None.
            ddp (bool, optional): 是否分布式. Defaults to False.
            train_sampler (Sampler, optional): 训练集采样器
            val_sampler (Sampler, optional): 验证集采样器
            test_sampler (Sampler, optional): 测试集采样器
        """
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.train_batch_sampler = train_batch_sampler
        self.val_batch_sampler = val_batch_sampler
        self.test_batch_sampler = test_batch_sampler
        # 如果分布式并且处于训练状态
        if ddp and self.training:
            if self.train_sampler is not None or train_batch_sampler is not None:
                logging.warning(f"train sampler is not None, ddp train may not work correctly...")
            self.train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.val_sampler = DistributedSampler(dataset=self.val_dataset)

        if self.training:

            train_collate_fn = (
                self.train_dataset.collate_fn
                if hasattr(self.train_dataset, "collate_fn")
                else None
            )
            val_collate_fn = (
                self.val_dataset.collate_fn
                if hasattr(self.val_dataset, "collate_fn")
                else None
            )
            if train_batch_sampler is not None:
                self.train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_sampler=train_batch_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=train_collate_fn,
                    prefetch_factor=prefetch_factor,
                )
            else:
                self.train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=train_batch_size,
                    sampler=self.train_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=(self.train_sampler is None),
                    drop_last=True,
                    collate_fn=train_collate_fn,
                    prefetch_factor=prefetch_factor,
                )
            if val_batch_sampler is not None:
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    batch_sampler=val_batch_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=train_collate_fn,
                    prefetch_factor=prefetch_factor,
                )
            else:
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    batch_size=train_batch_size,
                    sampler=self.val_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    collate_fn=val_collate_fn,
                    prefetch_factor=prefetch_factor,
                )
        test_collate_fn = (
            self.test_dataset.collate_fn
            if hasattr(self.test_dataset, "collate_fn")
            else None
        )
        if test_batch_sampler is not None:
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_sampler=test_batch_sampler,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=test_collate_fn,
            )
        else:
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=val_batch_size,
                sampler=self.test_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=test_collate_fn,
            )

    def init_ddp(
        self,
        backend: str = "nccl",
        rank: int = 0,
        world_size: int = 1,
        init_method: str = "env://",
    ):
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
        dist.init_process_group(
            backend=backend, world_size=world_size, rank=rank, init_method=init_method
        )

    def init_model(
        self,
        model: torch.nn.Module = None,
        ddp: bool = False,
        gpu: Optional[int] = None,
        use_swa: bool = False,
        swa_w: float = 0.1,
    ) -> torch.nn.Module:
        """初始化模型，根据策略送入到gpu或cpu中

        Args:
            model (AgentModel, optional): 模型. Defaults to None.
            ddp (bool, optional): 是否是分布式. Defaults to False.
            gpu (Optional[int], optional): gpu编号. Defaults to None.
            use_swa: bool: 是否使用模型平均训练,
            swa_w int: 模型平均权重参数，对以前epoch的权重

        Returns:
            torch.nn.Module: 模型
            torch.nn.Module: 平均模型
        """
        #  模型初始化
        device = torch.device("cpu")
        if gpu is None:
            logging.warning(f"使用cpu训练，可能会导致训练时间很长...")  # 默认放到cpu上
            if ddp:
                raise RuntimeError("非gpu环境不支持分布式训练")
        else:
            device = torch.device("cuda:" + str(gpu))
            if not torch.cuda.is_available():
                logging.error(f"cuda不可用，使用cpu训练")
                raise Exception("cuda不可用,考虑使用cpu训练")
            model = model.to(device)
            logging.debug(f"模型放到gpu{gpu}上训练")
        # swa
        swa_model = None

        if use_swa:
            ema_avg = (
                lambda avg_model_params, model_params, num_avgraged: swa_w
                * avg_model_params
                + (1 - swa_w) * model_params
            )
            swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        if ddp:
            # BN同步
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.debug("sync bn finish")
            model = torch.nn.parallel.DistributedDataParallel(model)
            logging.debug("parallel model finish")
            from torch.distributed.algorithms.ddp_comm_hooks import default

            model.register_comm_hook(
                state=None, hook=default.fp16_compress_hook
            )  # 默认使用分布式参数压缩
        return model, swa_model

    def fit(
        self,
        ccml_module=None,
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None,
        dataloader_params: dict = None,
    ):
        """
        整个流程的整合
        """
        logging.debug("fit prepare...")
        self.ccml_module = ccml_module
        if train_dataset is not None:  # 如果fit的时候指定数据集那么就使用这个数据集
            self.train_dataset = train_dataset
        if val_dataset is not None:
            self.val_dataset = val_dataset
        if test_dataset is not None:
            self.test_dataset = test_dataset
        if dataloader_params is None:
            dataloader_params = self.ccml_module.dataloader_param
        self.dataloader_params = dataloader_params
        self.ccml_module.point_trainer(self)
        self.trainer_prepare()
        logging.debug("trainer_prepare done")

        context = contextlib.nullcontext
        for epoch in range(self.current_epoch, self.total_epoch):
            self.current_epoch = epoch
            # 分布式sampler设置
            if self.train_sampler is not None and self.val_sampler is not None:
                self.train_sampler.set_epoch(self.current_epoch)
                self.val_sampler.set_epoch(self.current_epoch)
            self.model.train()
            all_train_results = []  # 用于训练结束的回调
            avg_accumulate_loss = 0.0  # 计算一个梯度累加的平均loss
            accumlate_count = 0
            moving_total_loss = 0

            # before train_epoch
            self.exec_callbacks(
                stage="before_train_epoch",
                value={},
            )
            self.before_train_loop(value={})
            with tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc="训练中",
                disable=self.local_rank > 0,
            ) as tbar:
                self.tbar = tbar  # 用于日志模块的在tqdm中的调用
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    context = self.model.join
                else:
                    context = contextlib.nullcontext
                with context():
                    last_time = time.time()
                    for i, batch in tbar:
                        if i > self.train_data_factor * len(self.train_dataloader):
                            break  # 用于快速验证数据训练流程
                        # forward time 统计
                        pre_time = time.time()
                        _time_cost_recoder.recoder("get_batch", last_time-pre_time)
                        # 如果是分布式又采用的梯度累加
                        # https://github.com/wenet-e2e/wenet/blob/ae52150ab0f108767f33147d9fdb3aefb346ccb8/wenet/utils/executor.py
                        if (
                            self.ddp  # 分布式
                            and i % self.accumulate_grad != 0  # 非梯度累加的最后一次
                            and i != len(self.train_dataloader) - 1  # # 非一个epoch训练的最后一次
                        ):
                            context = self.model.no_sync
                        else:
                            context = contextlib.nullcontext
                        with context():
                            with torch.cuda.amp.autocast(enabled=self.use_amp):
                                # batch to device
                                batch = self.batch_to_device(batch)
                                train_out = self.train_loop(batch)
                                # detach其中的参数
                                detach_pre_time = time.time()
                                all_train_results.append(self.detach_dict(train_out))
                                loss = train_out["loss"] / self.accumulate_grad
                                avg_accumulate_loss = (
                                    avg_accumulate_loss + loss.detach().item()
                                )
                                moving_total_loss = (
                                    moving_total_loss + loss.detach().item()
                                )
                                accumlate_count += 1
                                _time_cost_recoder.recoder("forward.detach", time.time() - detach_pre_time)
                            scale_pre_time = time.time()
                            self.scalar.scale(loss).backward()
                            _time_cost_recoder.recoder("forward.backward", time.time() - scale_pre_time)
                        _time_cost_recoder.recoder("forward", time.time() - pre_time)
                        pre_time = time.time()
                        # 梯度累积
                        if (
                            i % self.accumulate_grad == self.accumulate_grad - 1
                            or i == len(self.train_dataloader) - 1
                        ):
                            self.scalar.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=20
                            )
                            self.scalar.step(self.optimizer)
                            self.scalar.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            # 学习率调度
                            if (
                                self.sche_interval == "step"
                                and self.sche_monitor is not None
                                and self.lr_scheduler is not None
                            ):
                                self.lr_scheduler.step(train_out[self.sche_monitor])
                            elif (
                                self.sche_interval == "step"
                                and self.sche_monitor is None
                                and self.lr_scheduler is not None
                            ):
                                self.lr_scheduler.step()
                            # after train_loop回调
                            self.exec_callbacks(
                                stage="after_train_loop",
                                value={
                                    "avg_accumulate_loss": avg_accumulate_loss
                                    / accumlate_count,
                                    "moving_avg_loss": moving_total_loss / (i + 1),
                                },
                            )
                            avg_accumulate_loss = 0.0
                            accumlate_count = 0
                            self.current_step += 1
                            _time_cost_recoder.recoder("loss.step", time.time() - pre_time)
                        last_time = time.time()
            # do swa
            if (
                self.use_swa
                and self.current_epoch > self.swa_config[1] * self.total_epoch
            ):
                logging.info("doing Stochastic Weight Averaging")
                self.swa_model.update_parameters(self.model)
            # after train_epoch回调
            self.exec_callbacks(
                stage="after_train_epoch",
                value={},
            )
            self.train_loop_end(all_train_results)
            if not self.current_epoch % self.eval_interval == self.eval_interval - 1:
                continue
            
            # 验证开始
            self.model.eval()
            all_val_results = []
            moving_total_loss = 0.0
            moving_avg_loss = 0.0
            with tqdm(
                enumerate(self.val_dataloader),
                total=len(self.val_dataloader),
                desc="测试中",
                disable=self.local_rank > 0,
            ) as tbar:
                self.tbar = tbar  # 用于日志模块的在tqdm中的调用

                for i, batch in tbar:
                    if i > self.train_data_factor * len(self.val_dataloader):
                        break  # 用于快速验证数据训练流程
                    with torch.no_grad():
                        batch = self.batch_to_device(batch)
                        eval_out = self.eval_loop(batch)
                        all_val_results.append(self.detach_dict(eval_out))
                    moving_total_loss += eval_out["val_loss"].detach().item()
                    moving_avg_loss = moving_total_loss / (i + 1)
                # after eval loop finished callback
                self.exec_callbacks(
                    stage="after_eval_loop",
                    value={
                        "moving_avg_loss": moving_avg_loss,
                        "all_val_results": all_val_results,
                    },
                )
            # 学习率调度
            if (
                self.sche_interval == "epoch"
                and self.sche_monitor is not None
                and self.lr_scheduler is not None
            ):
                self.lr_scheduler.step(moving_avg_loss)
                logging.debug(f"step lr by loss {moving_avg_loss}")
            elif (
                self.sche_interval == "epoch"
                and self.sche_monitor is None
                and self.lr_scheduler is not None
            ):
                self.lr_scheduler.step()
                logging.debug("step lr")
            self.eval_loop_end(all_val_results)
            # after eval finished callback
            self.exec_callbacks(
                stage="after_eval_epoch",
                value={
                    "avg_val_loss": moving_total_loss / (i + 1),
                    "all_val_results": all_val_results,
                    "epoch": self.current_epoch,
                },
            )
        # eoch loop end
        # swa bn
        if self.use_swa:
            self.model.train()
            with tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc="swa bn update...",
                disable=self.local_rank > 0,
            ) as tbar:
                self.tbar = tbar
                for i, batch in tbar:
                    if i > self.train_data_factor * len(self.train_dataloader):
                        break  # 用于快速验证数据训练流程
                    with torch.no_grad():
                        batch = self.batch_to_device(batch)
                        eval_out = self.eval_loop(batch)
                self.exec_callbacks(
                    stage="after_eval_epoch",
                    value={"swa": True},
                )

    # 测试需要什么： model, dataset
    def test(self, ccml_module, dataset: Dataset, dataloader_params: dict):
        self.test_dataset = dataset
        self.ccml_module = ccml_module
        self.dataloader_params = dataloader_params
        self.trainer_prepare()
        self.ccml_module.point_trainer(self)
        self.ccml_module.model.eval()
        all_test_results = []
        moving_total_loss = 0.0
        with tqdm(
            enumerate(self.test_dataloader),
            total=len(self.test_dataloader),
            desc="测试中",
        ) as tbar:
            self.tbar = tbar  # 用于日志模块的在tqdm中的调用

            for i, batch in tbar:
                with torch.no_grad():
                    batch = self.batch_to_device(batch)
                    eval_out = self.test_loop(batch)
                    all_test_results.append(self.detach_dict(eval_out))
                moving_avg_loss = moving_total_loss / (i + 1)
        self.ccml_module.test_loop_end(all_test_results)
        self.exec_callbacks(
            stage="test_loop_end",
            value={
                "avg_test_loss": moving_avg_loss,
                "all_test_results": all_test_results,
            },
        )

    def resume_from_checkpoint(
        self,
        checkpoint_path: str = None,
        resume_train_states: bool = True,
        gpu_id: Optional[int] = None,
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scalar: torch.cuda.amp.GradScaler = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        logger: Logger = None,
    ) -> Tuple[
        torch.nn.Module,
        int,
        torch.optim.Optimizer,
        torch.cuda.amp.GradScaler,
        torch.optim.lr_scheduler._LRScheduler,
        Logger,
    ]:
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
        device = (
            torch.device("cuda:" + str(gpu_id))
            if isinstance(gpu_id, int)
            else torch.device("cpu")
        )
        state_dicts = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dicts["model"])
        epoch = state_dicts["epoch"]
        if not resume_train_states:
            return model, 0, optimizer, scalar, lr_scheduler, logger
        optimizer.load_state_dict(state_dicts["optimizer"])
        optimizer.param_groups[0]['capturable'] = True
        if scalar is not None:
            scalar.load_state_dict(state_dicts["scalar"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
        logger.load_state_dict(state_dicts["logger"])
        return model, epoch, optimizer, scalar, lr_scheduler, logger

    def exec_callbacks(self, stage: str = None, value: Any = None):
        """根据训练阶段执行回调函数

        Args:
            stage (str, optional): 所处的阶段. 可选after_train_loop.
            value (Any, optional): [description]. Defaults to None.
        """
        for callback in self.callbacks:
            if hasattr(callback, stage):
                eval(f"callback.{stage}")(value)
            else:
                logging.warning(f"no method {stage} in callbacks")

    def detach_dict(self, data: dict):
        """对字典中的value进行detach，以便于结果统计的同时不占用显存

        Args:
            data (dict): 训练或者验证一个loop的结果字典
        """
        new_dict = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.detach().data
            else:
                new_dict[key] = value
        return new_dict

    @register_cost_statistic(need_return=True)
    def batch_to_device(self, batch: List[Any]):
        batch = list(batch)
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(self.device)
            if isinstance(batch[i], list):
                for j in range(len(batch[i])):
                    if isinstance(batch[i][j], torch.Tensor):
                        batch[i][j] = batch[i][j].to(self.device)
        return batch


if __name__ == "__main__":
    # example
    # model = CCMLModule()
    trainer = Trainer(gpu_id=0, checkpoint_path="/path/to/ckpt.pt")
    trainer.fit()
    trainer.test()
