from collections import defaultdict
import logging
import sys, os

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("."))

from spec_pred.agent_module import SpecModule
from spec_pred.spec_dataset import DataSource, SpecDataset
from ccml.callbacks.ckpt_callback import CkptCallback
from ccml.callbacks.lr_callback import LrCallback
from ccml.callbacks.profile_callback import ProfileCallback
from ccml import seed_everything
from ccml.trainer import Trainer
import hydra
from omegaconf import DictConfig
from ccml.loggers.comet_logger import CometLogger


@hydra.main(config_path="conf", config_name="config_cnnlstm")
def main(cfg: DictConfig) -> None:
    logging.info("python " + " ".join(sys.argv))
    """
    代码启动主函数
    """
    train_conf = cfg["trainer"]
    # 模型参数
    model_conf = cfg["model"]
    logging.info("模型参数: " + str(model_conf))
    module_conf = cfg["module"]
    logging.info(f"模块参数: {module_conf}")
    # 数据加载参数
    data_conf = cfg["data"]
    logging.info(f"data模块参数: {data_conf}")
    comet_conf = cfg["logger"]["comet"]

    seed_everything(0)  # 随机数种子

    # profile callback
    profile_callback = ProfileCallback()
    ckpt_callback = CkptCallback(  # checkpint 保存回调
        file_name_metric=["epoch", "val_loss"],
        save_topk=2,
    )
    lr_callback = LrCallback()  # 学习率记录回调
    model_agent = None

    model_agent = SpecModule(**module_conf, **model_conf)  # 模型训练封装
    datasource = DataSource(
        data_path=data_conf["data_path"],
        split=data_conf["split"],  # [0.8, 0.1, 0.1]
        spec_range=data_conf["spec_range"],  # (71000,76500)
    )
    train_dataset = SpecDataset(
        train_type="train",
        aug=True,
        datasource=datasource,
        win_len=data_conf["win_len"],
        aug_factor=data_conf["aug"]["factor"],
        exchange=data_conf["aug"]["exchange"],
        mask=data_conf["aug"]["mask"],
    )
    val_dataset = SpecDataset(
        train_type="dev",
        aug=False,
        datasource=datasource,
        win_len=data_conf["win_len"],
    )
    test_dataset = SpecDataset(
        train_type="test",
        aug=False,
        datasource=datasource,
        win_len=data_conf["win_len"],
    )
    dataloader_params = dict(data_conf["dataloader_params"])
    comet_logger = CometLogger(**comet_conf)
    trainer = Trainer(
        callbacks=[ckpt_callback, lr_callback],  # callbacks=[ckpt_callback, lr_callback],
        loggers=[comet_logger],
        **train_conf,
    )  # 训练器
    trainer.fit(
        model_agent,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataloader_params=dataloader_params,
    )  # 开始训练
    
    trainer.test(model_agent, test_dataset, dataloader_params)


if __name__ == "__main__":
    main()
