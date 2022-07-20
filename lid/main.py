from collections import defaultdict
import logging
import sys, os

sys.path.append(os.path.join(".."))

from ccml.callbacks.ckpt_callback import CkptCallback
from ccml.callbacks.lr_callback import LrCallback
from ccml import seed_everything
from ccml.trainer import Trainer
from lid.LidModule_ASR import LidModule
from lid.LidModule_ASR_Supervised import LidSuperviseModule
from lid.raw_datasets import MergedDataset, MutiBatchSampler
from lid.tokenizer import CTCTokenizer
import hydra
from omegaconf import DictConfig
from ccml.loggers.wandb_logger import WandbLogger

# 为什么使用hydra？
# hydra提供简易的配置文件读取接口、实验日志按照时间归档管理，并且对关键代码侵入性
# 较小，因此在这里使用，虽然增加了代码的复杂性，但是比起自己实现一套配置文件读取以及
# python的args parser，代码更加结构化
@hydra.main(config_path="conf", config_name="lid_base")
def main(cfg: DictConfig) -> None:
    """
    代码启动主函数
    """
    train_conf = cfg["trainer"]
    # 模型参数
    model_conf = cfg["model"]
    module_conf = cfg["module"]
    # 数据加载参数
    data_conf = cfg["data"]
    wandb_conf = cfg["logger"]["wandb"]
    supervised = cfg["supervised"]

    seed_everything(0)  # 随机数种子

    ckpt_callback = CkptCallback(  # checkpint 保存回调
        file_name_metric=["epoch", "val_loss"],
        save_topk=2,
    )
    lr_callback = LrCallback()  # 学习率记录回调
    # data

    train_dataset_list = []
    val_dataset_list = []
    test_dataset_list = []
    lang2tokenizer_dict = defaultdict(dict)
    lang2index_dict = defaultdict(int)
    lang2vocablen_dict = defaultdict(int)
    for item in data_conf["langs"]:
        lang2tokenizer_dict[item["lang"]] = CTCTokenizer(item["vocab"])
        train_dataset_list.append(item["train_manifest"])
        val_dataset_list.append(item["val_manifest"])
        test_dataset_list.append(item["test_manifest"])
        lang2index_dict[item["lang"]] = item["id"]
        lang2vocablen_dict[item["lang"]] = len(lang2tokenizer_dict[item["lang"]].export_vocab())

    model_agent = None
    if supervised:
        logging.info("使用监督损失模型")
        model_agent = LidSuperviseModule(**module_conf, **model_conf,
                                lang2vocab=lang2vocablen_dict,
                                lang2index_dict=lang2index_dict,
                                tokenizer_dict=lang2tokenizer_dict)
    else:
        logging.info("使用无监督预训练模型")
        model_agent = LidModule(**module_conf, **model_conf,
                                lang2vocab=lang2vocablen_dict,
                                lang2index_dict=lang2index_dict,
                                tokenizer_dict=lang2tokenizer_dict)  # 模型训练封装
    
    train_dataset = MergedDataset(
        train=True,
        manifest_files=train_dataset_list,
        lang2index_dict=lang2index_dict,
        lang2tokenizer=lang2tokenizer_dict,
        max_duration=data_conf["max_duration"],
        source=data_conf["source"]
    )
    val_dataset = MergedDataset(
        train=False,
        manifest_files=val_dataset_list,
        lang2index_dict=lang2index_dict,
        lang2tokenizer=lang2tokenizer_dict,
        max_duration=16.7,
        source=data_conf["source"]
    )
    test_dataset = MergedDataset(
        train=False,
        manifest_files=test_dataset_list,
        lang2index_dict=lang2index_dict,
        lang2tokenizer=lang2tokenizer_dict,
        max_duration=16.7,
        source=data_conf["source"]
    )

    train_batch_sample = MutiBatchSampler(
        train_dataset.samplers,
        batch_size=data_conf["sampler_common"]["train_batch_size"],
        drop_last=True,
    )
    val_batch_sample = MutiBatchSampler(
        val_dataset.samplers,
        batch_size=data_conf["sampler_common"]["val_batch_size"],
        drop_last=False,
    )
    test_batch_sample = MutiBatchSampler(
        test_dataset.samplers,
        batch_size=data_conf["sampler_common"]["test_batch_size"],
        drop_last=False,
    )
    dataloader_params = dict(data_conf["dataloader_params"])
    dataloader_params["train_batch_sampler"] = train_batch_sample
    dataloader_params["val_batch_sampler"] = val_batch_sample
    dataloader_params["test_batch_sampler"] = test_batch_sample
    wandb_logger = WandbLogger(**wandb_conf)  # 在线日志记录回调
    trainer = Trainer(
        callbacks=[ckpt_callback, lr_callback], loggers=[wandb_logger], **train_conf
    )  # 训练器
    if cfg["stage"] == "train":
        trainer.fit(
            model_agent,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            dataloader_params=dataloader_params,
        )  # 开始训练
    else:
        trainer.test(
            ccml_module=model_agent,
            dataloader_params=dataloader_params,
            dataset=test_dataset,
        )  # 测试


if __name__ == "__main__":
    main()
