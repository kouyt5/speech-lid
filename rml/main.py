import sys, os

sys.path.append(".")
sys.path.append("..")
import torch
import torchvision
from ccml.callbacks.ckpt_callback import CkptCallback
from ccml.callbacks.lr_callback import LrCallback
from ccml.loggers.wandb_logger import WandbLogger
from ccml.trainer import Trainer
from ccml import seed_everything
import torch.multiprocessing as mp
import logging
import argparse
from rml.rml_datasets import RML16Dataset, RML16aDatasetManager
from rml.RMLModule import RMLModule


class Config:
    rotation = 30
    dropout = 0.1
    hidden_dim = 512
    lr = 0.01
    port = 11488


def main(gpu=0, world_size=1, local_rank=0, config=None):
    logging.basicConfig(
        format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    logging.info(f"world size {world_size}")
    logging.info(f"rank {local_rank}")
    seed_everything(0)
    if config.exp_name is None:
        config.exp_name = f"lr_{config.lr}_kernel_{config.kernel}_filters_{config.base_filters}"\
            f"_block_{config.n_block}_class_{config.num_class}_{config.optim_name}"\
                f"drop_{config.dr}_rnn_{config.use_rnn}_aug_{config.aug}_mix_{config.mix_up}"\
                    f"snr_{config.use_snr_info}_w_{config.snr_loss_weight}"
        if config.test:
            config.exp_name = "test_" + config.exp_name
    if config.ckpt_path is None:
        config.ckpt_path = f"exp/{config.exp_name}"
        if config.test:
            config.ckpt_path = "ckpt"
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity="kouyt5",
        name=config.exp_name,
        wandb_id=config.wandb_id,
    )
    data_manager = RML16aDatasetManager(
        config.data_path,
        scale=[8, 1, 1],
        split_type="seen",
    )
    train_dataset = RML16Dataset(
        data_manager.get_data("train"), data_manager.get_key_mapping(), aug=config.aug
    )
    val_dataset = RML16Dataset(
        data_manager.get_data("val"), data_manager.get_key_mapping()
    )
    test_dataset = RML16Dataset(
        data_manager.get_data("test"), data_manager.get_key_mapping()
    )
    ckpt_callback = CkptCallback(
        file_name_metric=["epoch", "val_loss"],
        save_topk=2,
        ckpt_path=f"{config.ckpt_path}",
    )
    lr_callback = LrCallback()
    trainer = Trainer(
        total_epoch=config.total_epoch,
        gpu_id=gpu if torch.cuda.is_available() else None,
        callbacks=[lr_callback, ckpt_callback],
        loggers=[wandb_logger],
        log_interval=10,
        ddp=False,
        local_rank=local_rank,
        world_size=world_size,
        backend="nccl",
        # master_port=config.port,
        master_addr="192.168.1.131",
        init_method="tcp://",
        checkpoint_path=config.resume_from,
        use_amp=True,
        use_swa=config.use_swa
    )
    dataloader_params = {
        "train_batch_size": config.train_bs,
        "val_batch_size": 32,
        "pin_memory": True,
        "num_workers": 6,
        "prefetch_factor": 40,
    }
    ccml_module = RMLModule(
        base_filters=config.base_filters,
        lr=config.lr,
        kernel_size=config.kernel,
        optim_name=config.optim_name,
        n_classes=config.num_class,
        n_block=config.n_block,
        dr=config.dr,
        rnn=config.use_rnn,
        increasefilter_gap=config.increasefilter_gap,
        use_snr_info=config.use_snr_info,
        snr_loss_weight=config.snr_loss_weight,
        mix_up=config.mix_up
        
    )
    if config.test:
        trainer.test(ccml_module, test_dataset, dataloader_params)
        return
    trainer.fit(
        ccml_module, train_dataset, val_dataset, dataloader_params=dataloader_params
    )


if __name__ == "__main__":
    """
    Radio Machine Learning Dataset Generation with GNU Radio(2016)
    https://blog.csdn.net/QAQIknow/article/details/107282372

    """
    # CUDA_VISIBLE_DEVICES=1 python main.py --rank=1 --world_size=2 --proc=1
    # ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
    parse = argparse.ArgumentParser(description="mnist")
    parse.add_argument("--wandb_project", type=str, default="rml")
    parse.add_argument("--exp_name", type=str, default=None)
    parse.add_argument("--wandb_id", type=str, default=None)
    parse.add_argument("--base_filters", type=int, default=128) 
    parse.add_argument("--kernel", type=int, default=7)
    parse.add_argument("--n_block", type=int, default=5)
    parse.add_argument("--dr", type=float, default=0.5, help="dropout rate")
    parse.add_argument("--lr", type=float, default=0.1)
    parse.add_argument("--snr_loss_weight", type=float, default=0.1)
    parse.add_argument("--wc", type=float, default=1e-4)
    parse.add_argument("--num_class", type=int, default=11)
    parse.add_argument("--increasefilter_gap", type=int, default=4)
    parse.add_argument("--optim_name", type=str, default="sgd")
    parse.add_argument("--aug", action="store_true", help="数据增强")
    parse.add_argument("--use_snr_info", action="store_true", help="使用snr信息")
    parse.add_argument("--use_rnn", action="store_true")
    parse.add_argument("--mix_up", action="store_true")
    parse.add_argument("--train_bs", type=int, default=256)
    parse.add_argument("--test", action="store_true")
    parse.add_argument("--use_swa", action="store_true")
    parse.add_argument("--resume_from", type=str, default=None)
    parse.add_argument("--ckpt_path", type=str, default=None)
    parse.add_argument("--total_epoch", type=int, default=50)
    parse.add_argument(
        "--data_path",
        type=str,
        default="/home/cc/workdir/data/rml/RML2016.10a_dict.pkl",
    )
    arg = parse.parse_args()
    main(gpu=0, config=arg)
    # mp.spawn(main, nprocs=n_proc, args=(arg.world_size, arg.rank, arg))
