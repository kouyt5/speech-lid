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


from mnist.mnist_module import MnistModule

class Config:
    rotation = 30
    dropout = 0.1
    hidden_dim = 512
    lr = 0.01
    port = 11488
    
def main(gpu=0, world_size=1, local_rank=0, config=None):
    logging.basicConfig(format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",level=logging.DEBUG)
    logging.info(f"world size {world_size}")
    logging.info(f"rank {local_rank}")
    seed_everything(0)
    wandb_logger = WandbLogger(project="mnist", entity="kouyt5")
    train_transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.RandomCrop((24,24)), #.CenterCrop((24, 24))
            torchvision.transforms.RandomRotation(config.rotation),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = torchvision.datasets.MNIST("/tmp/", train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.MNIST("/tmp/", train=False, download=True, transform=test_transform)
    # ckpt_callback = CkptCallback(file_name_metric=["epoch", "val_loss"])
    lr_callback = LrCallback()
    trainer = Trainer(
        total_epoch=100,
        gpu_id=gpu if torch.cuda.is_available() else None,
        callbacks=[lr_callback],
        loggers=[wandb_logger],
        log_interval=10,
        ddp=True,
        local_rank=local_rank,
        world_size=world_size,
        backend="nccl",
        master_port=config.port,
        master_addr="192.168.1.131",
        init_method="tcp://"
    )
    dataloader_params = {
        "train_batch_size": 256,
        "val_batch_size": 32,
        "pin_memory": True,
        "num_workers": 6,
    }
    
    ccml_module = MnistModule(droprate=config.dropout, hidden_dim=config.hidden_dim, lr = config.lr)

    trainer.fit(ccml_module, train_dataset, val_dataset, dataloader_params=dataloader_params)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python main.py --rank=1 --world_size=2 --proc=1
    # ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
    parse = argparse.ArgumentParser(description="mnist")
    parse.add_argument("--rank", type=int, default=0)
    parse.add_argument("--world_size", type=int, default=1)
    parse.add_argument("--proc", type=int, default=1)
    parse.add_argument("--port", type=str, default="11401")
    arg = parse.parse_args()
    # config = wandb.config
    config = Config()
    config.port = arg.port
    n_proc = arg.proc
    mp.set_start_method("spawn")
    mp.spawn(main, nprocs=n_proc, args=(arg.world_size, arg.rank, config))
    
