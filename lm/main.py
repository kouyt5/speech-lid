import argparse
import logging
logging.basicConfig(
    format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
import sys, os

os.chdir("lm")
sys.path.append(".")
sys.path.append("..")

from lm_agent import LMModule

logging.basicConfig(
    format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
import torch
from ccml.callbacks.ckpt_callback import CkptCallback
from ccml.callbacks.lr_callback import LrCallback
from ccml.loggers.wandb_logger import WandbLogger
from ccml.trainer import Trainer
from ccml import seed_everything
from tokenizer import build_vocab, Tokenizer
from wiki_dataset import Wiki102Dataset

    

def main(config=None):
    logging.info(f"world size {config.world_size}")
    logging.info(f"rank {config.local_rank}")
    seed_everything(0)
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity="kouyt5",
        name=config.exp_name,
        wandb_id=config.wandb_id,
    )
    vocab = build_vocab(config.train_manifest, word_level=True, min_count=3)
    print(f"vocab size {len(vocab)}")
    tokenizer = Tokenizer(vocab=vocab)
    train_dataset = Wiki102Dataset(data_path=config.train_manifest,
                                   tokenizer=tokenizer, mask=True, mask_prob=config.mask_prob)
    val_dataset = Wiki102Dataset(data_path=config.val_manifest,
                                   tokenizer=tokenizer, mask=False)
    ckpt_callback = CkptCallback(
        file_name_metric=["epoch", "avg_val_loss"],
        save_topk=3,
        ckpt_path=config.ckpt_path,
    )
    lr_callback = LrCallback()
    trainer = Trainer(
        total_epoch=config.total_epoch,
        gpu_id=config.gpu if torch.cuda.is_available() else None,
        callbacks=[lr_callback, ckpt_callback],
        loggers=[wandb_logger],
        log_interval=1,
        ddp=False,
        local_rank=config.local_rank,
        world_size=config.world_size,
        backend="nccl",
        use_amp=True,
        master_port=config.port,
        master_addr="192.168.1.131",
        init_method="tcp://",
        train_data_factor=1.,
        checkpoint_path=config.resume_from,
        eval_interval=config.eval_interval
    )
    dataloader_params = {
        "train_batch_size": config.train_bs,
        "val_batch_size": config.val_bs,
        "pin_memory": True,
        "num_workers": 6,
        "prefetch_factor": 40
    }
    ccml_module = LMModule(lr=config.lr, vocab=vocab, embedding_dim=config.embedding_dim,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           lstm_dropout=config.lstm_dropout,
                           bidirectional=False)
    if config.stage == "test":
        trainer.test(ccml_module, val_dataset, dataloader_params=dataloader_params)
    else:
        trainer.fit(
            ccml_module, train_dataset, val_dataset, dataloader_params=dataloader_params
        )


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="lstm lm")

    parse.add_argument(
        "--train_manifest",
        type=str,
        default="/home/cc/workdir/code/lm/data/wikitext-103/wiki.train.tokens",
    )
    parse.add_argument(
        "--val_manifest",
        type=str,
        default="/home/cc/workdir/code/lm/data/wikitext-103/wiki.valid.tokens",
    )
    parse.add_argument(
        "--test_manifest",
        type=str,
        default="/home/cc/workdir/code/lm/data/wikitext-2/wiki.test.tokens",
    )
    parse.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="ckpt file path to resume experiment, if None do not train from scratch",
    )

    parse.add_argument("--embedding_dim", type=int, default=128)
    parse.add_argument("--hidden_size", type=int, default=512)
    parse.add_argument("--num_layers", type=int, default=1)
    parse.add_argument("--lstm_dropout", type=float, default=0.)
    parse.add_argument("--mask_prob", type=float, default=0.05)
    
    parse.add_argument("--total_epoch", type=int, default=40)
    parse.add_argument("--train_bs", type=int, default=8)
    parse.add_argument("--val_bs", type=int, default=8)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parse.add_argument("--gpu", type=int, default=0)
    parse.add_argument("--ckpt_path", type=str, default="exp/tmp")
    parse.add_argument("--eval_interval", type=int, default=1)
    
    parse.add_argument("--wandb_id", type=str, default=None)
    parse.add_argument("--exp_name", type=str, default="base")
    parse.add_argument("--stage", type=str, default="train")
    parse.add_argument("--wandb_project", type=str, default="lm")
    # ddp
    parse.add_argument("--world_size", type=int, default=1)
    parse.add_argument("--local_rank", type=int, default=0)
    parse.add_argument("--proc", type=int, default=1)
    parse.add_argument("--port", type=str, default="11401")
    arg = parse.parse_args()
    main(config=arg)
