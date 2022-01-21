import argparse
import logging

logging.basicConfig(
    format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
import sys, os

os.chdir("wav2vec-exp")
sys.path.append(".")
sys.path.append("..")
import torch
from ccml.callbacks.ckpt_callback import CkptCallback
from ccml.callbacks.lr_callback import LrCallback
from ccml.loggers.wandb_logger import WandbLogger
from ccml.trainer import Trainer
from ccml import seed_everything

from wav2vec_module import Wav2vecModule
from raw_audio_dataset import RawAudioDatasets
from tokenizer import CTCTokenizer


class Config:
    dropout = 0.1
    pt_path = "/home/cc/workdir/code/wav2vec-exp/ckpts/wav2vec_small.pt"  # "/home/cc/workdir/code/wav2vec-exp/ckpts/xlsr_53_56k.pt"
    train_manifest = (
        "/home/cc/workdir/code/wav2vec-exp/datasets/libri/train-clean-100_5k.json"
    )
    val_manifest = "/home/cc/workdir/code/wav2vec-exp/datasets/libri/dev-clean.json"
    test_manifest = "/home/cc/workdir/code/wav2vec-exp/datasets/libri/dev-clean.json"
    vocab = "/home/cc/workdir/code/wav2vec-exp/datasets/libri/vocab.txt"
    lr = 1e-4
    port = 11488
    checkpoint_path = None


def main(config=None):
    logging.info(f"world size {config.world_size}")
    logging.info(f"rank {config.local_rank}")
    seed_everything(0)
    wandb_logger = WandbLogger(
        project="asr_wav2vec",
        entity="kouyt5",
        name=config.exp_name,
        wandb_id=config.wandb_id,
    )

    tokenizer = CTCTokenizer(vocab=config.vocab)
    train_dataset = RawAudioDatasets(
        manifest_path=config.train_manifest, tokenizer=tokenizer, max_duration=config.max_duration
    )
    val_dataset = RawAudioDatasets(
        manifest_path=config.val_manifest, tokenizer=tokenizer, max_duration=37
    )
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
        train_data_factor=1.0,
        checkpoint_path=config.resume_from,
        use_swa=config.use_swa,
        swa_config=(0.1, 0.6)
    )
    dataloader_params = {
        "train_batch_size": config.train_bs,
        "val_batch_size": config.val_bs,
        "pin_memory": True,
        "num_workers": 6,
    }
    ccml_module = Wav2vecModule(
        pt_path=config.pt_path,
        feature_selection=config.feature,
        dropout=config.dropout,
        vocab_size=len(tokenizer.export_vocab()) + 1,
        lr=config.lr,
        tokenizer=tokenizer,
        metric_interval=20,
        reset_param=config.reset_param,
        freeze_decoder_epoch=config.freeze_decoder_epoch,
        freeze_encoder_epoch=config.total_epoch
        if config.freeze_encoder_epoch is None
        else config.freeze_encoder_epoch,
        linear_dim=config.linear_dim
    )
    if config.stage == "test":
        trainer.test(ccml_module, val_dataset, dataloader_params=dataloader_params)
    else:
        trainer.fit(
            ccml_module, train_dataset, val_dataset, dataloader_params=dataloader_params
        )


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python main.py --rank=1 --world_size=2 --proc=1
    # ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
    parse = argparse.ArgumentParser(description="wav2vec finetune to ASR")
    parse.add_argument(
        "--pt_path",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/ckpts/wav2vec_small.pt",
        help="path to the pretrain wav2vec model",
    )
    parse.add_argument(
        "--train_manifest",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/datasets/libri/train-clean-100_3k.json",
    )
    parse.add_argument(
        "--val_manifest",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/datasets/libri/dev-clean.json",
    )
    parse.add_argument(
        "--test_manifest",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/datasets/libri/dev-clean.json",
    )
    parse.add_argument(
        "--vocab",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/datasets/libri/vocab.txt",
    )
    parse.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="ckpt file path to resume experiment, if None do not train from scratch",
    )
    parse.add_argument(
        "--reset_param", action="store_true", help="reset model paramters to train"
    )
    parse.add_argument(
        "--feature",
        type=str,
        default="last_hidden_state",
        choices=["last_hidden_state", "hidden_states"],
        help="select feature including hidden_states and last_hidden_state",
    )
    parse.add_argument("--max_duration", type=float, default=16.7)
    parse.add_argument("--linear_dim", type=int, default=768, help="768 for base, 1024 for large or xlsr")
    parse.add_argument("--total_epoch", type=int, default=40)
    parse.add_argument("--train_bs", type=int, default=4)
    parse.add_argument("--val_bs", type=int, default=4)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parse.add_argument("--dropout", type=float, default=0.0)
    parse.add_argument("--gpu", type=int, default=0)
    parse.add_argument("--freeze_decoder_epoch", type=int, default=10)
    parse.add_argument("--freeze_encoder_epoch", type=int, default=None)
    parse.add_argument("--use_swa", action="store_true")

    parse.add_argument("--ckpt_path", type=str, default="exp/tmp")
    parse.add_argument("--wandb_id", type=str, default=None)
    parse.add_argument("--exp_name", type=str, default="base")
    parse.add_argument("--stage", type=str, default="train")
    # ddp
    parse.add_argument("--world_size", type=int, default=1)
    parse.add_argument("--local_rank", type=int, default=0)
    parse.add_argument("--proc", type=int, default=1)
    parse.add_argument("--port", type=str, default="11401")
    arg = parse.parse_args()
    main(config=arg)
