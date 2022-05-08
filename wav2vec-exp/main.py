import argparse
import logging

import wandb

logging.basicConfig(
    format="[%(asctime)s] - [%(filename)s :%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
import sys, os

# os.environ['PYTHONWARNINGS'] ='ignore:semaphore_tracker:UserWarning'

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
    # build exp name
    if config.exp_name is None:
        config.exp_name = f"lr{config.lr}_dr{config.dropout}"\
            f"_de_freeze{config.freeze_decoder_epoch}"\
                f"_en_freeze_{config.freeze_encoder_epoch}_bs{config.train_bs}"\
                    f"_lstm_{config.num_layer}_glu_{config.use_glu}"
    if config.ckpt_path is None:
        config.ckpt_path = f'exp/{config.exp_name}'
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity="kouyt5",
        name=config.exp_name,
        wandb_id=config.wandb_id,
    )

    tokenizer = CTCTokenizer(vocab=config.vocab)
    train_dataset = RawAudioDatasets(
        manifest_path=config.train_manifest,
        tokenizer=tokenizer,
        max_duration=config.max_duration,
        upper=config.use_lm,  # 如果使用语言模型，字母转化为大些
    )
    val_dataset = RawAudioDatasets(
        manifest_path=config.val_manifest,
        tokenizer=tokenizer,
        max_duration=37,
        upper=config.use_lm,
    )
    ckpt_callback = CkptCallback(
        file_name_metric=["epoch", "avg_val_loss"],
        save_topk=3,
        ckpt_path=config.ckpt_path,
    )
    lr_callback = LrCallback()
    if not torch.cuda.is_available():
        config.gpu = -1
    trainer = Trainer(
        total_epoch=config.total_epoch,
        gpu_id=config.gpu if config.gpu > -1 else None,
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
        swa_config=(0.1, 0.6),
        eval_interval=config.eval_interval,
    )
    dataloader_params = {
        "train_batch_size": config.train_bs,
        "val_batch_size": config.val_bs,
        "pin_memory": True,
        "num_workers": 6,
        "prefetch_factor": 40,
    }
    lm_model = None
    if config.use_lm:
        from lm_decoder import BeamSearchDecoderWithLM

        lm_model = BeamSearchDecoderWithLM(
            tokenizer.export_vocab(),
            beam_width=100,
            alpha=1,
            beta=0.5,
            lm_path=config.lm_path,
            num_cpus=12,
            cutoff_prob=1,
            cutoff_top_n=40,
        )
    ccml_module = Wav2vecModule(
        pt_path=config.pt_path,
        feature_selection=config.feature,
        dropout=config.dropout,
        vocab_size=len(tokenizer.export_vocab()) + 1,
        lr=config.lr,
        tokenizer=tokenizer,
        metric_interval=50,
        reset_param=config.reset_param,
        freeze_decoder_epoch=config.freeze_decoder_epoch,
        freeze_encoder_epoch=config.total_epoch
        if config.freeze_encoder_epoch is None
        else config.freeze_encoder_epoch,
        linear_dim=config.linear_dim,
        use_cer=config.use_cer,
        optimizer_name=config.optimizer_name,
        feature_mask=config.no_mask,
        lm_model=lm_model,
        num_layers=config.num_layer,
        glu=config.use_glu
    )
    if config.stage == "test":
        trainer.test(ccml_module, val_dataset, dataloader_params=dataloader_params)
    else:
        trainer.fit(
            ccml_module, train_dataset, val_dataset, dataloader_params=dataloader_params
        )
    # close process pool
    tokenizer.pool.close()


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
        "--lm_path",
        type=str,
        default="/home/cc/workdir/code/wav2vec-exp/ckpts/4-gram.arpa",
        help="language model path",
    )
    parse.add_argument(
        "--train_manifest",
        type=str,
        default="/data/chenc/libri-light2/librispeech_finetuning-processed/all.json",
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
    parse.add_argument("--use_cer", action="store_true")
    parse.add_argument("--use_lm", action="store_true")
    parse.add_argument("--max_duration", type=float, default=16.7)
    parse.add_argument(
        "--linear_dim",
        type=int,
        default=768,
        help="768 for base, 1024 for large or xlsr",
    )
    parse.add_argument("--num_layer", type=int, default=1)
    parse.add_argument("--use_glu", action="store_true")
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
    parse.add_argument("--ckpt_path", type=str, default=None)
    parse.add_argument("--optimizer_name", type=str, default="adam")
    parse.add_argument(
        "--no_mask",
        action="store_false",
        help="if apply a mask to feature extract, default True",
    )
    parse.add_argument("--eval_interval", type=int, default=1)

    parse.add_argument("--wandb_id", type=str, default=None)
    parse.add_argument("--exp_name", type=str, default=None)
    parse.add_argument("--stage", type=str, default="train")
    parse.add_argument("--wandb_project", type=str, default="asr_wav2vec")
    # ddp
    parse.add_argument("--world_size", type=int, default=1)
    parse.add_argument("--local_rank", type=int, default=0)
    parse.add_argument("--proc", type=int, default=1)
    parse.add_argument("--port", type=str, default="11401")
    arg = parse.parse_args()
    main(config=arg)

# test for wenet:
# python wav2vec-exp/main.py --stage test --use_cer
# --linear_dim 1024 --pt_path /data/chenc/aishell/ssl/model/xlsr_wenet/wenetspeech_wav2vec2_model/checkpoint_best.pt
# --resume_from exp/lr2e5_dr3_wd1e6_bs4_mask_ct2_25_de0_wenet_pre_lstm/last.pt --val_bs 1
# --val_manifest /data/chenc/asr/wenet/test_meeting.json
# --vocab /data/chenc/asr/wenet/vocab.txt
