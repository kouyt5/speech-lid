import logging
from typing import Any, Dict, List, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from ccml.ccml_module import CCMLModule
from ccml.optim.novograd import Novograd
from ccml.optim.tri_state import TriStageLRSchedule
from ccml.utils.profile import register_cost_statistic
from lid.PretrainLidModel import PretrainLidModel, LidModel
from ccml.utils.profile import _time_cost_recoder
from lid.audio_processor import wav2mel


class LidModuleCross(CCMLModule):
    """基于交叉熵的语种识别"""

    def __init__(
        self,
        optimizer_name: str = "adam",
        optimizer_param: Dict = None,
        scheduler: str = "reduce",
        scheduler_param: Dict = None,
        interval: int = 10,
        freeze_tranformer_epoch: int = 1,
        freeze_encoder_epoch: int = 100,
        froze_wav2vec_model_epoch: int = 100,
        pt_path: str = None,
        dropout: float = 0.0,
        linear_dim: int = 768,
        mask: bool = True,
        num_layers: int = 1,
        last_model_name: str = "xvector",
        lang2index_dict: Dict = None,
        use_pre_train: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
        sr: int = 22050,
        supervised: bool = False,
        pre_train_name:str = "wav2vec",
        *args,
        **kwargs,
    ):
        super().__init__(
            pt_path=pt_path,
            linear_dim=linear_dim,
            mask=mask,
            num_layers=num_layers,
            lang2index_dict=lang2index_dict,
            mask_channel_prob=mask_channel_prob,
            mask_prob=mask_prob,
            last_model_name=last_model_name,
            supervised=supervised,
            pre_train_name=pre_train_name
        )
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param
        self.scheduler = scheduler
        self.scheduler_param = scheduler_param
        self.lang2index_dict = lang2index_dict
        self.interval = interval
        self.freeze_tranformer_epoch = freeze_tranformer_epoch
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.froze_wav2vec_model_epoch = froze_wav2vec_model_epoch
        self.sr = sr
        self.supervised = supervised
        self.index2lang_dict = {}
        for key in self.lang2index_dict.keys():
            self.index2lang_dict[self.lang2index_dict[key]] = key

        logging.info(f"采样率: {sr}")
        logging.info(f"mask channel prob{mask_channel_prob}")
        if supervised:
            self.model = LidModel(
                linear_dim=linear_dim,
                num_class=len(self.index2lang_dict.keys()),
                last_model_name=last_model_name,
            )
        else:
            self.model = PretrainLidModel(
                pt_path=pt_path,
                dropout=dropout,
                linear_dim=linear_dim,
                num_class=len(self.index2lang_dict.keys()),
                use_pre_train=use_pre_train,
                mask=mask,
                mask_channel_prob=mask_channel_prob,
                mask_prob=mask_prob,
                last_model_name=last_model_name,
                pre_train_name=pre_train_name,
            )
        self.count = 1
        self.avg_loss = 0
        self.avg_acc = 0
        self.predict_texts = None
        self.countdown_20 = 0

    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_param)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), **self.optimizer_param
            )
        elif self.optimizer_name == "novograd":
            optimizer = Novograd(self.model.parameters(), **self.optimizer_param)
        else:
            logging.warn(
                f"optimizer name {self.optimizer_name} is not exist, choice SGD optimizer"
            )
            optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_param)

        if self.scheduler == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.scheduler_param
            )
            return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}
        elif self.scheduler == "tristage":
            scheduler = TriStageLRSchedule(
                optimizer=optimizer,
                phase_ratio=[0.1, 0.4, 0.5],
                init_lr_scale=0.05,
                final_lr_scale=0.2,
                max_update=self.trainer.total_steps,
                lr=self.optimizer_param["lr"],
            )
        # return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}
        return optimizer, scheduler, {"monitor": None, "interval": "step"}

    @register_cost_statistic(need_return=True)
    def common_loop(self, batch, train_stat: bool = True) -> Dict:
        pre_time = time.time()
        wavs = batch[0]
        texts = batch[1]
        langs = batch[5]
        out = self.model(wavs, self.sr)

        loss = F.cross_entropy(out, target=langs)
        acc = torchmetrics.functional.accuracy(
            F.softmax(out, dim=-1).cpu(),
            langs.cpu(),
            task="multiclass",
            num_classes=len(self.lang2index_dict.keys()),
        )
        pred_label = torch.argmax(out, dim=-1).cpu().tolist()
        true_label = langs.int()
        _time_cost_recoder.recoder("common_loop.only_loss", time.time() - pre_time)

        return {"loss": loss, "acc": acc, "pred": pred_label, "true": true_label}

    def infer(self, x: torch.Tensor, sample_rate: int = 22050):
        device = x.device
        if self.supervised:
            x = wav2mel(
                x.cpu(),
                use_kaildi=False,
                win_length=0.025,
                hop_length=0.01,
                n_mels=80,
                n_fft=512,
                pad=0,
                sr=16000,
            ).transpose(1, 2)
        out = self.model(x.to(device), sample_rate)
        pred_label = torch.argmax(out, dim=-1).cpu().tolist()
        pre_scores = F.softmax(out, dim=-1)[0].cpu().tolist()
        pre_lang = self.index2lang_dict[pred_label[0]]
        return pre_lang, pre_scores, pred_label[0]

    def getIndexByLangName(self, name: str):
        """根据语种名称获取其index"""

        if name in self.lang2index_dict.keys():
            return self.lang2index_dict[name]
        else:
            logging.error(f"{name} is not in the lang key!")
            return 0

    @register_cost_statistic(need_return=True)
    def train_loop(self, batch):
        # out.keys(): "loss" "wer" "lang" "predict_texts" "label_texts" lid_acc
        # torch.cuda.empty_cache()
        out = self.common_loop(batch)
        if self.trainer.current_step % self.interval == self.interval - 1:
            pass

        # https://zhuanlan.zhihu.com/p/151786842
        if not torch.isnan(out["loss"]).item():
            self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
            self.avg_acc = (0.98 * self.avg_acc) + 0.02 * out["acc"]
            self.count += 1
            self.trainer.logger.log(
                data={
                    "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                    "tr_acc": self.avg_acc / (1 - np.power(0.98, self.count)),
                },
                progress=True,
                stage="train",
            )
        return {
            "loss": out["loss"],
            "acc": out["acc"],
        }

    def before_train_loop(self, value):
        self.count = 1
        self.avg_loss = 0
        self.avg_acc = 0
        epoch = self.trainer.current_epoch
        if epoch <= self.freeze_encoder_epoch:
            self.model.freeze_feature_extractor()
            logging.info("freeze encoder")
        else:
            self.model.unfreeze_feature_extractor()
            logging.info("unfreeze encoder")
        if epoch <= self.freeze_tranformer_epoch:
            self.model.freeze_tranformer_encoder()
            logging.info("freeze tranformer")
        else:
            self.model.unfreeze_tranformer_encoder()
            logging.info("unfreeze tranformer")

    def train_loop_end(self, outputs: List[Any] = None):
        self.count = 1
        self.avg_loss = 0
        self.avg_acc = 0
        # torch.cuda.empty_cache()
        total_train_loss = 0.0
        total_train_acc = 0.0
        for item in outputs:
            total_train_loss += item["loss"]
            total_train_acc += item["acc"]
        logging.info(f"tr_avg_loss {total_train_loss / len(outputs)}")
        logging.info(f"tr_avg_acc {total_train_acc / len(outputs)}")
        self.trainer.logger.log(
            data={
                "tr_avg_loss": total_train_loss / len(outputs),
                "tr_avg_acc": total_train_acc / len(outputs),
            },
            progress=False,
            stage="val",
            commit=False,
            only_tbar=False,
        )

    def val_loop(self, batch):
        # out.keys(): "wer" "lang" "predict_texts" "label_texts"
        # torch.cuda.empty_cache()
        raw_wavs = batch[0][0].clone().unsqueeze(0)
        out = self.common_loop(batch, False)
        if self.count % self.interval == self.interval - 1:
            logging.info("acc: " + str(out["acc"]))
        if not torch.isnan(out["loss"]).item():
            self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
            self.avg_acc = (0.98 * self.avg_acc) + 0.02 * out["acc"]
            self.count += 1
            self.trainer.logger.log(
                data={
                    "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                    "val_acc": self.avg_acc / (1 - np.power(0.98, self.count)),
                },
                progress=True,
                only_tbar=True,
                stage="val",
            )
        # outacc = self.infer(raw_wavs, 16000)
        # index = torch.argmax(outacc[1], dim=-1)
        # index = index[0].item()
        return {
            "val_loss": out["loss"],
            "val_acc": out["acc"],
            "pred": out["pred"],
            "true": out["true"],
        }

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        total_pred_label = []
        total_true_label = []
        for item in outputs:
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]
            total_pred_label.extend(item["pred"])
            total_true_label.extend(item["true"])

        total_acc = torchmetrics.functional.accuracy(
            torch.LongTensor(total_true_label),
            torch.LongTensor(total_pred_label),
            task="multiclass",
            num_classes=len(self.lang2index_dict.keys()),
        )
        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
                "val_acc": total_acc,
                "epoch": self.trainer.current_epoch,
            },
            progress=True,
            stage="val",
            commit=False,
            only_tbar=False,
        )
        logging.info(
            f"val_acc={total_acc}, val_avg_loss={total_val_loss / len(outputs)}"
        )
        self.trainer.logger.remove_key(["loss"])
