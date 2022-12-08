import logging
from typing import Any, Dict, List, Tuple
import time

import numpy as np
import torch
from ccml.ccml_module import CCMLModule
from ccml.optim.novograd import Novograd
from ccml.optim.tri_state import TriStageLRSchedule
from ccml.utils.profile import register_cost_statistic
from lid.Wav2vecMutiLangModel import Wav2vecMutiLangModel
from lid.WavLMMutiLangModel import WavLMMutiLangModel
from ccml.utils.profile import _time_cost_recoder


class LidModule(CCMLModule):
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
        feature_selection: str = "hidden_states",
        dropout: float = 0.0,
        linear_dim: int = 768,
        mask: bool = True,
        num_layers: int = 1,
        hidden_dim: int = 128,
        lang2vocab: Dict = None,  # {"cn": 4442}) -> None
        lang2index_dict: Dict = None,
        tokenizer_dict: Dict = None,
        use_wav2vec: bool = False,
        conformer_linear: bool = False,
        double_swish: bool = False,
        use_pre_train: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
        sr: int = 22050,
        conformer_pure: bool = False,  # 兼容conformer监督模型
        extrme_mode: bool = False,
        keep_train_lang: str = None,
        use_mask: bool = False,
        dim_head: int = 32,  # Conformer
        num_head: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__(
            pt_path=pt_path,
            feature_selection=feature_selection,
            linear_dim=linear_dim,
            mask=mask,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            lang2vocab=lang2vocab,
            lang2index_dict=lang2index_dict,
            tokenizer_dict=tokenizer_dict,
            use_wav2vec=use_wav2vec,
            conformer_linear=conformer_linear,
            double_swish=double_swish,
            mask_channel_prob=mask_channel_prob,
            mask_prob=mask_prob,
            keep_train_lang=keep_train_lang,
            use_mask=use_mask,
            dim_head=dim_head,
            num_head=num_head,
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
        self.index2lang_dict = {}
        for key in self.lang2index_dict.keys():
            self.index2lang_dict[self.lang2index_dict[key]] = key
        self.tokenizer_dict = tokenizer_dict
        self.extrme_mode = extrme_mode
        self.keep_train_lang = keep_train_lang

        logging.info(f"采样率: {sr}")
        logging.info(
            f"使用double swish: {double_swish}, mask channel prob{mask_channel_prob}"
        )
        if use_wav2vec:
            self.model = Wav2vecMutiLangModel(
                pt_path=pt_path,
                feature_selection=feature_selection,
                dropout=dropout,
                linear_dim=linear_dim,
                mask=mask,
                num_layers=num_layers,
                lang2vocab=lang2vocab,  # {"cn": 4442}) -> None
                lang2index=lang2index_dict,
                hidden_dim=hidden_dim,
                conformer_linear=conformer_linear,
                use_mask=use_mask,
                dim_head=dim_head,
                num_head=num_head,
            )
        else:
            self.model = WavLMMutiLangModel(
                pt_path=pt_path,
                feature_selection=feature_selection,
                dropout=dropout,
                linear_dim=linear_dim,
                mask=mask,
                num_layers=num_layers,
                lang2vocab=lang2vocab,  # {"cn": 4442}) -> None
                lang2index=lang2index_dict,
                hidden_dim=hidden_dim,
                conformer_linear=conformer_linear,
                double_swish=double_swish,
                use_pre_train=use_pre_train,
                mask_channel_prob=mask_channel_prob,
                mask_prob=mask_prob,
                conformer_pure=conformer_pure,
                use_mask=use_mask,
                dim_head=dim_head,
                num_head=num_head,
            )
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0
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
        wav_percents = batch[2]
        text_percents = batch[3]
        langs = batch[5]
        lang = self.index2lang_dict[langs[0].item()]
        out, (lid_asr, lid_linear) = self.model(wavs, self.sr, lang)
        out = out[lang]
        loss = self.model.model.loss_fns[lang](
            torch.log_softmax(out, dim=-1).transpose(1, 0),
            texts,
            (out.shape[1] * wav_percents).long().cpu(),
            (texts.shape[-1] * text_percents).long().cpu(),
        )
        loss = torch.mean(loss)
        _time_cost_recoder.recoder("common_loop.only_loss", time.time() - pre_time)

        # 训练 or 测试
        if (
            self.countdown_20 == 0
            or (self.predict_texts is None or not self.extrme_mode)
            or not train_stat
        ):
            self.countdown_20 = 20
            self.predict_texts = self.tokenizer_dict[lang].ctc_decode(
                torch.argmax(out, dim=-1),
                predictions_len=(torch.argmax(out, dim=-1).shape[1] * batch[2]).long(),
            )
            self.label_texts = self.tokenizer_dict[lang].decoder(
                texts, target_lengths=(texts.shape[1] * text_percents).long()
            )
            self.wer = self.model.model.wer_fn(
                self.predict_texts, self.label_texts
            ).item()

        self.countdown_20 -= 1
        return {
            "loss": loss,
            "wer": self.wer,
            "lang": lang,
            "predict_texts": self.predict_texts,
            "label_texts": self.label_texts,
        }

    def lid_infer(self, lid_out: torch.Tensor) -> List:
        """根据lid输出获取其语种信息

        Args:
            lid_out (torch.Tensor): B*C

        Returns:
            List: [{"cn": 0.3,}]
        """
        lid = []
        # (B), (B)
        max_value, argmax = torch.max(torch.softmax(lid_out, dim=-1), dim=-1)
        argmax_list = argmax.cpu().numpy().tolist()
        for i in len(argmax_list):
            tmp = {}
            tmp[self.index2lang_dict[argmax_list[i]]] = max_value[i].item()
            lid.append(tmp)
        return lid

    def infer(self, x: torch.Tensor, sample_rate: int = 22050, language: str = None):
        out, (lid_asr, lid_linear) = self.model(
            [x[0, :]], sample_rate, language
        )  # {"cn": (1, T, V)}, (1 * C)
        predict_texts = {}
        for lang in out.keys():
            predict_texts[lang] = self.tokenizer_dict[lang].ctc_decode(
                torch.argmax(out[lang], dim=-1),
            )
        return predict_texts, lid_asr, out

    @register_cost_statistic(need_return=True)
    def train_loop(self, batch):
        # out.keys(): "loss" "wer" "lang" "predict_texts" "label_texts" lid_acc
        # torch.cuda.empty_cache()
        out = self.common_loop(batch)
        if self.trainer.current_step % self.interval == self.interval - 1:
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")

        # https://zhuanlan.zhihu.com/p/151786842
        if not torch.isnan(out["loss"]).item():
            self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
            self.avg_wer = (0.98 * self.avg_wer) + 0.02 * out["wer"]
            self.count += 1
            self.trainer.logger.log(
                data={
                    "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                    "tr_wer": self.avg_wer / (1 - np.power(0.98, self.count)),
                },
                progress=True,
                stage="train",
            )
        return {
            "loss": out["loss"],
            "wer": out["wer"],
        }

    def before_train_loop(self, value):
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0
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
        if self.keep_train_lang is not None:
            self.model.keep_last_lang_model_train(self.keep_train_lang)
        # 语种判别固定wav2vec模型
        # if epoch <= self.froze_wav2vec_model_epoch:
        #     self.model.froze_wav2vec_model()
        #     logging.info("froze wav2vec and rnn")
        # else:
        #     self.model.unfroze_wav2vec_model()
        #     logging.info("unfroze wav2vec and rnn")

    def train_loop_end(self, outputs: List[Any] = None):
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0
        # torch.cuda.empty_cache()
        total_train_loss = 0.0
        total_train_wer = 0.0
        for item in outputs:
            total_train_loss += item["loss"]
            total_train_wer += item["wer"]
        logging.info(f"tr_avg_loss {total_train_loss / len(outputs)}")
        logging.info(f"tr_avg_wer {total_train_wer / len(outputs)}")
        self.trainer.logger.log(
            data={
                "tr_avg_loss": total_train_loss / len(outputs),
                "tr_avg_wer": total_train_wer / len(outputs),
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
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")
        if not torch.isnan(out["loss"]).item():
            self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
            self.avg_wer = (0.98 * self.avg_wer) + 0.02 * out["wer"]
            self.count += 1
            self.trainer.logger.log(
                data={
                    "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                    "val_wer": self.avg_wer / (1 - np.power(0.98, self.count)),
                },
                progress=True,
                only_tbar=True,
                stage="val",
            )
        outacc = self.infer(raw_wavs, 16000)
        index = torch.argmax(outacc[1], dim=-1)
        index = index[0].item()
        pre_lang = self.index2lang_dict[index]
        true_lang = self.index2lang_dict[batch[5][0].item()]
        return {
            "val_loss": out["loss"],
            "val_wer": out["wer"],
            "predict_texts": out["predict_texts"],
            "label_texts": out["label_texts"],
            "lang_corr": (pre_lang is true_lang)
        }

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        all_predict_texts = []
        all_label_texts = []
        lang_corr = 0
        for item in outputs:
            all_predict_texts.extend(item["predict_texts"])
            all_label_texts.extend(item["label_texts"])
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]
            if item["lang_corr"]:
                lang_corr += 1
        total_wer = self.model.model.wer_fn(all_predict_texts, all_label_texts)

        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
                "val_acc": lang_corr/len(outputs),
                "val_wer": total_wer,
                "epoch": self.trainer.current_epoch,
            },
            progress=True,
            stage="val",
            commit=False,
            only_tbar=False,
        )
        logging.info(
            f"val_wer={total_wer}, val_avg_loss={total_val_loss / len(outputs)}"
        )
        logging.info(f"val acc: {lang_corr/len(outputs)}")
        self.trainer.logger.remove_key(["loss", "wer"])
