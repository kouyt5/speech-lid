import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ccml.ccml_module import CCMLModule
from ccml.optim.novograd import Novograd
from ccml.optim.tri_state import TriStageLRSchedule
from lid.ConformerLangModel import ConformerMutiLangModel


class LidSuperviseModule(CCMLModule):
    def __init__(
        self,
        optimizer_name: str = "adam",
        optimizer_param: Dict = None,
        scheduler: str = "reduce",
        scheduler_param: Dict = None,
        interval: int = 10,
        lang2index_dict: Dict = None,
        tokenizer_dict: Dict = None,
        lang2vocab: Dict = None,  # {"cn": 4442}) -> None
    
        num_layers: int = 1,  # LSTM层数
        hidden_dim: int = 32,  # 语种识别模型隐藏层维度
        use_cer: bool = True,
        conformer_linear: bool = True,
        dropout: float = 0.0,  # 最后的线性映射层dropout
        linear_dim: int = 144,  # 最后线性层输入维度
        n_blocks: int = 14,
        win_len=0.025,
        hop_length: float = 0.01,
        sr=16000,
        n_mels: int = 80,
        encoder_dim: int = 144,  # 和linear_dim保持一致
        t_mask_prob: float = 0.05,  # 时域mask概率
        f_mask=27,
        mask_times: int = 2,  # mask次数
        dim_head=64,  # att head 维度
        last_dim_head:int=32,  # 最后层的head维度
        heads=4,  # att head数
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        double_swish=False,
        sub_sampling:int=2,
        *args,
        **kwargs,
    ):
        super().__init__(
            lang2vocab=lang2vocab,
            lang2index_dict=lang2index_dict,
            tokenizer_dict=tokenizer_dict,
        
            num_layers=num_layers,
            hidden_dim=hidden_dim,  # 语种识别隐藏层
            conformer_linear = conformer_linear,
            linear_dim = linear_dim,  # 最后线性层输入维度
            n_blocks = n_blocks,  # Conformer 模型参数
            win_len= win_len,
            hop_length = hop_length,
            sr=sr,
            n_mels = n_mels,
            encoder_dim = encoder_dim,
            dim_head=dim_head,
            last_dim_head=last_dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            double_swish=double_swish,
            lang2index=lang2index_dict,  # 语种判别模块
            sub_sampling=sub_sampling,
        )
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param
        self.scheduler = scheduler
        self.scheduler_param = scheduler_param
        self.lang2index_dict = lang2index_dict
        self.interval = interval
        self.freeze_tranformer_epoch = -1
        self.freeze_encoder_epoch = -1
        self.froze_wav2vec_model_epoch = -1
        self.sr = sr
        self.index2lang_dict = {}
        for key in self.lang2index_dict.keys():
            self.index2lang_dict[self.lang2index_dict[key]] = key
        self.tokenizer_dict = tokenizer_dict

        logging.info(f"采样率: {sr}")
        logging.info(f"使用double swish: {double_swish}")
        self.model = ConformerMutiLangModel(
            num_layers = num_layers,
            lang2vocab = lang2vocab,  # {"cn": 4442}
            use_cer = use_cer,
            conformer_linear = conformer_linear,
            dropout = dropout,  # 最后的线性映射层dropout
            linear_dim = linear_dim,  # 最后线性层输入维度
            n_blocks = n_blocks,  # Conformer 模型参数
            win_len= win_len,
            hop_length = hop_length,
            sr=sr,
            n_mels = n_mels,
            encoder_dim = encoder_dim,
            t_mask_prob = t_mask_prob,
            f_mask=f_mask,
            mask_times = mask_times,
            dim_head=dim_head,
            last_dim_head=last_dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            double_swish=double_swish,
            
            lang2index=lang2index_dict,  # 语种判别模块
            hidden_dim=hidden_dim,
            sub_sampling=sub_sampling,
        )
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0

    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_param)
        if self.optimizer_name == "adam":
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
                final_lr_scale=0.02,
                max_update=self.trainer.total_steps,
                lr=self.optimizer_param["lr"],
            )
        # return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}
        return optimizer, scheduler, {"monitor": None, "interval": "step"}

    def common_loop(self, batch) -> Dict:
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
        predict_texts = self.tokenizer_dict[lang].ctc_decode(
            torch.argmax(out, dim=-1),
            predictions_len=(torch.argmax(out, dim=-1).shape[1] * batch[2]).long(),
        )
        label_texts = self.tokenizer_dict[lang].decoder(
            texts, target_lengths=(texts.shape[1] * text_percents).long()
        )
        wer = self.model.model.wer_fn(predict_texts, label_texts).item()

        return {
            "loss": loss,
            "wer": wer,
            "lang": lang,
            "predict_texts": predict_texts,
            "label_texts": label_texts,
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
            [x[0,:]], sample_rate, language
        )  # {"cn": (1, T, V)}, (1 * C)
        predict_texts = {}
        for lang in out.keys():
            predict_texts[lang] = self.tokenizer_dict[lang].ctc_decode(
                torch.argmax(out[lang], dim=-1),
            )
        return predict_texts, lid_asr, out

    def train_loop(self, batch):
        # out.keys(): "loss" "wer" "lang" "predict_texts" "label_texts" lid_acc
        torch.cuda.empty_cache()
        out = self.common_loop(batch)
        if self.trainer.current_step % self.interval == self.interval - 1:
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")

        # https://zhuanlan.zhihu.com/p/151786842
        if (not torch.isnan(out["loss"]).item()):
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

    def train_loop_end(self, outputs: List[Any] = None):
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
        out = self.common_loop(batch)
        if self.count % self.interval == self.interval - 1:
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")
        if (not torch.isnan(out["loss"]).item()):
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
        return {
            "val_loss": out["loss"],
            "val_wer": out["wer"],
            "predict_texts": out["predict_texts"],
            "label_texts": out["label_texts"],
        }

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        all_predict_texts = []
        all_label_texts = []
        for item in outputs:
            all_predict_texts.extend(item["predict_texts"])
            all_label_texts.extend(item["label_texts"])
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]

        total_wer = self.model.model.wer_fn(all_predict_texts, all_label_texts)
        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
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
        self.trainer.logger.remove_key(["loss", "wer", "_runtime", "_timestamp"])
