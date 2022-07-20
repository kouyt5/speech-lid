import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ccml.ccml_module import CCMLModule
from ccml.optim.novograd import Novograd
from ccml.optim.tri_state import TriStageLRSchedule
from lid.Wav2vecMutiLangModel import Wav2vecMutiLangModel
from lid.WavLMMutiLangModel import WavLMMutiLangModel


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
        use_wav2vec:bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param
        self.scheduler = scheduler
        self.scheduler_param = scheduler_param
        self.lang2index_dict = lang2index_dict
        self.interval = interval
        self.freeze_tranformer_epoch = freeze_tranformer_epoch
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.froze_wav2vec_model_epoch = froze_wav2vec_model_epoch
        self.index2lang_dict = {}
        for key in self.lang2index_dict.keys():
            self.index2lang_dict[self.lang2index_dict[key]] = key
        self.tokenizer_dict = tokenizer_dict
        
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
            )
        self.count = 1
        self.avg_loss = 0
        self.avg_wer = 0
        self.avg_acc = 0
        self.avg_acc_asr = 0

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
        out, (lid_asr, lid_linear) = self.model(wavs, 22050)
        lang = self.index2lang_dict[langs[0].item()]
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

        # lid acc
        loss_lid = 0
        lid_acc_asr = self.model.lang_discriminator.acc(
                torch.softmax(lid_asr, dim=-1), langs
            )
        lid_acc = lid_acc_asr
        if self.trainer.current_epoch < self.froze_wav2vec_model_epoch:   
            lid_acc = self.model.lang_discriminator.acc(
                torch.softmax(lid_linear, dim=-1), langs
            )
            loss_lid = self.model.lang_discriminator.cross_entropy(lid_linear, langs)
        
        return {
            "loss": loss
            if self.trainer.current_epoch > self.froze_wav2vec_model_epoch
            else loss_lid,  # 如果只做语种识别，则ctc loss忽略
            "wer": wer,
            "lang": lang,
            "predict_texts": predict_texts,
            "label_texts": label_texts,
            "lid_acc": lid_acc,
            "lid_acc_asr": lid_acc_asr,
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

    def infer(
        self, x: List[torch.Tensor], lid_raw: bool = True, sample_rate: int = 22050
    ):
        out, (lid_asr, lid_linear) = self.model(
            x, sample_rate
        )  # {"cn": (1, T, V)}, (1 * C)
        predict_texts = {}
        for lang in out.keys():
            predict_texts.put(
                lang,
                self.tokenizer_dict[lang].ctc_decode(
                    torch.argmax(out, dim=-1),
                ),
            )
        lid = self.lid_infer(lid_asr if lid_raw else lid_linear)
        return predict_texts, lid

    def train_loop(self, batch):
        # out.keys(): "loss" "wer" "lang" "predict_texts" "label_texts" lid_acc
        torch.cuda.empty_cache()
        out = self.common_loop(batch)
        if self.trainer.current_step % self.interval == self.interval - 1:
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")
            logging.info(f"val acc: {self.avg_acc}, val acc asr: {self.avg_acc_asr}")

        # https://zhuanlan.zhihu.com/p/151786842
        self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
        self.avg_wer = (0.98 * self.avg_wer) + 0.02 * out["wer"]
        self.avg_acc = (0.98 * self.avg_acc) + 0.02 * out["lid_acc"].item()
        self.avg_acc_asr = (0.98 * self.avg_acc_asr) + 0.02 * out["lid_acc_asr"].item()
        self.count += 1
        self.trainer.logger.log(
            data={
                "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                "tr_wer": self.avg_wer / (1 - np.power(0.98, self.count)),
                "tr_acc": self.avg_acc / (1 - np.power(0.98, self.count)),
                "tr_acc2": self.avg_acc_asr / (1 - np.power(0.98, self.count)),
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
        self.avg_acc = 0
        self.avg_acc_asr = 0
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
        self.avg_acc = 0
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
        self.avg_loss = (0.98 * self.avg_loss) + 0.02 * out["loss"].item()
        self.avg_wer = (0.98 * self.avg_wer) + 0.02 * out["wer"]
        self.avg_acc = (0.98 * self.avg_acc) + 0.02 * out["lid_acc"].item()
        self.count += 1
        if self.count % self.interval == self.interval - 1:
            logging.info("wer: " + str(out["wer"]))
            logging.info(f"predict_text: {out['predict_texts'][0]}")
            logging.info(f"label_text:   {out['label_texts'][0]}")
        self.trainer.logger.log(
            data={
                "loss": self.avg_loss / (1 - np.power(0.98, self.count)),
                "val_wer": self.avg_wer / (1 - np.power(0.98, self.count)),
                "val_acc": self.avg_acc / (1 - np.power(0.98, self.count)),
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
            "lid_acc": out["lid_acc"],
            "lid_acc_asr": out["lid_acc_asr"],
        }

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        total_val_acc = 0.0
        total_val_acc_asr = 0.0
        all_predict_texts = []
        all_label_texts = []
        for item in outputs:
            all_predict_texts.extend(item["predict_texts"])
            all_label_texts.extend(item["label_texts"])
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]
            total_val_acc += item["lid_acc"]
            total_val_acc_asr += item["lid_acc_asr"]

        total_wer = self.model.model.wer_fn(all_predict_texts, all_label_texts)
        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
                "val_wer": total_wer,
                "epoch": self.trainer.current_epoch,
                "val_acc": total_val_acc / len(outputs),
                "val_acc_asr": total_val_acc_asr / len(outputs),
            },
            progress=True,
            stage="val",
            commit=False,
            only_tbar=False,
        )
        logging.info(
            f"val_wer={total_wer}, val_avg_loss={total_val_loss / len(outputs)}"
        )
        logging.info(
            f"val_acc={total_val_acc/(len(outputs))}, val_acc_asr={total_val_acc_asr / len(outputs)}"
        )
        self.trainer.logger.remove_key(["loss", "wer"])
