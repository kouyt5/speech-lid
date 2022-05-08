import logging
from typing import Any, List, Tuple, Union
import ccml
from ccml.ccml_module import CCMLModule
from tokenizer import CTCTokenizer
import torch
import torch.nn.functional as F
from s3prl_model import S3prlModel
from ccml.optim.novograd import Novograd


class Wav2vecModule(CCMLModule):
    def __init__(
        self,
        pt_path: str = None,
        feature_selection: str = "hidden_states",
        lr: float = 1e-3,
        wd: float = 0.0,
        dropout: float = 0.0,
        vocab_size: int = 29,
        metric_interval: int = 20,
        tokenizer: CTCTokenizer = None,
        reset_param: bool = False,
        freeze_decoder_epoch: int = 10,  # before warm_epoch epoch, tranformer layer was been freezed
        freeze_encoder_epoch: int = 10,
        linear_dim: int = 768,
        feature_mask: bool = True,
        use_cer: bool = False,
        optimizer_name: str = "adam",  # adam novograd
        lm_model: torch.nn.Module = None,
        num_layers:int = 1,
        glu:bool = False, 
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        logging.info(f"mask feature = {feature_mask}")
        self.model = S3prlModel(
            pt_path=pt_path,
            feature_selection=feature_selection,
            dropout=dropout,
            vocab_size=vocab_size,
            linear_dim=linear_dim,
            use_cer=use_cer,
            mask=feature_mask,
            num_layers=num_layers,
            glu=glu
        )
        if reset_param:
            logging.info("reset parameters...")
            for layer in self.model.modules():
                if hasattr(layer, "reset_parameters"):
                    logging.debug(f"reset {layer._get_name()}")
                    layer.reset_parameters()
        self.save_hyper_parameters(
            {
                "feature_selection": feature_selection,
                "dropout": dropout,
                "vocab_size": vocab_size,
                "linear_dim": linear_dim,
                "num_layers": num_layers,
                "glu": glu
            }
        )
        self.lr = lr
        self.wd = wd
        self.metric_interval = metric_interval
        self.tokenizer = tokenizer
        self.freeze_decoder_epoch = freeze_decoder_epoch
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.optimizer_name = optimizer_name
        self.lm_model = lm_model

        self.step = 0

    def common_loop(self, batch):
        wav_tensor_list = batch[0]
        text_tensor = batch[1]
        out = self.model(wav_tensor_list)
        # T*B*C
        loss = self.model.loss_fn(
            F.log_softmax(out, dim=-1).transpose(1, 0),
            text_tensor,
            (out.shape[1] * batch[2]).long().cpu(),
            (text_tensor.shape[-1] * batch[3]).long().cpu(),
        )
        loss = torch.mean(loss)

        predict = torch.argmax(out, dim=-1)
        predict_texts = self.tokenizer.ctc_decode(
            predict, predictions_len=(predict.shape[1] * batch[2]).long()
        )
        label_texts = self.tokenizer.decoder(
            text_tensor, target_lengths=(text_tensor.shape[1] * batch[3]).long()
        )
        wer = self.model.wer_fn(predict_texts, label_texts).item()

        return out, loss, wer, predict_texts, label_texts

    def train_loop(self, batch):
        out, loss, wer, predict_texts, label_texts = self.common_loop(batch)

        if self.trainer.current_step % self.metric_interval == self.metric_interval - 1:
            logging.info("current wer = " + str(wer)+'\n')
            logging.info("predict:" + predict_texts[0])
            logging.info(" labels:" + label_texts[0] + "\n")

        self.trainer.logger.log(
            data={"loss": loss, "train_wer": wer},
            progress=True,
            stage="train",
        )
        return {"loss": loss, "wer": wer}

    def before_train_loop(self, value):
        epoch = self.trainer.current_epoch
        if epoch <= self.freeze_decoder_epoch:
            self.model.freeze_tranformer_encoder()
            logging.info("freeze tranformers encoder parameter")
        else:
            self.model.unfreeze_tranformer_encoder()
        if epoch <= self.freeze_encoder_epoch:
            self.model.freeze_feature_extractor()
            logging.info("freeze feature extractor parameter")
        else:
            self.model.unfreeze_feature_extractor()
            logging.info("unfreeze feature extractor parameter")

    def val_loop(self, batch):
        out, loss, wer, predict_texts, label_texts = self.common_loop(batch)
        self.step += 1
        if self.step % self.metric_interval == self.metric_interval - 1:
            logging.info("predict:" + predict_texts[0])
            logging.info(" labels:" + label_texts[0] + "\n")
        self.trainer.logger.log(
            data={"loss": loss, "wer": wer},
            progress=True,
            only_tbar=True,
        )
        return {
            "val_loss": loss,
            "val_wer": wer,
            "predict_texts": predict_texts,
            "label_texts": label_texts,
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

        total_wer = self.model.wer_fn(all_predict_texts, all_label_texts)
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
        self.trainer.logger.remove_key(["loss", "wer"])

    def test_loop(self, batch):
        out, loss, wer, predict_texts, label_texts = self.common_loop(batch)
        lm_predict_texts = predict_texts
        # prefix_texts = self.tokenizer.parallel_ctc_prefix_search(
        #     out,
        #     (out.shape[1] * batch[2]).cpu().int(),
        #     len(self.tokenizer.export_vocab())+1,
        # )
        prefix_texts = predict_texts
        lm_predict_texts = predict_texts
        if self.lm_model is not None:
            lm_predict_texts = self.lm_model.forward(
                torch.softmax(out, dim=-1).cpu().numpy(),
                (out.shape[1] * batch[2]).cpu().int().numpy(),
            )
        self.step += 1
        if self.step % self.metric_interval == self.metric_interval - 1:
            if self.lm_model is not None:
                logging.info("     lm:" + lm_predict_texts[0])
            logging.info("predict:" + predict_texts[0])
            logging.info(" prefix:" + prefix_texts[0])
            logging.info(" labels:" + label_texts[0] + "\n")

        self.trainer.logger.log(
            data={"loss": loss, "wer": wer},
            progress=True,
            only_tbar=True,
        )
        return {
            "val_loss": loss,
            "val_wer": wer,
            "predict_texts": predict_texts,
            "label_texts": label_texts,
            "prefix_texts": prefix_texts,
            "lm_predict_texts": lm_predict_texts,
        }

    def test_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        all_predict_texts = []
        all_lm_predict_texts = []
        all_label_texts = []
        all_prefix_texts = []
        for item in outputs:
            all_predict_texts.extend(item["predict_texts"])
            all_label_texts.extend(item["label_texts"])
            all_lm_predict_texts.extend(item["lm_predict_texts"])
            all_prefix_texts.extend(item["prefix_texts"])
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]
        total_wer = self.model.wer_fn(all_predict_texts, all_label_texts)
        logging.info(f"test wer: {total_wer}")
        logging.info(f"test loss: {total_val_loss/len(outputs)}")

        prefix_wer = self.model.wer_fn(all_prefix_texts, all_label_texts)
        logging.info(f"test prefix wer: {prefix_wer}")
        if self.lm_model is not None:
            total_wer = self.model.wer_fn(all_lm_predict_texts, all_label_texts)
            logging.info(f"test wer with lm : {total_wer}")
        self.trainer.logger.log(
            data={"test_wer": total_wer},
            progress=True,
            only_tbar=False,
        )

    def train_loop_end(self, outputs: List[Any] = None):
        total_train_loss = 0.0
        total_train_wer = 0.0
        for item in outputs:
            total_train_loss += item["loss"]
            total_train_wer += item["wer"]
        self.trainer.logger.log(
            data={
                "train_avg_loss": total_train_loss / len(outputs),
                "train_avg_wer": total_train_wer / len(outputs),
            },
            progress=True,
            stage="val",
            commit=False,
            only_tbar=False,
        )

    def with_lm(self, lm_model: torch.nn.Module):
        self.lm_model = lm_model

    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        # )
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.wd
            )
        elif self.optimizer_name == "novograd":
            optimizer = Novograd(
                self.model.parameters(), lr=self.lr, weight_decay=self.wd
            )
        else:
            logging.warn(
                f"optimizer name {self.optimizer_name} is not exist, choice SGD optimizer"
            )
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd
            )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode="min",
        #     factor=0.1,
        #     patience=5,
        #     cooldown=3,
        #     min_lr=1e-6,
        # )
        from ccml.optim.tri_state import TriStageLRSchedule

        scheduler = TriStageLRSchedule(
            optimizer=optimizer,
            phase_ratio=[0.1, 0.4, 0.5],
            init_lr_scale=0.01,
            final_lr_scale=0.02,
            max_update=self.trainer.total_steps,
            lr=self.lr,
        )
        # return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}
        return optimizer, scheduler, {"monitor": None, "interval": "step"}
