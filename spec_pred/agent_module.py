import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ccml.ccml_module import CCMLModule
from ccml.optim.novograd import Novograd
from ccml.optim.tri_state import TriStageLRSchedule
from spec_pred.CausalConvNet import MLP, TCN, CnnLSTM, SimpleLSTM
from spec_pred.Resnet1d import Transformer


class SpecModule(CCMLModule):
    def __init__(
        self,
        optimizer_name: str = "adam",
        optimizer_param: Dict = None,
        scheduler: str = "reduce",
        scheduler_param: Dict = None,
        loss_fn_name: str = "l1",
        model_name="TCN",  # tcn, lstm
        # tcn model param
        input_size=512,
        output_size=512,
        num_channels=[1024, 2048, 1024, 512],
        kernel_size=4,
        dropout=0.2,
        # lstm model param
        hidden_size=1024,
        num_layers=2,
        
        mean: float = -562.5556880622805,
        std: float = 127.53728596715978,
    ):
        super().__init__(
            model_name=model_name,
            input_size=input_size,
            output_size=output_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            hidden_size=hidden_size,
            num_layers=num_layers,
            mean=mean,
            std=std,
        )
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param
        self.scheduler = scheduler
        self.scheduler_param = scheduler_param
        self.mean = mean
        self.std = std

        if model_name == "TCN":
            self.model = TCN(
                input_size=input_size,
                output_size=output_size,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif model_name == "LSTM":
            self.model = SimpleLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_name == "MLP":
            self.model = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        elif model_name == "CNNLSTM":
            self.model = CnnLSTM(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif model_name == "TRANSFORMER":
            self.model = Transformer(
                in_size=input_size,
                out_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        
        if loss_fn_name == "mse":
            self.loss_fn = F.mse_loss
        elif loss_fn_name == "l1":
            self.loss_fn = F.l1_loss

    def config_optim(self):
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
                final_lr_scale=0.02,
                max_update=self.trainer.total_steps,
                lr=self.optimizer_param["lr"],
            )
        return optimizer, scheduler, {"monitor": None, "interval": "step"}

    @torch.no_grad()
    def infer(self, x, win_len, pred_len):
        """预测 pred_len 时间长度的谱

        Args:
            x (np.ndarray): B * T * D (T >= win_len)
            pred_len (int): 预测长度
        """
        out = torch.zeros((x.shape[0], pred_len, x.shape[2]), device=x.device)
        for i in range(pred_len):
            pred = self.model(x[:,-win_len:, :])  # -> B * D
            x = torch.cat([x, pred.unsqueeze(1)], dim=1)
            pred = pred * (1e-9 + self.std) + self.mean  # 去正则化
            out.narrow(1, i, 1).copy_(pred.unsqueeze(1))
        return out
    
    def common_loop(self, batch) -> Dict:
        x = batch[0]  # B * win_len * D
        label = batch[1]  # B * D
        out = self.model(x)
        loss = self.loss_fn(out, label, reduction="mean")

        return {"loss": loss, "pred": out}

    def train_loop(self, batch):
        res = self.common_loop(batch)
        self.trainer.logger.log(
            data={
                "loss": res["loss"],
            },
            progress=True,
            stage="train",
        )
        return {
            "loss": res["loss"],
        }
        
    def train_loop_end(self, outputs: List[Any] = None):
        torch.cuda.empty_cache()
        total_train_loss = 0.0
        for item in outputs:
            total_train_loss += item["loss"]
        logging.info(f"tr_avg_loss {total_train_loss / len(outputs)}")
        self.trainer.logger.log(
            data={
                "tr_avg_loss": total_train_loss / len(outputs),
            },
            progress=False,
            stage="val",
            commit=False,
            only_tbar=False,
        )
        self.trainer.train_dataset.factor_add()
        
    def val_loop(self, batch):
        res = self.common_loop(batch)
        self.trainer.logger.log(
            data={
                "loss": res["loss"],
            },
            progress=True,
            only_tbar=True,
            stage="val",
        )
        return {
            "val_loss": res["loss"],
        }
        
    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        for item in outputs:
            if torch.isnan(item["val_loss"]).item():
                logging.warning("loss is nan, it will be ignore..")
                continue
            total_val_loss += item["val_loss"]
        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
                "epoch": self.trainer.current_epoch,
            },
            progress=True,
            stage="val",
            commit=False,
            only_tbar=False,
        )
        logging.info(
            f"val_avg_loss: {total_val_loss / len(outputs)}, epoch: {self.trainer.current_epoch}"
        )
        self.trainer.logger.remove_key(["loss", "_runtime", "_timestamp"])
        
    def test_loop(self, batch):
        res = self.common_loop(batch)
        self.trainer.logger.log(
            data={
                "loss": res["loss"],
            },
            progress=True,
            only_tbar=True,
            commit=False,
            stage="val",
        )
        return {
            "test_loss": res["loss"],
        }
        
    def test_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        for item in outputs:
            total_val_loss += item["test_loss"]
        self.trainer.logger.log(
            data={
                "test_loss": total_val_loss / len(outputs),
            },
            progress=True,
            stage="val",
            commit=True,
        )
        logging.info(
            f"test_avg_loss: {total_val_loss / len(outputs)}"
        )