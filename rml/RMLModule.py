from collections import defaultdict
import logging
from typing import Any, List, Tuple

from ccml.ccml_module import CCMLModule
from mnist.model import MnistModel
import torch
import wandb
import numpy as np

from rml.Resnet1d import CCMLResNet1D


class RMLModule(CCMLModule):
    def __init__(
        self,
        lr: float = 0.01,
        in_channels: int = 2,
        base_filters: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        groups: int = 1,
        n_block: int = 5,
        n_classes: int = 11,
        optim_name: str = "sgd",
        dr: float = 0.5,
        rnn: bool = False,
        increasefilter_gap: int = 4,
        wc: float = 1e-4,
        use_snr_info: bool = False,
        snr_loss_weight: float = 0.1,
        mix_up:bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = CCMLResNet1D(
            in_channels,
            base_filters,
            kernel_size,
            stride,
            groups,
            n_block,
            n_classes,
            dr=dr,
            rnn=rnn,
            increasefilter_gap=increasefilter_gap,
            use_snr_info=use_snr_info,
        )
        self.save_hyper_parameters(
            {
                "base_filters": base_filters,
                "stride": stride,
                "n_classes": n_classes,
                "n_block": n_block,
                "kernel_size": kernel_size,
                "rnn": rnn,
                "use_snr_info": use_snr_info,
            }
        )
        self.lr = lr
        self.optim_name = optim_name
        self.wc = wc
        self.snr_loss_weight = snr_loss_weight
        self.mix_up = mix_up
        
    def train_loop(self, batch):
        x = batch[0]
        x2 = batch[3]
        target = batch[1]
        target2 = batch[4]
        factor = 1.
        if self.mix_up:
            factor = np.random.beta(0.5, 0.5)
            # factor = 0.8 + 0.2*factor
            x = factor * x + (1-factor) * x2
        out, out_snr = self.model(x)
        cross_loss = factor * self.model.loss(out, target) + (1-factor) * self.model.loss(out, target2)
        snr_loss = factor * self.model.loss(out_snr, batch[2]) + (1-factor) * self.model.loss(out_snr, batch[5])
        loss = (1 - self.snr_loss_weight) * cross_loss + self.snr_loss_weight * snr_loss
        acc = self.model.acc(torch.softmax(out, dim=-1), target)
        self.trainer.logger.log(
            data={"loss": loss, "train_acc": acc}, progress=True, stage="train"
        )
        return {"loss": loss, "acc": acc}

    def val_loop(self, batch):
        x = batch[0]
        target = batch[1]
        out, out_snr = self.model(x)
        loss = (1 - self.snr_loss_weight) * self.model.loss(
            out, target
        ) + self.snr_loss_weight * self.model.loss(out_snr, batch[2])
        acc_snr = self.model.acc(torch.softmax(out_snr, dim=-1), batch[2])
        acc = self.model.acc(torch.softmax(out, dim=-1), target)
        return {"val_loss": loss, "val_acc": acc, "acc_snr": acc_snr}

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        total_val_acc = 0.0
        total_val_acc_snr = 0.0
        for item in outputs:
            total_val_loss += item["val_loss"]
            total_val_acc += item["val_acc"]
            total_val_acc_snr += item["acc_snr"]
        self.trainer.logger.log(
            data={
                "val_loss": total_val_loss / len(outputs),
                "val_acc": total_val_acc / len(outputs),
                "acc_snr": total_val_acc_snr / len(outputs),
                "epoch": self.trainer.current_epoch,
            },
            progress=True,
            stage="val",
            commit=False,
        )

    def test_loop(self, batch):
        x = batch[0]
        target = batch[1].cpu().numpy().tolist()
        snr = batch[2]
        out, out_snr = self.model(x)
        acc_snr = self.model.acc(torch.softmax(out_snr, dim=-1), batch[2])
        pred = torch.argmax(out, dim=-1)
        pred = pred.cpu().numpy().tolist()  # [0,2,3,4]
        pred_snr = torch.argmax(out_snr, dim=-1).cpu().numpy().tolist()
        target_snr = snr.cpu().numpy().tolist()
        res = [pred[i] == target[i] for i in range(len(pred))]
        res_snr = [pred_snr[i] == target_snr[i] for i in range(len(target_snr))]
        return {
            "snr": snr.cpu().numpy().tolist(),
            "res": res,
            "res_snr": res_snr,
        }  # [10, -10, 5] [True, False, True]

    def test_loop_end(self, outputs: List[Any] = None):
        res_true = defaultdict(int)
        res_total = defaultdict(int)
        res_snr_true = defaultdict(int)
        for item in outputs:
            snrs = item["snr"]
            reses = item["res"]
            reses_snr = item["res_snr"]
            for snr, res, res_snr in zip(snrs, reses, reses_snr):
                if res:
                    res_true[snr] += 1
                if res_snr:
                    res_snr_true[snr] += 1
                res_total[snr] += 1
        data = [
            [(key - 10) * 2, res_true[key] / res_total[key]] for key in res_true.keys()
        ]
        table = wandb.Table(data=data, columns=["snr", "acc"])
        wandb.log(
            {
                "acc over test": wandb.plot.line(
                    table, "snr", "acc", title="acc on 2016a over diff snr"
                )
            }
        )
        logging.info(f"test acc: {sum([x[1] for x in data])/len([x[1] for x in data])}")
        data_snr = [
            [(key - 10) * 2, res_snr_true[key] / res_total[key]]
            for key in res_snr_true.keys()
        ]
        table = wandb.Table(data=data_snr, columns=["snr", "acc"])
        wandb.log(
            {
                "snr": wandb.plot.line(
                    table, "snr", "acc", title="snr predict acc on 2016a over diff snr"
                )
            }
        )
        logging.info(
            f"test snr acc: {sum([x[1] for x in data_snr])/len([x[1] for x in data_snr])}"
        )

    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        optimizer = None
        if self.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wc
            )
        elif self.optim_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.wc
            )
        else:
            logging.error(f"{self.optim_name} is not support...")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            cooldown=3,
            min_lr=1e-4,
            verbose=True,
        )
        return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}
