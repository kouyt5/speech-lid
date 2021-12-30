from typing import Any, List, Tuple
from ccml.ccml_module import CCMLModule
from mnist.model import MnistModel
import torch



class MnistModule(CCMLModule):
    def __init__(self, droprate: int = 0.1, hidden_dim: int = 128,lr:int = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MnistModel(droprate, hidden_dim)
        self.save_hyper_parameters({"droprate": droprate, "hidden_dim": hidden_dim})
        self.lr = lr
        
    
    def train_loop(self, batch):
        x = batch[0]
        target = batch[1]
        out = self.model(x)
        loss = self.model.loss(out, target)
        acc = self.model.acc(torch.softmax(out, dim=-1), target)
        self.trainer.logger.log(data={
            "loss": loss,
            "train_acc": acc 
        }, progress=True, stage="train")
        return {
            "loss": loss,
            "acc": acc
        }
    
    def val_loop(self, batch):
        x = batch[0]
        target = batch[1]
        out = self.model(x)
        loss = self.model.loss(out, target)
        acc = self.model.acc(torch.softmax(out, dim=-1), target)
        return {
            "val_loss": loss,
            "val_acc": acc
        }
    
    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.
        total_val_acc = 0.
        for item in outputs:
            total_val_loss+= item['val_loss']
            total_val_acc += item['val_acc']
        self.trainer.logger.log(data={
            "val_avg_loss": total_val_loss/len(outputs),
            "val_acc": total_val_acc/len(outputs),
            "epoch": self.trainer.current_epoch
        }, progress=True, stage="val", commit=False)
        
        
    def config_optim(self, *args, **kwargs) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        # )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            cooldown=3,
            min_lr=1e-4,
        )
        return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}