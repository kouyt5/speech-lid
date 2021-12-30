from ccml.train_callback import Callback


class LrCallback(Callback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def after_train_epoch(self, value):
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]['lr']
        self.trainer.logger.log({'lr': lr}, stage="val", commit=False)