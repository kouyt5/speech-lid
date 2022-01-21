

from ccml.trainer import Trainer


class Callback:
    
    def __init__(self, interval:int = 1, *args, **kwargs) -> None:
        self.trainer = None
        self.interval = interval  # 间隔
        self.after_eval_epoch_count = 0
        self.after_eval_loop_count = 0
        self.after_train_epoch_count = 0
        self.after_train_loop_count = 0
        
    def add_trainer(self, trainer:Trainer):
        self.trainer = trainer
        
    def after_train_loop(self, *args, **kwargs):
        """在训练结束时调用，传入train_loop的返回值
        """
        pass
    
    def after_train_epoch(self, *args, **kwargs):
        pass
    
    def after_eval_loop(self, *args, **kwargs):
        """在验证loop结束时候调用，传入eval_loop的返回值
        """
        pass
    
    def after_eval_epoch(self, *args, **kwargs):
        """在验证结束时候调用
        """
        pass
    
    def before_train_epoch(self, *args, **kwargs):
        """在训练开始时候调用
        """
        pass
    