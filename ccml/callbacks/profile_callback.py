from ccml.train_callback import Callback
from ccml.utils.profile import _time_cost_recoder


class ProfileCallback(Callback):
    
    def after_train_epoch(self, *args, **kwargs):
        _time_cost_recoder.format_print()