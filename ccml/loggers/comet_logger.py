from typing import Any, Dict
import logging
import torch.nn as nn
from comet_ml import Experiment

from ccml.loggers.base_logger import BaseLogger


class CometLogger(BaseLogger):
    def __init__(
        self,
        api_key: str = None,
        project: str = None,  # 项目名，例如asr
        entity: str = None,  # 用户名 kouyt5
        name: str = None,  # 实验的名字
        code_path: str = None,
    ):
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project,
            workspace=entity,
        )
        self.experiment.set_name(name)
        self.step = 1
        if code_path is not None:
            self.experiment.log_code(folder=code_path)

    def get_checkpoint_by_name(self, name: str, path: str = None):
        return None

    def log(self, data: Dict[str, Any], commit:bool=False, *args, **kwargs):
        self.experiment.log_metrics(data, step=self.step, *args, **kwargs)
        self.step += 1
