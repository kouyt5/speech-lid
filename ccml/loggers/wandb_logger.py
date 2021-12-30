from typing import Any, Dict
import wandb
import logging
import torch.nn as nn

from ccml.loggers.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str = None,  # 项目名，例如asr
        entity: str = None,  # 用户名 kouyt5
        name: str = None,  # 实验的名字
        wandb_id: str = None,  # 实验唯一标志
        group: str = None,
    ):
        if wandb_id is not None:
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                resume="allow",
                save_code=True,
                id=wandb_id,
                group=group,
            )
            logging.info(f"wandb 初始化完成: checkpoint=" + str(self.checkpint_path))
        else:
            wandb.init(
                project=project, entity=entity, name=name, save_code=True, group=group
            )  # 新建一个wandb
            logging.info("新建wandb成功")
            wandb.run.log_code(".")

    def get_checkpoint_by_name(self, name: str, path: str = None):
        """获取wandb的checkpoint

        Returns:
            str: 本地的checkpoint路径
        """
        return wandb.restore(name, run_path=path)  # 从本地或者wandb获取checkpoint

    def log(self, data: Dict[str, Any], *args, **kwargs):
        wandb.log(data, step=None, *args, **kwargs)

    def watch_model(self, model: nn.Module, *args, **kwargs):
        super().watch_model(model, *args, **kwargs)
        wandb.watch(models=model, *args, **kwargs)


if __name__ == "__main__":
    import time
    from tqdm import tqdm

    wb = WandbLogger(project="asr", entity="kouyt5", name="test")

    for i in tqdm(range(1000)):
        wb.log(data={"train/loss": i})
        wb.log(data={"val/loss": i / 10})
        time.sleep(0.4)
    import torch

    torch.save()
    wandb.save()
