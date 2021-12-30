import logging
from typing import List
from ccml.train_callback import Callback
import os
import queue
import torch
import uuid


"""
保存checkpoint回调
"""


class CkptCallback(Callback):
    def __init__(
        self,
        interval: int = 1,
        ckpt_path: str = "ckpt",
        save_topk: int = 1,
        file_name_metric: List = ["epoch", "avg_val_loss"],
        metric: str = "avg_val_loss",
        manager: str = "min",  # 可选max
        *args,
        **kwargs,
    ) -> None:
        super().__init__(interval=interval, *args, **kwargs)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        self.ckpt_path = os.path.abspath(ckpt_path)
        self.parttern = file_name_metric
        self.save_topk = save_topk
        self.metric = metric
        self.topk_queue = queue.PriorityQueue(maxsize=self.save_topk)
        self.manager = manager  # 指标越小越好还是越大越好，python优先队列默认最小

    def parse2abspath(self, value: dict, file_name_metric: List):
        """将用户定义的文件名解析成绝对路径名

        Args:
            value (dict): 结果字典
            name (str): 文件名包含的指标，例如 ['epoch', 'avg_val_loss', 'wer']

        Returns:
            返回指标名和对应值组成的文件路径， 例如 epoch_21avg_val_loss_19.43wer_0.11.ckpt
        """
        file_name = ""
        for name in file_name_metric:
            if name in value.keys():
                file_name += str(name) + "_"
                if isinstance(value[name], int):
                    file_name += str(value[name]) + "_"
                else:
                    file_name += "{:.2f}".format(name) + "_"
            else:  # 一定在all_val_results中
                total_values = 0.
                total_count = 0.
                if "all_val_results" not in value.keys():
                    logging.warning(f"结果中没有key all_val_results, break")
                    continue
                for item in value["all_val_results"]:
                    if name not in item.keys():
                        logging.warning(f"结果中没有key{name}, break")
                        break
                    total_values += item[name]
                    total_count += 1
                if total_count != 0:
                    file_name += "{:.2f}".format(total_values/total_count) + "_"
        if len(file_name) == 0:
            file_name = "default_"+ str(uuid.uuid4())[:4]
        file_name = file_name .strip("_") + ".pt"
        return os.path.join(self.ckpt_path, file_name)

    def get_state(self) -> dict:
        """获取当前trainer的状态字典

        Returns:
            dict: 状态字典
        """
        state = {}

        state["model"] = self.trainer.model.state_dict()
        state["hyper_parameters"] = self.trainer.ccml_module.get_hyper_parameters()
        state["epoch"] = self.trainer.current_epoch
        state["optimizer"] = self.trainer.optimizer.state_dict()
        state["scalar"] = self.trainer.scalar.state_dict()
        state["logger"] = self.trainer.logger.state_dict()
        if self.trainer.lr_scheduler is not None:
            state["lr_scheduler"] = self.trainer.lr_scheduler.state_dict()

        return state

    def after_eval_epoch(self, value: dict):
        self.after_eval_epoch_count += 1
        if self.after_eval_epoch_count % self.interval != 0:
            return
        if self.trainer.local_rank > 0:  # 非master节点不保存ckpt
            return
        state = self.get_state()
        torch.save(state, os.path.join(self.ckpt_path, "last.pt"))

        if self.topk_queue.full():
            [priority, old_path] = self.topk_queue.get()  # 获取优先级最高的元素

            metric = self.result_has_key(value, self.metric)
            if metric is not None:
                # 保存topk
                save_path = self.parse2abspath(value, self.parttern)
                # 从队列中取一个最小值（优先级最高）
                if self.manager == "min":
                    if metric < 1 / priority:
                        priority = 1 / metric
                        self.topk_queue.put([priority, save_path])
                        torch.save(state, save_path)
                        os.remove(old_path)
                        logging.info(f"save a ckpt in {save_path}")
                elif self.manager == "max":
                    if metric > priority:
                        priority = metric
                        self.topk_queue.put([priority, save_path])
                        torch.save(state, save_path)
                        os.remove(old_path)
                        logging.info(f"save a ckpt in {save_path}")
        else:
            metric = self.result_has_key(value, self.metric)
            save_path = self.parse2abspath(value, self.parttern)
            torch.save(state, save_path)
            self.topk_queue.put([1/metric, save_path])
            logging.info(f"save a ckpt in {save_path}")

    def result_has_key(self, target: dict, key: str):
        """遍历返回结果，查看是否包含key，有则返回对应的value

        Args:
            target (dict): 目标字典
            key (str): 查询的key
        """
        if not isinstance(target, dict):
            logging.warning("target is not a dict")
            return None
        if key in target.keys():
            return target[key]

        # 遍历验证集的结果中的value
        if key not in target["all_val_results"][0]:
            return None

        total_values = 0.0
        total_count = 0.0
        for item in target["all_val_results"]:
            total_values += item[key]
            total_count += 1
        return total_values / total_count


if __name__ == "__main__":
    ckpt_callback = CkptCallback()
