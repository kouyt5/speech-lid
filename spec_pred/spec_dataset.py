from collections import defaultdict
import logging

# logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',level=logging.DEBUG)
import pickle
from tqdm import tqdm
from typing import DefaultDict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import json
import numpy as np
import random
from ccml.cache.cache_core import cacheable

from ccml.cache.time_unit import TimeUnit


class DataSource:
    def __init__(
        self,
        data_path: str = None,
        split: List[float] = [0.8, 0.1, 0.1],
        spec_range: List[float] = [71000, 76500],
    ):
        self.datas = self.__read_json(data_path=data_path, spec_range=spec_range)  # T*F
        print("数据shape: " + str(self.datas.shape))
        time_len = self.datas.shape[0]

        self.train_len = int(split[0] * time_len)
        self.train_start = 0

        self.dev_len = int(split[1] * time_len)
        self.dev_start = self.train_len

        self.test_len = time_len - self.train_len - self.dev_len
        self.test_start = self.train_len + self.dev_len
        self.mean = np.mean(self.datas[: self.train_len])
        self.std = np.std(self.datas[: self.train_len])
        self.min = np.min(self.datas[: self.train_len])
        logging.info(f"mean: {self.mean} , std: {self.std}")

    @cacheable(
        cache_key="data_path",
        project="spec_pred",
        time_unit=TimeUnit.WEEK,
        cache_extra_param="spec_range",
    )
    def __read_json(
        self, data_path: str = None, spec_range: Tuple[int] = (71000, 76500)
    ):
        datas = []
        all_time = []
        with open(data_path) as f:
            print("data loading...")
            for line in tqdm(f):
                item = json.loads(line)
                data = item["data"]
                data = list(data)
                datas.append(data[spec_range[0] : spec_range[1]])
                all_time.append(item["date"])
        datas = np.asanyarray(datas)  # T*F
        return datas


class SpecDataset(Dataset):
    def __init__(
        self,
        train_type: str = "train",
        aug: bool = True,
        datasource: DataSource = None,
        win_len: int = 40,
        aug_factor: float = 0.01,
        mask: bool = False,
        exchange: bool = False,
    ):
        super().__init__()
        self.datasource = datasource
        if train_type == "train":
            self.data_len = self.datasource.train_len
            self.data_start = self.datasource.train_start
        elif train_type == "dev":
            self.data_len = self.datasource.dev_len
            self.data_start = self.datasource.dev_start
        else:
            self.data_len = self.datasource.test_len
            self.data_start = self.datasource.test_start
        self.win_len = win_len  # 历史时间窗口大小
        self.rand_factor = 0  # 随机数种子，控制每一批次的数据起点偏移
        self.aug = aug
        self.aug_factor = aug_factor
        self.mask = mask
        self.exchange = exchange
        logging.info(f"{train_type} win_len: {self.win_len}")

    def factor_add(self):
        self.rand_factor = (self.rand_factor + 1) % self.win_len

    def __getitem__(self, index):
        start_loc = (
            self.data_start + self.rand_factor + index * (self.win_len + 1)
        )  # 数据起始下标
        end_loc = start_loc + self.win_len  # 数据结束下标
        data = self.datasource.datas[start_loc : end_loc + 1, :]
        
        data = torch.as_tensor(data, dtype=torch.float32)
        # deoise
        mask = data < -994 + 80
        data = torch.masked_fill(data, mask, -1390)
        
        data = (data - self.datasource.mean) / (self.datasource.std + 1e-9)
        if self.aug:
            # rand replace
            if self.exchange:
                choice = random.choices(range(len(data)), k=2)
                tmp = data[choice[0]]
                data[choice[0]] = data[choice[1]]
                data[choice[1]] = tmp
            # speech aug
            if self.mask:
                mask_len = random.randint(0, int(0.1 * self.win_len))
                start = random.randint(0, len(data) - 1 - mask_len)
                data[start : start + mask_len] = 0

        return data

    def __len__(self):
        # win_len * F
        return torch.as_tensor(
            (self.data_len - self.rand_factor) // (self.win_len + 1)
        ).int()

    def collate_fn(self, batch):
        x = torch.cat(
            [
                batch[i][
                    : self.win_len,
                ].unsqueeze(0)
                for i in range(len(batch))
            ]
        )  # B * win_len * F
        if self.aug:
            x = x + torch.randn(x.shape) * self.aug_factor
        label = torch.cat(
            [batch[i][self.win_len, :].unsqueeze(0) for i in range(len(batch))]
        )  # B * F

        return x, label


if __name__ == "__main__":
    data_path = "/data/chenc//data.json"
    datasource = DataSource(
        data_path, split=[0.8, 0, 1, 0, 1], spec_range=[73456, 73968]
    )
    train_dataset = SpecDataset("train", aug=True, datasource=datasource, win_len=40)
    dataloader = DataLoader(
        train_dataset,
        2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )
    for i, batch in enumerate(dataloader):
        print(batch)
