from collections import defaultdict
import logging
import pickle
import random
from typing import DefaultDict
from torch.utils.data import Dataset, DataLoader
import torch


class RML16Dataset(Dataset):
    def __init__(
        self, data: dict = None, key_mapping: dict = None, aug: bool = False
    ) -> None:
        """_summary_

        Args:
            path (str): 文件路径

        """
        super().__init__()
        self.data = data
        self.key_mapping = key_mapping
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, snr, data = self.data[index]
        data = torch.from_numpy(data)
        data = (
            data - torch.mean(data)
        ) / (torch.std(data) + 1e-9)

        # data aug
        if self.aug:
            data = 1e-2 * torch.rand_like(data) + data  # 噪声注入
            mask_len = random.randint(0, 60)  # 30
            point = random.randint(0, data.size(1) - mask_len)
            data[:, point : point + mask_len] = torch.randn((2, mask_len))
            # data[0, point : point + mask_len] = torch.randn((mask_len,))
            # mask_len = random.randint(0, 60)  # 30
            # point = random.randint(0, data.size(1) - mask_len)
            # data[1, point : point + mask_len] = torch.randn((mask_len,))
        return key, snr, data

    def collate_fn(self, batch):
        data = torch.cat(
            [batch[i][2].unsqueeze(0) for i in range(len(batch))], dim=0
        )  # N * 2 * 128
        key = torch.LongTensor(
            [self.key_mapping[batch[i][0]] for i in range(len(batch))]
        )  # N
        snrs = torch.LongTensor(
            [self.snr2index(batch[i][1]) for i in range(len(batch))]
        )

        index = torch.randperm(len(batch))
        data2 = torch.cat(
            [batch[index[i].item()][2].unsqueeze(0) for i in range(len(batch))], dim=0
        )  # N * 2 * 128
        key2 = torch.LongTensor(
            [self.key_mapping[batch[index[i].item()][0]] for i in range(len(batch))]
        )  # N
        snrs2 = torch.LongTensor(
            [self.snr2index(batch[index[i].item()][1]) for i in range(len(batch))]
        )
        return data, key, snrs, data2, key2, snrs2

    def snr2index(self, snr):
        return (snr + 20) // 2


class RML16aDatasetManager:
    def __init__(
        self, path: str, scale: list = [8, 1, 1], split_type: str = "seen"
    ) -> None:
        """_summary_

        Args:
            path (str): 文件路径
            scale (list, optional): _description_. Defaults to [8,1,1].
            split_type (str, optional): 是否信噪比可见. Defaults to "seen". "unseen"表示训练和测试条件匹配
        """
        # (b'QPSK', 2): np.array(1000, 2, 128)
        # 220 key -> 1000 sample
        self.x = pickle.load(open(path, "rb"), encoding="latin-1")
        self.keymap = defaultdict(list)
        for key_tuple in list(self.x.keys()):
            self.keymap[key_tuple[0]].append(key_tuple[1])
        self.data = {"train": list(), "val": list(), "test": list()}
        train_scale = scale[0] / sum(scale)
        val_scale = scale[1] / sum(scale)
        test_scale = scale[2] / sum(scale)
        if split_type == "seen":
            for key, snr in self.x.keys():
                value = self.x[(key, snr)]
                # train
                for i in range((int)(len(value) * train_scale)):
                    self.data["train"].append((key, snr, value[i]))
                for i in range(
                    (int)(len(value) * train_scale),
                    (int)(len(value) * (train_scale + val_scale)),
                ):
                    self.data["val"].append((key, snr, value[i]))
                for i in range(
                    (int)(len(value) * (train_scale + val_scale)), len(value)
                ):
                    self.data["test"].append((key, snr, value[i]))
        if split_type == "unseen":
            for key in self.keymap:
                snr = self.keymap[key]
                # train
                for i in range((int)(len(snr) * train_scale)):
                    for value in self.x[(key, snr[i])]:
                        self.data["train"].append((key, snr[i], value))
                for i in range(
                    (int)(len(snr) * train_scale),
                    (int)(len(snr) * (train_scale + val_scale)),
                ):
                    for value in self.x[(key, snr[i])]:
                        self.data["val"].append((key, snr[i], value))
                for i in range((int)(len(snr) * (train_scale + val_scale)), len(snr)):
                    for value in self.x[(key, snr[i])]:
                        self.data["test"].append((key, snr[i], value))

    def get_key_mapping(self):

        keys = list(self.keymap.keys())
        keys = sorted(keys)
        res = {}
        for i in range(len(keys)):
            res[keys[i]] = i
        logging.info(f"调制方式mapping:{res}, 长度为{len(keys)}")
        return res

    def get_data(self, stage: str = "train"):
        """_summary_

        Args:
            stage (str, optional): 训练阶段 train val test. Defaults to "train".

        Returns:
            _type_: _description_
        """
        return self.data[stage]


if __name__ == "__main__":
    data_manager = RML16aDatasetManager(
        "/home/cc/workdir/data/rml/RML2016.10a_dict.pkl",
        scale=[8, 1, 1],
        split_type="unseen",
    )

    dataset = RML16Dataset(
        data_manager.get_data("train"), data_manager.get_key_mapping()
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn
    )
    for data, i in enumerate(dataloader):
        data
        import soundfile as sf

        sf.write("rml.wav", data=i[0][0].numpy().T, samplerate=200 * 1000)
