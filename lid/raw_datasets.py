import csv, os
import sys

sys.path.append("../")
from email.mime import base
from email.policy import default
import random
from typing import Any, Dict, Iterator, List
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchaudio
import logging
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import augment

from ccml.cache.cache_core import cacheable
from ccml.cache.time_unit import TimeUnit
from lid.tokenizer import CTCTokenizer


class RawDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        max_duration=16.7,  # 音频最大时长, -1表示不过滤
        train=False,
        source: str = "common_voice",
    ) -> None:
        super().__init__()
        self.train = train
        # =================read datasets==================
        if source == "common_voice":
            datasets = self._get_dataset(manifest_path=manifest_path)
        else:
            datasets = self._get_dataset_xf(manifest_path=manifest_path)
        self.datasets = []
        total_filter_count = 0
        total_filter_duration = 0.0
        total_duration = 0.0
        if max_duration > 0:
            for data in datasets:
                if data["duration"] > max_duration:
                    total_filter_count += 1
                    total_filter_duration += data["duration"]
                    continue
                self.datasets.append(data)
                total_duration += data["duration"]
        logging.info(
            f"数据集语种: {self.lang()}, 过滤{total_filter_count}条," +
            f"{total_filter_duration/ 60}分钟, 总时长{total_duration / 60}, train:{self.train}"
        )
        # =================read datasets end==================

    @cacheable(cache_key="manifest_path", project="lid", time_unit=TimeUnit.MONTH)
    def _get_dataset(self, manifest_path: str = None):
        """获取数据集
        文件即为原始common-voice解压后文件目录下的tsv文件,同级目录下应该有clips文件夹, 存放mp3
        manifest文件构成:
            'client_id':'xxxxx...xxxx'
            'path':'/xxx/cv-corpus-9.0-2022-04-27/yue/clips/common_voice_yue_31204887.mp3'
            'sentence':'數字五筆冇嗰個字'
            'up_votes':'4'
            'down_votes':'0'
            'age':'twenties'
            'gender':'male'
            'accents':'母語係順德話，同時熟悉廣州音'
            'locale':'yue'
            'segment':''
        处理后字段:
            'locale':'yue'
            'path':'/xxx/cv-corpus-9.0-2022-04-27/yue/clips/common_voice_yue_31204887.mp3'
            'sentence':'數字五筆冇嗰個字'
            'duration': 16.3
        Args:
            manifest_path (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        datasets = []
        csv.register_dialect("tsv_dialect", delimiter="\t", quoting=csv.QUOTE_ALL)
        with open(manifest_path, "r", encoding="utf-8") as wf:
            reader = csv.DictReader(wf, fieldnames=None, dialect="tsv_dialect")
            for row in reader:
                data = dict(row)
                new_data = {}
                base_path = "/".join(manifest_path.split("/")[:-1])
                data["path"] = os.path.join(base_path, "clips", data["path"])
                audio_info = torchaudio.info(data["path"])
                duration = audio_info.num_frames / audio_info.sample_rate
                new_data["duration"] = duration
                new_data["path"] = data["path"]
                new_data["locale"] = data["locale"]
                new_data["sentence"] = data["sentence"]
                datasets.append(new_data)
        csv.unregister_dialect("tsv_dialect")
        return datasets

    @cacheable(cache_key="manifest_path", project="xfasr", time_unit=TimeUnit.WEEK)
    def _get_dataset_xf(self, manifest_path: str = None):
        datasets = []
        with open(manifest_path, "r") as f:
            lang = manifest_path.split("/")[-2]
            base_path = "/".join(manifest_path.split("/")[:-1])
            base_path = os.path.join(
                base_path, "wav", "train" # if self.train else "test"
            )
            for line in f.readlines():
                data = {}
                name = line.split("\t")[0]
                text = line.split("\t")[1].strip()

                data["path"] = os.path.join(base_path, name)
                audio_info = torchaudio.info(data["path"])
                duration = audio_info.num_frames / audio_info.sample_rate
                data["duration"] = duration
                data["locale"] = lang
                data["sentence"] = text
                datasets.append(data)
        return datasets

    def __getitem__(self, index) -> Any:
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def lang(self):
        """数据集语种

        Returns:
            str: 语种 zh, yue ...etc
        """
        return self.datasets[0]["locale"]

    def export_vocab(self):
        """导出词典

        Returns:
            list: 该语种的所有词
        """
        vocab = set()
        for item in self.datasets:
            for c in item["sentence"]:
                vocab.add(c)
        vocab = sorted(list(vocab))
        return vocab


class RandomSamplerWithBase(Sampler[int]):

    data_source: RawDataset

    def __init__(self, data_source: RawDataset, generator=None) -> None:
        self.data_source = data_source
        self.generator = generator
        self.base_value = 0

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def set_base_value(self, value: int):
        self.base_value = value

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        perm_list = torch.randperm(n, generator=generator).tolist()
        perm_list = [item + self.base_value for item in perm_list]
        yield from perm_list

    def __len__(self) -> int:
        return self.num_samples


class MergedDataset(Dataset):
    def __init__(
        self,
        train: bool = False,
        manifest_files: List[str] = None,
        lang2index_dict: dict = None,
        lang2tokenizer: Dict = None,
        max_duration: float = 16.7,
        source: str = "common_voice",
    ) -> None:
        super().__init__()
        self.datasets = []
        self.samplers = []
        self.lang2index_dict = lang2index_dict
        self.lang2tokenizer = lang2tokenizer
        self.train = train
        #
        for manifest_file in manifest_files:
            dataset = RawDataset(
                manifest_path=manifest_file,
                train=train,
                max_duration=max_duration,
                source=source,
            )
            sampler = RandomSamplerWithBase(data_source=dataset)
            sampler.set_base_value(len(self.datasets))
            self.samplers.append(sampler)
            self.datasets.extend(dataset.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, index) -> Any:
        """

        Args:
            index (int): index

        Returns:
            wav: is a Tensor that shape is [1, L]
            target_text: contain encoded. text shape is [L,]
            path: audio abs path

            new_data["duration"] = duration
            new_data["path"] = data["path"]
            new_data["locale"] = data["locale"]
            new_data["sentence"] = data["sentence"]

        """
        item = self.datasets[index]
        audio_path = item["path"]
        text = item["sentence"]
        wav, sr = torchaudio.load(audio_path)
        wav = self.normalize_wav(wav)  # 归一化
        if self.train:
            # wav = self.sub_secquence(wav, 0.05)  # 取音频子序列
            # dither
            wav += 1e-5 * torch.rand_like(wav)
            # preemyhasis
            wav = torch.cat(
                (wav[:, 0].unsqueeze(1), wav[:, 1:] - 0.97 * wav[:, :-1]),
                dim=1,
            )
            # speed preturb [0.9, 1, 1.1]
            speed = random.choice([0.9, 1.0, 1.1])
            pitchs = random.choice([-80, -60, -40, -20, 0, 0, 20, 40, 60, 80])
            pitchs = random.choice([-240, -200, -160, -120, -80, -40, 0, 40, 80, 120, 160, 200, 240])
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                wav,
                sr,
                [["speed", str(speed)], 
                 ["pitch", str(pitchs)], 
                 ["rate", str(sr)]],
            )  # 调速
            # reverb
            room_size = random.randint(0, 100)
            wav = augment.EffectChain().reverb(50,50,room_size).channels(1).apply(wav, src_info={'rate': sr})
            # wav = self.random_cut(wav, 0.05, 3)  # 任意裁剪三次音频
        lang = item["locale"]
        return wav, self.lang2tokenizer[lang].encoder(text), audio_path, lang

    def normalize_wav(self, wav: torch.Tensor):
        """对音频做归一化处理

        Args:
            wav (torch.Tensor): (1, T)
        """
        std, mean = torch.std_mean(wav, dim=-1)
        return torch.div(wav - mean, std + 1e-6)

    def random_cut(
        self, x: torch.Tensor, weight: float = 0.1, count: int = 1
    ) -> torch.Tensor:
        """
        对音频做随机长度裁剪
        :param x: shape(1, T)
        :return: tensor shape(1, T)
        """
        sh = x.shape
        min_len = 0
        max_len = int(sh[1] * weight)
        for i in range(count):
            target_len = int(np.random.uniform(min_len, max_len))  # 随机一个音频长度
            start_index = int(np.random.uniform(0, sh[1] - target_len))
            x[:, start_index : start_index + target_len] = 0
        return x

    def sub_secquence(self, x: torch.Tensor, weight: float = 0.1):
        """
        获取子序列
        :param x: (T)
        :param weight:
        :return:
        """
        length = x.shape[1]
        target_length = int(length * (1 - np.random.uniform(0, weight)))  # 0-0.1随机采样
        location = int(np.random.uniform(0, length - target_length))
        return x[:, location : location + target_length]

    def collate_fn(self, batch):
        wavs = [batch[i][0].squeeze(0) for i in range(len(batch))]
        texts = pad_sequence([batch[i][1] for i in range(len(batch))]).transpose(1, 0)
        audio_paths = [batch[i][2] for i in range(len(batch))]

        longest_len = max(batch, key=lambda x: x[0].shape[-1])[0].shape[-1]
        wav_percents = torch.FloatTensor(
            [batch[i][0].shape[-1] / longest_len for i in range(len(batch))]
        )
        text_percents = torch.FloatTensor(
            [batch[i][1].shape[-1] / (texts.shape[1] + 1e-9) for i in range(len(batch))]
        )
        langs = [self.lang2index_dict[batch[i][3]] for i in range(len(batch))]
        langs = torch.LongTensor(langs)
        return wavs, texts, wav_percents, text_percents, audio_paths, langs

    def export_dict(self):
        vocab = dict()
        for sampler in self.samplers:
            vocab[sampler.data_source.lang()] = sampler.data_source.export_vocab()
        return vocab


class MutiBatchSampler(Sampler[List[int]]):
    def __init__(
        self, samplers: List[Sampler[int]], batch_size: int, drop_last: bool
    ) -> None:
        self.samplers = samplers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.weight = [len(sampler) for sampler in self.samplers]
        logging.debug(f"the total samplers: {len(samplers)}")

    def get_weight_rand_index(self) -> int:
        """根据数据集比例获取对应的sampler index

        Args:
            weight (List): 包含每个数据集的大小的list. egs [100,200,0,40]

        Returns:
            int: 下标. egs 1
        """
        assert sum(self.weight) > 0
        area = random.randint(0, sum(self.weight) - 1)
        index = 0
        while area >= 0 and index < len(self.weight):
            area -= self.weight[index]
            index += 1
        return index - 1

    def __iter__(self) -> Iterator[List[int]]:
        """
        每个for .. in . 调用初始化一次

        Yields:
            Iterator[List[int]]: 一个可以迭代对象
        """

        batch = []
        iter_samplers = [iter(sampler) for sampler in self.samplers]
        remain_len = [len(sampler) for sampler in self.samplers]
        while sum(remain_len) > 0:
            # switch sampler
            index = random.randint(0, len(iter_samplers) - 1)
            # 加权
            index = self.get_weight_rand_index()

            sampler = iter_samplers[index]
            while remain_len[index] == 0:
                index -= 1
                sampler = iter_samplers[index]

            remain_total_count = remain_len[index]
            while len(batch) < self.batch_size and remain_total_count > 0:
                idx = next(sampler)
                batch.append(idx)
                remain_total_count -= 1
            remain_len[index] = remain_total_count

            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum([len(sampler) // self.batch_size for sampler in self.samplers])  # type: ignore[arg-type]
        else:
            return sum([(len(sampler) + self.batch_size - 1) // self.batch_size for sampler in self.samplers])  # type: ignore[arg-type]


def test_common_voice():
    logging.basicConfig(level=logging.NOTSET)
    file_path = "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/en/dev.tsv"
    file_path2 = (
        "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/zh-CN/train.tsv"
    )
    file_path3 = (
        "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/ja/train.tsv"
    )
    file_path4 = (
        "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/ru/train.tsv"
    )
    file_path5 = (
        "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/hi/train.tsv"
    )
    file_path6 = (
        "/home/cc/workdir/data/common-voice/cv-corpus-9.0-2022-04-27/zh-TW/train.tsv"
    )

    tokenizer_dict = {
        "en": CTCTokenizer("/home/cc/workdir/code/lid/data/en-vocab.txt"),
        "zh-CN": CTCTokenizer("/home/cc/workdir/code/lid/data/zh-CN-vocab.txt"),
        "ja": CTCTokenizer("/home/cc/workdir/code/lid/data/ja-vocab.txt"),
        "ru": CTCTokenizer("/home/cc/workdir/code/lid/data/ru-vocab.txt"),
        "hi": CTCTokenizer("/home/cc/workdir/code/lid/data/hi-vocab.txt"),
        "zh-TW": CTCTokenizer("/home/cc/workdir/code/lid/data/zh-TW-vocab.txt"),
    }
    lang2index_dict = {
        "en": 2,
        "zh-CN": 0,
        "ja": 3,
        "ru": 4,
        "hi": 5,
        "zh-TW": 1,
    }

    merge_dataset = MergedDataset(
        manifest_files=[
            file_path,
            file_path2,
            file_path3,
            file_path4,
            file_path5,
            file_path6,
        ],
        lang2tokenizer=tokenizer_dict,
        lang2index_dict=lang2index_dict,
    )

    # 生成vocab文件
    # vocab_dict = merge_dataset.export_dict()
    # for lang, vocab in vocab_dict.items():
    #     vocab_path = os.path.join("lid/data", lang + "-vocab.txt")
    #     with open(vocab_path, "w") as f:
    #         for v in vocab:
    #             f.write(v + "\n")
    ####################生成结束########################
    batch_sample = MutiBatchSampler(
        merge_dataset.samplers, batch_size=8, drop_last=False
    )
    dataloader = DataLoader(
        dataset=merge_dataset,
        batch_sampler=batch_sample,
        num_workers=6,
        collate_fn=merge_dataset.collate_fn,
        pin_memory=True,
        # batch_size=8,
        # shuffle=True,
    )

    for i, batch in enumerate(dataloader):
        print(batch)
    # dataloader = DataLoader()


def test_xf():
    logging.basicConfig(level=logging.NOTSET)
    file_path = "/home/cc/workdir/code/lid/data/xf/data/Persian/train.label"
    file_path2 = "/home/cc/workdir/code/lid/data/xf/data/Swahili/train.label"
    file_path3 = "/home/cc/workdir/code/lid/data/xf/data/Vietnamese/train.label"

    tokenizer_dict = {
        "Persian": CTCTokenizer(
            "/home/cc/workdir/code/lid/data/xf/data/Persian-vocab.txt"
        ),
        "Swahili": CTCTokenizer(
            "/home/cc/workdir/code/lid/data/xf/data/Swahili-vocab.txt"
        ),
        "Vietnamese": CTCTokenizer(
            "/home/cc/workdir/code/lid/data/xf/data/Vietnamese-vocab.txt"
        ),
    }
    lang2index_dict = {"Persian": 0, "Swahili": 1, "Vietnamese": 2}

    merge_dataset = MergedDataset(
        manifest_files=[
            file_path,
            file_path2,
            file_path3,
        ],
        lang2tokenizer=tokenizer_dict,
        lang2index_dict=lang2index_dict,
        source="xf",
        train=True,
    )

    # 生成vocab文件
    vocab_dict = merge_dataset.export_dict()
    for lang, vocab in vocab_dict.items():
        vocab_path = os.path.join(
            "/home/cc/workdir/code/lid/data/xf/data", lang + "-vocab.txt"
        )
        with open(vocab_path, "w") as f:
            for v in vocab:
                f.write(v + "\n")
    ####################生成结束########################
    batch_sample = MutiBatchSampler(
        merge_dataset.samplers, batch_size=1, drop_last=False
    )
    dataloader = DataLoader(
        dataset=merge_dataset,
        batch_sampler=batch_sample,
        num_workers=0,
        collate_fn=merge_dataset.collate_fn,
        pin_memory=True,
        # batch_size=8,
        # shuffle=True,
    )

    for i, batch in enumerate(dataloader):
        print(batch)
    # dataloader = DataLoader()


if __name__ == "__main__":
    # test_common_voice()
    test_xf()
