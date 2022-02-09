import imp
import json
from typing import Any, Union
from torch.utils.data import Dataset, DataLoader
import logging
import torch
from tokenizer import CTCTokenizer
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class RawAudioDatasets(Dataset):
    def __init__(
        self,
        manifest_path: Union[str, list],
        max_duration=16.7,
        mask=False,
        sr=16000,
        tokenizer: CTCTokenizer = None,
        upper:bool = False
    ) -> None:
        super().__init__()
        self.mask = mask
        self.sr = sr
        self.tokenizer = tokenizer
        # =================read datasets==================
        self.datasets = []
        total_filter_count = 0
        total_filter_duration = 0.0
        total_duration = 0.
        if isinstance(manifest_path, list):
            for item in manifest_path:
                with open(item, encoding="utf-8") as f:
                    for line in f.readlines():
                        data = json.loads(line, encoding="utf-8")
                        if upper:
                            data['text'] = data['text'].upper()
                        if data["duration"] > max_duration:
                            total_filter_count += 1
                            total_filter_duration += data["duration"]
                            continue
                        self.datasets.append(data)
                        total_duration += data["duration"]
                    total_filter_duration = total_filter_duration / 60
        elif isinstance(manifest_path, str):
            with open(manifest_path, encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line, encoding="utf-8")
                    if upper:
                            data['text'] = data['text'].upper()
                    if data["duration"] > max_duration:
                        total_filter_count += 1
                        total_filter_duration += data["duration"]
                        continue
                    self.datasets.append(data)
                    total_duration += data["duration"]
                total_filter_duration = total_filter_duration / 60
        logging.info("过滤音频条数:{:d}条".format(total_filter_count))
        logging.info("过滤音频时长:{:.2f}分钟".format(total_filter_duration))
        logging.info("数据集时长:{:.2f}分钟".format(total_duration / 60))
        # =================read datasets end==================

    def __getitem__(self, index) -> Any:
        """

        Args:
            index (int): index

        Returns:
            wav: is a Tensor that shape is [1, L]
            target_text: contain encoded. text shape is [L,]
            path: audio abs path

        """
        audio_path = self.datasets[index]["audio_filepath"]
        text = self.datasets[index]["text"]
        wav, sr = torchaudio.load(audio_path)
        # dither
        wav += 1e-5 * torch.rand_like(wav)
        # preemyhasis
        wav = torch.cat(
            (wav[:, 0].unsqueeze(1), wav[:, 1:] - 0.97 * wav[:, :-1]),
            dim=1,
        )

        return wav, self.tokenizer.encoder(text), audio_path

    def __len__(self):
        return len(self.datasets)

    def collate_fn(self, batch):
        wavs = [batch[i][0].squeeze(0) for i in range(len(batch))]
        texts = pad_sequence([batch[i][1] for i in range(len(batch))]).transpose(1,0)
        audio_paths = [batch[i][2] for i in range(len(batch))]

        longest_len = max(batch, key=lambda x: x[0].shape[-1])[0].shape[-1]
        wav_percents = torch.FloatTensor(
            [batch[i][0].shape[-1] / longest_len for i in range(len(batch))]
        )
        text_percents = torch.FloatTensor(
            [batch[i][1].shape[-1] / texts.shape[1] for i in range(len(batch))]
        )
        return wavs, texts, wav_percents, text_percents, audio_paths
