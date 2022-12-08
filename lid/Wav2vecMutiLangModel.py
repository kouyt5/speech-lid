from collections import OrderedDict
import logging
import math, sys, os

# sys.path.append(os.path.join(".."))
from typing import Dict, List
import torch
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import numpy as np
from lid.WavLMMutiLangModel import ConformerLinear
from s3prl_updream.wav2vec.wav2vec2_expert import UpstreamExpert
from s3prl_updream.interfaces import Featurizer


class Wav2vecMutiLangModel(torch.nn.Module):
    """
    基于wav2vec的多语种模型, 依次构成为:
        DataProcessor -> Wav2vecMutiModel -> LangDiscriminator
    """

    def __init__(
        self,
        pt_path: str = None,
        feature_selection: str = "hidden_states",
        dropout: float = 0.0,
        linear_dim: int = 768,
        mask: bool = True,
        num_layers: int = 1,
        lang2vocab: Dict = None,  # {"cn": 4442}) -> None
        lang2index: Dict = None,
        hidden_dim: int = 128,
        conformer_linear:bool = False,
        use_mask:bool = False,
        dim_head: int = 32,  # Conformer
        num_head: int = 8,
    ):
        super().__init__()
        self.data_processor = DataProcessor()
        self.model = Wav2vecMutiModel(
            pt_path=pt_path,
            feature_selection=feature_selection,
            dropout=dropout,
            linear_dim=linear_dim,
            mask=mask,
            num_layers=num_layers,
            lang2vocab=lang2vocab,
            conformer_linear=conformer_linear,
            use_mask=use_mask,
            dim_head=dim_head,
            num_head=num_head,
        )
        self.lang_discriminator = LangDiscriminator(
            lang2index=lang2index, lang2vocab=lang2vocab, hidden_dim=hidden_dim
        )

    def forward(self, x, sample_rate: int = 16000, lang:str=None):
        x = self.data_processor(x, sample_rate)
        x = self.model(x, lang)
        if lang is not None:
            return x, (None, None)
        lid = self.lang_discriminator(x)
        return x, lid

    def freeze_feature_extractor(self):
        model_freezes = []
        model_freezes.append(self.model.featurizer.upstream.model.feature_extractor)
        model_freezes.append(self.model.featurizer.upstream.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_feature_extractor(self):
        model_freezes = []
        model_freezes.append(self.model.featurizer.upstream.model.feature_extractor)
        model_freezes.append(self.model.featurizer.upstream.model.post_extract_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def freeze_tranformer_encoder(self):
        model_freezes = []
        model_freezes.append(self.model.featurizer.upstream.model.encoder)
        if self.model.featurizer.upstream.model.target_glu is not None:
            model_freezes.append(self.model.featurizer.upstream.model.target_glu)
        model_freezes.append(self.model.featurizer.upstream.model.final_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_tranformer_encoder(self):
        model_freezes = []
        model_freezes.append(self.model.featurizer.upstream.model.encoder)
        if self.model.featurizer.upstream.model.target_glu is not None:
            model_freezes.append(self.model.featurizer.upstream.model.target_glu)
        model_freezes.append(self.model.featurizer.upstream.model.final_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True
    
    def froze_wav2vec_model(self):
        for params in self.model.parameters():
            params.requires_grad = False
    
    def unfroze_wav2vec_model(self):
        for params in self.model.parameters():
            params.requires_grad = True


class DataProcessor(torch.nn.Module):
    """
    数据预处理, 将音频原始波形转换为模型输入的数据
    """

    def __init__(self, target_rate: int = 16000) -> None:
        super().__init__()
        self.target_rate = target_rate
        self.resampler22k = torchaudio.transforms.Resample(
            orig_freq=22050, new_freq=target_rate
        )
        self.resampler441k = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=target_rate
        )

    def forward(self, x, sample_rate: int = 16000):
        """
        输入进行resample, 如果未指明,默认为16000
        Args:
            x (List[torch.Tensor]): 输入的多个音频
            sample_rate (int): 采样率 16000 44100 etc.

        Return:
            (List[torch.Tensor]): 和输入格式相同
        """
        if sample_rate == 16000:
            return x
        longest_len = max(x, key=lambda y: y.size(-1)).shape[-1]
        wav_percent = [x[i].shape[-1] / longest_len for i in range(len(x))]
        x = pad_sequence(x, batch_first=True)
        if sample_rate == 22050:
            x = self.resampler22k(x)
        elif sample_rate == 44100:
            x = self.resampler441k(x)
        else:
            return x
        wav_len = [int(percent * x.shape[-1]) for percent in wav_percent]
        x = self.unpad_sequence(x, wav_len)
        return x

    def unpad_sequence(self, x: torch.Tensor, wav_len: List):
        """

        Args:
            x (torch.Tensor): 重采样后的数据
            wav_len (List): 重采样后长度
        """
        return [x[i, : wav_len[i]] for i in range(len(wav_len))]


class Wav2vecMutiModel(torch.nn.Module):
    """
    wav2vec模型和额外线形层, 输出语音识别结果
    """

    def __init__(
        self,
        pt_path: str = None,
        feature_selection: str = "hidden_states",
        dropout: float = 0.0,
        linear_dim: int = 768,
        mask: bool = True,
        num_layers: int = 1,
        lang2vocab: Dict = None,  # {"cn": 4442}
        use_cer: bool = True,
        conformer_linear: bool = False,
        use_mask: bool = False,
        dim_head: int = 32,  # Conformer
        num_head: int = 8,
    ) -> None:
        super().__init__()
        self.lang2vocab = lang2vocab
        drop_layer = feature_selection == "last_hidden_state"
        self.conformer_linear = conformer_linear
        self.featurizer = Featurizer(
            upstream=UpstreamExpert(ckpt=pt_path, drop_layer=drop_layer, mask=mask),
            feature_selection=feature_selection,  # last_hidden_state, hidden_state_{0-24}
            upstream_device="cpu",
            layer_selection=None,  # 选择后的第几层特征 0-24
        )
        # self.rnn = torch.nn.LSTM(
        #     input_size=linear_dim,
        #     batch_first=True,  # input = (batch, seq, feature)
        #     hidden_size=linear_dim // 2,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     bidirectional=True,
        # )
        if conformer_linear:
            self.last_projects = torch.nn.ModuleDict(
                    [
                        [
                            key,
                            ConformerLinear(
                                dropout=dropout,
                                linear_dim=linear_dim,
                                num_layers=num_layers,
                                vocab_size=lang2vocab[key],
                                double_swish=False,
                                use_mask=use_mask,
                                dim_head=dim_head,
                                num_head=num_head,
                            ),
                        ]
                        for key in lang2vocab.keys()
                    ]
                )
        else:
            logging.info("使用LSTM")
            self.last_projects = torch.nn.ModuleDict(
                [
                    [
                        key,
                        LSTMLinear(
                            dropout=dropout,
                            linear_dim=linear_dim,
                            num_layers=num_layers,
                            vocab_size=lang2vocab[key],
                        ),
                    ]
                    for key in lang2vocab.keys()
                ]
            )
        self.loss_fns = {}
        self.wer_fns = {}
        for key in lang2vocab.keys():
            self.loss_fns[key] = torch.nn.CTCLoss(
                blank=self.lang2vocab[key], reduction="none", zero_infinity=True
            )
        self.wer_fn = (
            torchmetrics.WER() if not use_cer else torchmetrics.CharErrorRate()
        )

    def forward(self, batch, lang=None):
        feature = self.featurizer.upstream(batch)
        feature = self.featurizer(batch, feature)
        max_len = max(batch, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in batch]
        feature_len = [percent * (feature.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        res = {}
        if lang is not None:
            res[lang] = self.last_projects[lang](feature, length)
        else:
            for key in self.lang2vocab.keys():
                    res[key] = self.last_projects[key](feature, length)
        return res


class LSTMLinear(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_layers: int = 1,
        vocab_size: int = 0,
    ):
        super().__init__()

        self.rnn = torch.nn.LSTM(
            input_size=linear_dim,
            batch_first=True,  # input = (batch, seq, feature)
            hidden_size=linear_dim // 2,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.dr = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(linear_dim, vocab_size + 1)

    def forward(self, feature, length):
        feature = torch.nn.utils.rnn.pack_padded_sequence(
            feature, enforce_sorted=False, lengths=length, batch_first=True
        )
        x, h = self.rnn(feature)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dr(x)
        x = self.linear(x)
        return x


class LangDiscriminator(torch.nn.Module):
    """
    语种判别器
    """

    def __init__(
        self,
        lang2vocab: Dict = None,
        lang2index: Dict = None,
        hidden_dim=128,
    ):  # {"cn": 4442}) -> None:
        super().__init__()
        self.lang2vocab = lang2vocab
        self.lang2index = lang2index
        self.classes = len(self.lang2vocab.keys())
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.classes, hidden_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, self.classes, bias=True),
        )
        self.acc = torchmetrics.Accuracy()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        tmp = x[list(x.keys())[0]]
        res_tensor = torch.zeros(size=(tmp.shape[0], self.classes), device=tmp.device)
        for lang in x.keys():
            max_value, argmax = torch.max(torch.log_softmax(x[lang], dim=-1), dim=-1)
            mask = (argmax - self.lang2vocab[lang]).bool()
            len = mask.sum(dim=-1)
            avg_confidence = torch.sum(max_value.masked_fill(~mask, 0), dim=-1) / (
                len * np.log(self.lang2vocab[lang]) + 1e-5
            )
            res_tensor[:, self.lang2index[lang]] = avg_confidence
        linear_discriminate = self.linear(res_tensor.detach())
        return res_tensor, linear_discriminate  # (B, C), (B, C)


if __name__ == "__main__":
    x = [
        torch.randn(
            16000,
        ),
        torch.randn(
            17000,
        ),
    ]
    # pt_path = "/home/cc/workdir/code/wav2vec-exp/ckpts/wav2vec_small.pt"
    # feature_selection = "last_hidden_state"
    # model = Wav2vecMutiLangModel(pt_path=pt_path, lang2vocab={"cn": 4221, "en": 100})
    # model = DataProcessor()
    x = {"cn": torch.randn((8, 100, 4112)), "en": torch.randn((8, 100, 30))}
    model = LangDiscriminator(
        lang2index={"cn": 0, "en": 1}, lang2vocab={"cn": 4111, "en": 29}, hidden_dim=128
    )
    out = model(x)
    print(out)
