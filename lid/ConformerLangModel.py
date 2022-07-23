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
from lid.conformer import ConformerBlock, ConformerModel


class ConformerMutiLangModel(torch.nn.Module):
    """
    基于wav2vec的多语种模型, 依次构成为:
        DataProcessor -> Wav2vecMutiModel -> LangDiscriminator
    """

    def __init__(
        self,
        num_layers: int = 1,
        lang2vocab: Dict = None,  # {"cn": 4442}) -> None
        lang2index: Dict = None,
        hidden_dim: int = 32,  # 语种识别模型隐藏层维度
        use_cer: bool = True,
        conformer_linear: bool = False,
        dropout: float = 0.0,  # 最后的线性映射层dropout
        linear_dim: int = 144,  # 最后线性层输入维度
        n_blocks: int = 14,
        win_len=0.025,
        hop_length: float = 0.01,
        sr=16000,
        n_mels: int = 80,
        encoder_dim: int = 144,  # 和linear_dim保持一致
        t_mask_prob: float = 0.05,  # 时域mask概率
        f_mask=27,
        mask_times: int = 2,  # mask次数
        dim_head=64,  # att head 维度
        last_dim_head:int = 32,  # 最后一层的head维度
        heads=4,  # att head数
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        double_swish=False,
        sub_sampling:int = 2,
        
    ):
        super().__init__()
        self.data_processor = DataProcessor()
        self.model = ConformerMutiModel(
            num_layers = num_layers,
            lang2vocab = lang2vocab,  # {"cn": 4442}
            use_cer = use_cer,
            conformer_linear = conformer_linear,
            dropout = dropout,  # 最后的线性映射层dropout
            linear_dim = linear_dim,  # 最后线性层输入维度
            n_blocks = n_blocks,  # Conformer 模型参数
            win_len= win_len,
            hop_length = hop_length,
            sr=sr,
            n_mels = n_mels,
            encoder_dim = encoder_dim,
            t_mask_prob = t_mask_prob,
            f_mask=f_mask,
            mask_times = mask_times,
            dim_head=dim_head,
            last_dim_head=last_dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            double_swish=double_swish,
            sub_sampling=sub_sampling,
            
        )
        self.lang_discriminator = LangDiscriminator(
            lang2index=lang2index, lang2vocab=lang2vocab, hidden_dim=hidden_dim
        )

    def forward(self, x, sample_rate: int = 16000, lang: str = None):
        x = self.data_processor(x, sample_rate)
        x = self.model(x, lang)
        if lang is not None:
            return x, (None, None)
        lid = self.lang_discriminator(x)
        return x, lid

    def freeze_feature_extractor(self):
        model_freezes = []
        if hasattr(self.model.featurizer, "model"):
            model_freezes.append(self.model.featurizer.model.feature_extractor)
            model_freezes.append(self.model.featurizer.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_feature_extractor(self):
        model_freezes = []
        if hasattr(self.model.featurizer, "model"):
            model_freezes.append(self.model.featurizer.model.feature_extractor)
            model_freezes.append(self.model.featurizer.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def freeze_tranformer_encoder(self):
        model_freezes = []
        if hasattr(self.model.featurizer, "model"):
            model_freezes.append(self.model.featurizer.model.encoder)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_tranformer_encoder(self):
        model_freezes = []
        if hasattr(self.model.featurizer, "model"):
            model_freezes.append(self.model.featurizer.model.encoder)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def reset_param(self):
        logging.info("reset parameters...")
        for layer in self.model.modules():
            if hasattr(layer, "reset_parameters"):
                logging.debug(f"reset {layer._get_name()}")
                layer.reset_parameters()
            else:
                logging.debug(f"reset ignore {layer._get_name()}")


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


class ConformerMutiModel(torch.nn.Module):
    """
    Conformer模型和额外线形层, 输出语音识别结果
    """

    def __init__(
        self,
        num_layers: int = 1,  # LSTM 层数, 如果使用conformer作为最后一层该参数无效
        lang2vocab: Dict = None,  # {"cn": 4442}
        use_cer: bool = True,
        conformer_linear: bool = False,
        dropout: float = 0.0,  # 最后的线性映射层dropout
        linear_dim: int = 768,  # 最后线性层输入维度
        n_blocks: int = 14,
        win_len=0.025,
        hop_length: float = 0.01,
        sr=16000,
        n_mels: int = 80,
        encoder_dim: int = 144,
        t_mask_prob: float = 0.05,
        f_mask=27,
        mask_times: int = 2,
        dim_head=64,
        last_dim_head: int = 32,
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        double_swish=False,
        sub_sampling:int = 2,
    ) -> None:
        super().__init__()
        self.lang2vocab = lang2vocab
        self.conformer_linear = conformer_linear
        self.featurizer = ConformerModel(
            n_blocks = n_blocks,  # Conformer 模型参数
            win_len= win_len,
            hop_length = hop_length,
            sr=sr,
            n_mels = n_mels,
            encoder_dim = encoder_dim,
            t_mask_prob = t_mask_prob,
            f_mask=f_mask,
            mask_times = mask_times,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            double_swish=double_swish,
            sub_sampling=sub_sampling,
            )  # (N, T, C), egs (4, 102, 80)
        logging.info("使用Conformer模型")
        if conformer_linear:
            logging.info("使用Conformer分类层")
            self.last_projects = torch.nn.ModuleDict(
                [
                    [
                        key,
                        ConformerLinear(
                            dropout=dropout,
                            linear_dim=linear_dim,
                            num_layers=num_layers,
                            vocab_size=lang2vocab[key],
                            double_swish=double_swish,
                            dim_head=last_dim_head
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

    def forward(self, batch, lang: str = None):

        feature = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        pad_mask = torch.ones_like(feature, device=feature.device)
        for i in range(len(batch)):
            pad_mask[i, : batch[i].size(0)] = 0
        feature = self.featurizer(feature, pad_mask)
        
        max_len = max(batch, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in batch]
        feature_len = [percent * (feature.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        if not self.conformer_linear:
            feature = torch.nn.utils.rnn.pack_padded_sequence(
                feature, enforce_sorted=False, lengths=length, batch_first=True
            )
        res = {}
        if lang is not None:
            res[lang] = self.last_projects[lang](feature)
        else:
            for key in self.lang2vocab.keys():
                res[key] = self.last_projects[key](feature)
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

    def forward(self, feature):
        x, h = self.rnn(feature)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dr(x)
        x = self.linear(x)
        return x


class ConformerLinear(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_layers: int = 1,
        vocab_size: int = 0,
        double_swish: bool = False,
        dim_head: int = 32,
    ) -> None:
        super().__init__()
        self.block = ConformerBlock(
            dim=linear_dim,
            dim_head=dim_head,  # 64
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0,
            ff_dropout=0,
            conv_dropout=0,
            double_swish=double_swish,
        )
        self.dr = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(linear_dim, vocab_size + 1)

    def forward(self, x):
        x = self.block(x)
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
            torch.nn.ReLU(inplace=False),
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
    model = WavLMMutiLangModel(
        lang2vocab={"cn": 4221, "en": 100},
        lang2index={"cn": 0, "en": 1},
    )
    model.eval()
    # model = DataProcessor()
    # x = {"cn": torch.randn((8, 100, 4112)), "en": torch.randn((8, 100, 30))}
    # model = LangDiscriminator(
    #     lang2index={"cn": 0, "en": 1}, lang2vocab={"cn": 4111, "en": 29}, hidden_dim=128
    # )
    out = model(x, 22050)
    print(out)
