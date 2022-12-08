from collections import OrderedDict
import logging
import math, sys, os

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("."))

from typing import Dict, List
import torch
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import numpy as np
from lid.conformer import ConformerBlock, ConformerModel
from lid.wavlm.example import WavLMModel


class WavLMMutiLangModel(torch.nn.Module):
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
        conformer_linear: bool = False,
        double_swish: bool = False,
        use_pre_train: bool = True,
        mask_channel_prob: float = 0,
        mask_prob: float = 0.,
        conformer_pure: bool = False,
        use_mask: bool = False,
        dim_head: int = 32,
        num_head: int = 8,
    ):
        super().__init__()
        self.data_processor = DataProcessor()
        self.model = WavLMMutiModel(
            pt_path=pt_path,
            feature_selection=feature_selection,
            dropout=dropout,
            linear_dim=linear_dim,
            mask=mask,
            num_layers=num_layers,
            lang2vocab=lang2vocab,
            conformer_linear=conformer_linear,
            double_swish=double_swish,
            use_pre_train=use_pre_train,
            mask_channel_prob=mask_channel_prob,
            mask_prob=mask_prob,
            conformer_pure=conformer_pure,
            use_mask=use_mask,
            dim_head=dim_head,
            num_head=num_head,
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
    
    def keep_last_lang_model_train(self, lang):
        model_freezes = []
        for item_lang in self.model.last_projects.keys():
            if lang == item_lang:
                continue;
            model_freezes.append(self.model.last_projects[item_lang])
            logging.info(f"freeze lang model: {item_lang}")
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

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


class WavLMMutiModel(torch.nn.Module):
    """
    wav2vec模型和额外线形层, 输出语音识别结果
    """

    def __init__(
        self,
        pt_path: str = None,
        feature_selection: str = "hidden_states",
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_layers: int = 1,
        lang2vocab: Dict = None,  # {"cn": 4442}
        use_cer: bool = True,
        conformer_linear: bool = False,
        double_swish: bool = False,
        use_pre_train: bool = True,
        mask: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
        conformer_pure: bool = False,
        use_mask: bool = False,
        dim_head: int = 32,
        num_head: int = 8,
    ) -> None:
        super().__init__()
        self.lang2vocab = lang2vocab
        self.conformer_linear = conformer_linear
        logging.info(f"mask channel prob: {mask_channel_prob}, mask_prob {mask_prob}")
        if not mask:
            mask_channel_prob = 0.0
            mask_prob = 0
        if conformer_pure:
            self.featurizer = ConformerModel()  # (N, T, C), egs (4, 102, 80)
            logging.info("使用Conformer模型")
        else:
            self.featurizer = WavLMModel(pt_path, use_pre_train, mask_channel_prob, mask_prob)
            logging.info("使用WavLM预训练模型")
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
                                double_swish=double_swish,
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
                        ConformerLSTMLinear(
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
        res = {}
        if lang is not None:
                res[lang] = self.last_projects[lang](feature, percents)
        else:
            for key in self.lang2vocab.keys():
                res[key] = self.last_projects[key](feature, percents)
        return res


class ConformerLinear(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_layers: int = 1,
        vocab_size: int = 0,
        double_swish: bool = False,
        use_mask: bool = False,
        dim_head: int = 32,
        num_head: int = 8,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_mask = use_mask
        logging.info(f"Conformer nums layers: {num_layers}")
        if num_layers == 1:
            self.block = ConformerBlock(
                dim=linear_dim,
                dim_head=dim_head,  # 32
                heads=num_head,  # 8
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=0,
                ff_dropout=0,
                conv_dropout=0,  # dropout
                double_swish=double_swish,
            )
        else:
            self.block = torch.nn.ModuleList()
            for i in range(num_layers):
                self.block.append(
                    ConformerBlock(
                        dim=linear_dim,
                        dim_head=dim_head,  # 32
                        heads=num_head,  # 8
                        ff_mult=4,
                        conv_expansion_factor=2,
                        conv_kernel_size=31,
                        attn_dropout=0,
                        ff_dropout=0,
                        conv_dropout=0,  # dropout
                        double_swish=double_swish,
                    )
                )
        
        self.dr = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(linear_dim, vocab_size + 1)

    def forward(self, x, percents):
        if self.use_mask:
            feature_len = [percent * (x.shape[1]) for percent in percents]
            mask = torch.ones_like(x)
            for length in feature_len:
                mask[:, :length, :] = 0
            x = x.masked_fill(mask.bool(), 0)
        if self.num_layers == 1:
            x = self.block(x)
        else:
            for block in self.block:
                x = block(x)
        x = self.dr(x)
        x = self.linear(x)
        return x

class ConformerLSTMLinear(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_layers: int = 1,
        vocab_size: int = 0,
        double_swish: bool = False,
    ) -> None:
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

    def forward(self, x, percents):
        feature_len = [percent * (x.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, enforce_sorted=False, lengths=length, batch_first=True
        )
        x, h = self.rnn(x)
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
    pt_path = "/home/cc/workdir/code/lid/wavlm/ckpts/WavLM-Base-plus.pt"
    feature_selection = "last_hidden_state"
    model = WavLMMutiLangModel(
        pt_path=pt_path,
        lang2vocab={"cn": 4221, "en": 100},
        lang2index={"cn": 0, "en": 1},
        conformer_linear=False,
    )
    model.eval()
    # model = DataProcessor()
    # x = {"cn": torch.randn((8, 100, 4112)), "en": torch.randn((8, 100, 30))}
    # model = LangDiscriminator(
    #     lang2index={"cn": 0, "en": 1}, lang2vocab={"cn": 4111, "en": 29}, hidden_dim=128
    # )
    out = model(x, 22050)
    print(out)
