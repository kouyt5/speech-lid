from tkinter.messagebox import NO
from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn
from s3prl_updream.interfaces import Featurizer
from s3prl_updream.wav2vec.wav2vec2_expert import UpstreamExpert
import torchmetrics


class S3prlModel(nn.Module):
    def __init__(
        self,
        pt_path: str = None,
        feature_selection: str = "hidden_states",
        dropout: float = 0.0,
        vocab_size: int = 29,
        linear_dim: int = 768,
        use_cer: bool = False,
        mask: bool = True
    ) -> None:
        super().__init__()
        drop_layer = feature_selection == "last_hidden_state"
        self.featurizer = Featurizer(
            upstream=UpstreamExpert(ckpt=pt_path, drop_layer=drop_layer, mask=mask),
            feature_selection=feature_selection,  # last_hidden_state, hidden_state_{0-24}
            upstream_device="cpu",
            layer_selection=None,  # 选择后的第几层特征 0-24
        )
        self.rnn = nn.LSTM(
            input_size=linear_dim,
            batch_first=True,  # input = (batch, seq, feature)
            hidden_size=linear_dim // 2,
            num_layers=1,
            dropout = dropout,
            bidirectional = True)
        self.last_linear = nn.Sequential(
            OrderedDict(
                {
                    "dropout": nn.Dropout(dropout),
                    "projection": nn.Linear(linear_dim, vocab_size),
                }
            )
        )
        self.loss_fn = nn.CTCLoss(blank=vocab_size - 1, reduction="none", zero_infinity=True)
        self.wer_fn = torchmetrics.WER() if not use_cer else torchmetrics.CharErrorRate()

    def forward(self, batch: List = None) -> Tuple[torch.Tensor]:
        """[summary]

        Args:
            batch (List, optional): raw speech list without padding
            example: [Tensor[0,...0.33], Tensor[0,....1.]]

        Returns:
            Tuple[torch.Tensor]: [description]
        """
        paired_features = self.featurizer.upstream(batch)
        feature = self.featurizer(batch, paired_features)
        max_len = max(batch, key=lambda x:x.shape[0]).shape[0]
        percents = [item.shape[0]/max_len for item in batch]
        feature_len = [percent*(feature.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        x = nn.utils.rnn.pack_padded_sequence(feature, enforce_sorted=False,
                                              lengths=length, batch_first=True)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        out = self.last_linear(x)
        return out

    def freeze_feature_extractor(self):
        model_freezes = []
        model_freezes.append(self.featurizer.upstream.model.feature_extractor)
        model_freezes.append(self.featurizer.upstream.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_feature_extractor(self):
        model_freezes = []
        model_freezes.append(self.featurizer.upstream.model.feature_extractor)
        model_freezes.append(self.featurizer.upstream.model.post_extract_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def freeze_tranformer_encoder(self):
        model_freezes = []
        model_freezes.append(self.featurizer.upstream.model.encoder)
        if self.featurizer.upstream.model.target_glu is not None:
            model_freezes.append(self.featurizer.upstream.model.target_glu)
        model_freezes.append(self.featurizer.upstream.model.final_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_tranformer_encoder(self):
        model_freezes = []
        model_freezes.append(self.featurizer.upstream.model.encoder)
        if self.featurizer.upstream.model.target_glu is not None:
            model_freezes.append(self.featurizer.upstream.model.target_glu)
        model_freezes.append(self.featurizer.upstream.model.final_proj)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True
