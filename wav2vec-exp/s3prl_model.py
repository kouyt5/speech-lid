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
    ) -> None:
        super().__init__()
        drop_layer = feature_selection == "last_hidden_state"
        self.featurizer = Featurizer(
            upstream=UpstreamExpert(ckpt=pt_path, drop_layer=drop_layer),
            feature_selection=feature_selection,  # last_hidden_state, hidden_state_{0-24}
            upstream_device="cpu",
            layer_selection=None,  # 选择后的第几层特征 0-24
        )
        self.last_linear = nn.Sequential(
            OrderedDict(
                {
                    "dropout": nn.Dropout(dropout),
                    "projection": nn.Linear(linear_dim, vocab_size),
                }
            )
        )
        self.loss_fn = nn.CTCLoss(blank=vocab_size - 1, reduction="none")
        self.wer_fn = torchmetrics.WER()

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
        out = self.last_linear(feature)
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
