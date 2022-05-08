import math
from tkinter.messagebox import NO
from turtle import forward
from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn
import torchaudio
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
        mask: bool = True,
        num_layers: int = 1,
        glu: bool = False,
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
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.use_glu = glu
        if glu:
            self.glu = MutuGLU(linear_dim, 80, n_fft=320 * 2, num_layers=6, dropout=0.0, hidden_size=1024)
        self.last_linear = nn.Sequential(
            OrderedDict(
                {
                    "dropout": nn.Dropout(dropout),
                    "projection": nn.Linear(
                        linear_dim + 1024 if self.use_glu else linear_dim, vocab_size
                    ),
                }
            )
        )
        self.loss_fn = nn.CTCLoss(
            blank=vocab_size - 1, reduction="none", zero_infinity=True
        )
        self.wer_fn = (
            torchmetrics.WER() if not use_cer else torchmetrics.CharErrorRate()
        )

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
        max_len = max(batch, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in batch]
        feature_len = [percent * (feature.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        x = nn.utils.rnn.pack_padded_sequence(
            feature, enforce_sorted=False, lengths=length, batch_first=True
        )
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.use_glu:
            x = self.glu(x, batch)
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


class MutuGLU(nn.Module):
    def __init__(
        self, ssl_size=768, fbank_size=80, n_fft=320, num_layers=2, dropout=0.0, hidden_size=256,
    ) -> None:
        super().__init__()
        self.ssl_size = ssl_size
        self.fbank_size = fbank_size
        self.fbank = FBank(fbank_size=fbank_size, n_fft=n_fft)
        self.rnn = nn.LSTM(
            input_size=fbank_size,
            batch_first=True,  # input = (batch, seq, feature)
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.ssl_linear = nn.Linear(ssl_size, hidden_size)
        self.fbank_linear = nn.Linear(hidden_size, ssl_size)

    def forward(self, ssl_x, fbank_x):
        fbank_in = [self.fbank(x).transpose(0, 1) for x in fbank_x]
        fbank_in = torch.nn.utils.rnn.pad_sequence(fbank_in, batch_first=True)
        # fbank_in = fbank_in  # [:, :, 1:-1]
        if ssl_x.shape[1] < fbank_in.shape[1]:
            fbank_in = fbank_in[:, : ssl_x.shape[1], :]

        if ssl_x.shape[1] > fbank_in.shape[1]:
            ssl_x = ssl_x[:, : fbank_in.shape[1], :]

        max_len = max(fbank_x, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in fbank_x]
        feature_len = [percent * (fbank_in.shape[1]) for percent in percents]
        length = torch.floor(torch.Tensor(feature_len)).long()
        fbank_in = nn.utils.rnn.pack_padded_sequence(
            fbank_in, enforce_sorted=False, lengths=length, batch_first=True
        )
        fbank_in, h = self.rnn(fbank_in)
        fbank_in, _ = nn.utils.rnn.pad_packed_sequence(fbank_in, batch_first=True)
        return torch.cat(
            [
                ssl_x * (self.fbank_linear(fbank_in).sigmoid()),
                fbank_in * (self.ssl_linear(ssl_x).sigmoid()),
            ],
            # [ssl_x, fbank_in],
            dim=-1,
        )


class FBank(nn.Module):
    def __init__(self, fbank_size, n_fft) -> None:
        super().__init__()

        self.fbank_size = fbank_size
        self.n_fft = n_fft

    def forward(self, x):
        import torchaudio.functional as F
        
        fb = F.melscale_fbanks(            
                            n_freqs=self.n_fft // 2 + 1,
                            f_min=0,
                            f_max = 8000,
                            sample_rate = 16000,
                            n_mels = self.fbank_size,
                            norm=None,
                            mel_scale="htk"
                        )
        spec = F.spectrogram(x, pad=0, window=torch.hann_window(self.n_fft).to(x.device),
                             n_fft=self.n_fft,hop_length=self.n_fft//2, 
                             win_length=self.n_fft, power=2.0, 
                             normalized=False ,center=False, pad_mode="reflect").float()
        spec = torch.matmul(spec.transpose(-1, -2), fb.to(spec.device)).transpose(-1, -2).float()
        spec = F.amplitude_to_DB(spec, multiplier=10.0, 
                               amin=1e-10,
                               db_multiplier=0,
                               top_db=None).float()
        # torchaudio.transforms.MelSpectrogram
        std, mean = torch.std_mean(spec)
        return (spec - mean) / (std + 1e-9)


if __name__ == "__main__":
    model_ssl = S3prlModel(
        pt_path="/home/cc/workdir/code/wav2vec-exp/ckpts/wav2vec_small.pt",
        feature_selection="last_hidden_state",
    )
    model = MutuGLU(29, 80, 320 * 2)
    for i in range(16000, 48000, 10):
        x = torch.randn((i,))
        out = model_ssl([x])
        # ssl_x = torch.randn((1, 49, 1024))
        fbank = [x]
        out = model(out, fbank)
        print(out.shape)
