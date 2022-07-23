import logging
import math
from typing import Tuple, Union
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import random

from einops import rearrange
from einops.layers.torch import Rearrange


# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# helper classes


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class DoubleSwish(nn.Module):
    """Swish 改进 https://mp.weixin.qq.com/s/IWFPpA6JMqdkItSbMBRJrA"""

    def forward(self, x):
        return x * (x - 1).sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
        double_swish=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish() if not double_swish else DoubleSwish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
        double_swish=False,
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish() if not double_swish else DoubleSwish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        double_swish=False,
    ):
        super().__init__()
        self.ff1 = FeedForward(
            dim=dim,
            mult=ff_mult,
            dropout=ff_dropout,
            double_swish=double_swish,
        )
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
            double_swish=double_swish,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        #
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class FBank(nn.Module):
    def __init__(
        self,
        win_len=0.025,
        hop_length: float = 0.01,
        sr=16000,
        n_mels: int = 80,
        t_mask_prob: float = 0.05,
        f_mask=27,
        mask_times: int = 2,
    ) -> None:
        super().__init__()
        self.t_mask_prob = t_mask_prob
        self.mask_time = mask_times
        self.n_mels = n_mels
        win_length = int(win_len * sr)
        hop_length = int(hop_length * sr)
        self.mel = torchaudio.transforms.MelSpectrogram(
            n_fft=512,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
            # norm="slaney",
            onesided=True,
        )
        self.mel_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.f_mask = torchaudio.transforms.FrequencyMasking(f_mask)
        self.t_stretch = torchaudio.transforms.TimeStretch(
            hop_length=hop_length, n_freq=n_mels
        )

    def forward(self, x):
        """_summary_

        Args:
            x torch.Tensor: (N, T)

        Returns:
            FBank特征 torch.Tensor: (N, T, C)
        """
        x = self.mel(x)  # -> (N, n_mels, time)
        x = self.mel_to_db(x)

        # x = (
        #     torchaudio.compliance.kaldi.fbank(x, num_mel_bins=self.n_mels)
        #     .transpose(0, 1)
        #     .unsqueeze(0)
        # )  # (T, C) -> (C, T) -> (1, C, T)
        # std, mean = torch.std_mean(x)
        # x = (x - mean) / (std + 1e-6)
        if self.training:
            # x = self.t_stretch(x, random.choice([0.9, 1.0, 1.1])).float()
            for i in range(self.mask_time):
                # time mask
                x = torchaudio.functional.mask_along_axis(
                    x, int(x.size(2) * self.t_mask_prob), 0.0, 2  # , 1.0
                )
                x = self.f_mask(x)
        # norm
        return x.permute(0, 2, 1)  # (B, T, C)


class Conv1dSubSampling2(nn.Module):
    def __init__(self, idim: int, odim: int) -> None:
        super().__init__()
        self.sub_sampling = nn.Sequential(
            nn.Conv1d(idim, idim, kernel_size=3, stride=2, padding=1), Swish()
        )
        self.linear = nn.Linear(idim, odim)

    def forward(self, x):
        """
        Args:
            x (_type_):(N, T, idim)

        Return:
            x Tensor: (N, T, odim)
        """
        x = x.transpose(1, 2)  # -> (N, idim, T)
        x = self.sub_sampling(x)
        x = x.transpose(1, 2)  # -> (N, T, idim)
        x = self.linear(x)
        return x


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4
    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, idim: int, odim: int) -> None:
        """
        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        assert idim >= 7
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.
        Args:
          x:
            Its shape is (N, T, idim).
        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        return x


class ConformerModel(nn.Module):
    def __init__(
        self,
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
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        double_swish=False,
        sub_sampling: int = 2,
    ) -> None:
        super().__init__()
        self.fbank = FBank(
            win_len=win_len,
            sr=sr,
            hop_length=hop_length,
            n_mels=n_mels,
            t_mask_prob=t_mask_prob,
            f_mask=f_mask,
            mask_times=mask_times,
        )
        logging.info(f"下采样倍数: {sub_sampling}")
        if sub_sampling == 4:
            self.sub_sampling = Conv2dSubsampling(n_mels, encoder_dim)
        else:
            self.sub_sampling = Conv1dSubSampling2(n_mels, encoder_dim)
        self.pos = RelPositionalEncoding(encoder_dim, dropout_rate=0.0)
        self.linear = nn.Linear(n_mels, encoder_dim)
        self.encoders = nn.ModuleList()
        for i in range(n_blocks):
            self.encoders.append(
                ConformerBlock(
                    dim=encoder_dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_dropout=conv_dropout,
                    double_swish=double_swish,
                )
            )

    def forward(self, x, pad_mask=None):
        """
        Args:
            x (_type_): raw audio (N, L), egs: (4, 16000)
        """
        lengths = torch.sum(pad_mask, dim=1)
        feats = []
        for i in range(x.size(0)):
            tmp = self.fbank(
                x[i, : x.size(1) - int(lengths[i].item())].unsqueeze(0)
            ).squeeze(0)
            feats.append(tmp)
        x = pad_sequence(feats, batch_first=True)
        # x = self.sub_sampling(x.transpose(1, 2)).transpose(1, 2)  # (N, T, C) -> (N, T', C)
        x = self.sub_sampling(x)
        x, _ = self.pos(x)
        # x = self.linear(x)
        for layer in self.encoders:
            x = layer(x)
        return x  # (N, T, C)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int, apply_dropout: bool = True
    ) -> torch.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size < self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(
                offset.device
            )  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


if __name__ == "__main__":
    x = torch.randn((4, 16000))
    x = x.to("cuda:0")
    model = ConformerModel().to("cuda:0")
    out = model(x, torch.zeros_like(x))
    print(out.shape)

# look good: torchaudio-mel subsample4
