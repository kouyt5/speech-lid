import torchaudio
import torch
import random
import augment


def wav2mel(
    x,
    use_kaildi: bool = False,
    win_length: float = 0.025,
    hop_length: float = 0.01,
    n_mels: int = 80,
    n_fft: int = 512,
    pad: int = 0,
    sr: int = 16000,
):
    """Mel生成

    Args:
        x torch.Tensor: (1, T)

    Returns:
        x torch.Tensor: (1, n_mels, T)
    """
    if use_kaildi:
        return _kaidi_wav2mel(
            x, win_length=win_length, hop_length=hop_length, n_mels=n_mels, sr=sr
        )
    return _internal_wav2mel(
        x,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        n_fft=n_fft,
        pad=pad,
        sr=sr,
    )


def _kaidi_wav2mel(
    x,
    win_length: float = 0.025,
    hop_length: float = 0.01,
    n_mels: int = 80,
    sr: int = 16000,
    *args,
    **kwargs
):
    win_length = int(1000 * win_length)
    hop_length = int(1000 * hop_length)
    # https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/processor.py
    x = (
        torchaudio.compliance.kaldi.fbank(
            x, #  * (1 << 15),
            num_mel_bins=n_mels,
            dither=0.0,
            frame_length=win_length,
            frame_shift=hop_length,
            preemphasis_coefficient=1.0,
            sample_frequency=sr,
        )
        .transpose(0, 1)
        .unsqueeze(0)
    )  # (1, T) -> (T, n_mels) -> (n_mels, T) -> (1, n_mels, T)
    # do normal
    # std, mean = torch.std_mean(x, dim=1)
    # x = (x - mean) / (std + 1e-9)
    return x


def _internal_wav2mel(
    x,
    win_length: float = 0.025,
    hop_length: float = 0.01,
    n_mels: int = 80,
    n_fft: int = 512,
    pad: int = 0,
    sr: int = 16000,
):
    """torchaudio原生wav转mel谱

    Args:
        x torch.Tensor: (1, T)

    Return:
        x torch.Tensor: (1, n_mels, T)
    """
    win_length = int(sr * win_length)
    hop_length = int(sr * hop_length)
    x = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        pad=pad,
        n_mels=n_mels,
        center=True,
        pad_mode="reflect",
        power=2.0,
        onesided=True,
    )(
        x
    )  # -> (1, n_mels, T)
    x = torchaudio.transforms.AmplitudeToDB(top_db=80)(x)
    return x


def normalize_wav(wav: torch.Tensor):
    """对音频做归一化处理

    Args:
        wav (torch.Tensor): (1, T)
    """
    std, mean = torch.std_mean(wav, dim=-1)
    return torch.div(wav - mean, std + 1e-6)


def read_audio(audio_path: str, normalize: bool = True):
    wav, sr = torchaudio.load(audio_path)
    if normalize:
        wav = normalize_wav(wav)  # 归一化
    return wav, sr


def wav_augment(
    wav, sr, speed_shift: bool = False, pitch_shift: bool = False, reverb: bool = False
):
    # dither
    wav += 1e-5 * torch.rand_like(wav)
    # preemyhasis
    wav = torch.cat(
        (wav[:, 0].unsqueeze(1), wav[:, 1:] - 0.97 * wav[:, :-1]),
        dim=1,
    )
    speed_shift_value = 1.0
    pitch_shift_value = 0
    if speed_shift:
        # speed preturb [0.9, 1, 1.1]
        speed_shift_value = random.choice([0.9, 1.0, 1.1])
    if pitch_shift:
        pitch_shift_value = random.choice([-80, -60, -40, -20, 0, 0, 20, 40, 60, 80])
        # pitch_shift_value = random.choice(
        #     [-240, -200, -160, -120, -80, -40, 0, 40, 80, 120, 160, 200, 240]
        # )
    if speed_shift or pitch_shift:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav,
            sr,
            [
                ["speed", str(speed_shift_value)],
                ["pitch", str(pitch_shift_value)],
                ["rate", str(sr)],
            ],
        )  # 调速
    if reverb:
        # 混响
        room_size = random.randint(0, 100)
        wav = (
            augment.EffectChain()
            .reverb(50, 50, room_size)
            .channels(1)
            .apply(wav, src_info={"rate": sr})
        )
    return wav, sr


def spectrogram_augment(
    spec,
    sr: int = 16000,
    n_mels: int = 80,
    hop_length: float = 0.01,
    t_mask: float = 0.05,
    f_mask: float = 27,
    mask_times: int = 0,
    t_stretch: bool = False,
):
    """谱增强

    Args:
        spec torch.Tensor: 频谱 (1, n_mels, T)
        sr int: 采样率
        n_mels int: mel特征数
        hop_length float: hop length
        t_mask (float, optional): 时域mask比例. Defaults to 0.05.
        f_mask (float, optional): 频域mask点数. Defaults to 27.
        mask_times (int, optional): mask次数. Defaults to 0.
        t_stretch (bool, optional): 时间轴的延伸. Defaults to False.
    """
    if t_stretch:
        hop_length = int(sr * hop_length)
        spec = torchaudio.transforms.TimeStretch(hop_length=hop_length, n_freq=n_mels)(
            spec, random.choice([0.9, 1.0, 1.1])
        ).abs()
    for i in range(mask_times):
        spec = torchaudio.transforms.TimeMasking(int(spec.size(2) * t_mask))(spec)
        spec = torchaudio.transforms.FrequencyMasking(f_mask)(spec)
    return spec

if __name__ == "__main__":
    x = torch.randn((1, 16000), dtype=torch.float32)
    
    wav, sr = wav_augment(x, 1600, True, True, True)
    spec = wav2mel(wav, use_kaildi=False, sr=16000)
    spec = spectrogram_augment(spec, sr=16000, mask_times=2, t_stretch=True, t_mask=0.9)
    print(spec.shape)