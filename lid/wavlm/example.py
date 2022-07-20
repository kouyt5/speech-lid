import logging
import torch
from lid.wavlm.WavLM import WavLM, WavLMConfig


class WavLMModel(torch.nn.Module):
    def __init__(
        self,
        path: str,
        use_pre_train: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
    ):
        super().__init__()
        checkpoint = torch.load(path)
        cfg = WavLMConfig(checkpoint["cfg"])
        cfg.mask_channel_prob = mask_channel_prob
        cfg.mask_prob = mask_prob
        self.model = WavLM(cfg)
        if use_pre_train:
            self.model.load_state_dict(checkpoint["model"])
            logging.info("使用预训练模型: " + path)
        else:
            logging.info("不使用预训练模型")

    def forward(
        self,
        x: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        only_last: bool = True,
    ):
        """

        Args:
            x (torch.Tensor): 16k 音频 (N, 16000)
            only_last (bool, optional): 只取最后一层. Defaults to True.
            padding_mask: padding部分的mask, 值为1 (N, 16000)

        Returns:
            torch.Tensor or list[torch.Tensor]: (N, 31, 768)
        """
        # mask: 是否启用mask, 只在训练阶段使用,包括padding_mask和channel mask的应用
        mask = self.training
        if only_last:
            return self.model.extract_features(x, padding_mask, mask)[0]
        _, layer_results = self.model.extract_features(
            x,
            padding_mask=padding_mask,
            mask=mask,
            output_layer=self.model.cfg.encoder_layers,
            ret_layer_results=True,
        )[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]  # [(N, 31, 768),...]
        return layer_reps


if __name__ == "__main__":
    path = "/home/cc/workdir/code/lid/wavlm/ckpts/WavLM-Base-plus.pt"
    model = WavLMModel(path=path)
    x = torch.randn((2, 32000))
    out = model(x, False)
    print(out.shape)
