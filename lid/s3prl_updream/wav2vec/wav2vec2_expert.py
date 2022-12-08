# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2/expert.py ]
#   Synopsis     [ the wav2vec2 wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

# import os, sys

# os.chdir("/home/cc/workdir/code/wav2vec-exp")
# sys.path.append("/home/cc/workdir/code/wav2vec-exp/")
import logging
import time
from typing import Any, Dict, Optional
from packaging import version

import torch
import fairseq
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from s3prl_updream.interfaces import UpstreamBase
from s3prl_updream.wav2vec.wav2vec2 import Wav2Vec2Model
from s3prl_updream.helper import zero_mean_unit_var_norm
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.checkpoint_utils import (
    get_maybe_sharded_checkpoint_filename,
    load_checkpoint_to_cpu,
)
from fairseq.file_io import PathManager


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, drop_layer, mask=False, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        self.model, cfg, task = load_wav2vec2_for_finetune(
            ckpt_path=ckpt, drop_layer=drop_layer
        )
        # self.model = model[0]
        self.wav_normalize = cfg.task.normalize

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False
        self.mask = mask

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        self.model.extract_features(
            padded_wav,
            wav_padding_mask if self.apply_padding_mask else None,
            mask=self.mask if self.model.training else False,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks


def load_model_for_finetune(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    assert state is None or len(filenames) == 1

    from fairseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None
    for filename in filenames:
        orig_filename = filename
        model_shard_state = {"shard_weights": [], "shard_metadata": []}
        assert num_shards > 0
        st = time.time()
        for shard_idx in range(num_shards):
            filename = get_maybe_sharded_checkpoint_filename(
                orig_filename, suffix, shard_idx, num_shards
            )

            if not PathManager.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"Neither args nor cfg exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if "fsdp_metadata" in state and num_shards > 1:
                model_shard_state["shard_weights"].append(state["model"])
                model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
                # check FSDP import before the code goes too far
                if not has_FSDP:
                    raise ImportError(
                        "Cannot find FullyShardedDataParallel. "
                        "Please install fairscale with: pip install fairscale"
                    )
                if shard_idx == num_shards - 1:
                    consolidated_model_state = FSDP.consolidate_shard_weights(
                        shard_weights=model_shard_state["shard_weights"],
                        shard_metadata=model_shard_state["shard_metadata"],
                    )
                    model = task.build_model(cfg.model)
                    if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                    ):
                        model.set_num_updates(
                            state["optimizer_history"][-1]["num_updates"]
                        )
                    model.load_state_dict(
                        consolidated_model_state, strict=strict, model_cfg=cfg.model
                    )
            else:
                # turn off the layer dropout for finetune
                cfg.model["encoder_layerdrop"] = -1
                # model parallel checkpoint or unsharded checkpoint
                model = task.build_model(cfg.model)
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
                model.load_state_dict(
                    state["model"], strict=strict, model_cfg=cfg.model
                )

            # reset state so it gets loaded for the next model in ensemble
            state = None
            if shard_idx % 10 == 0 and shard_idx > 0:
                elapsed = time.time() - st
                logging.info(
                    f"Loaded {shard_idx} shards in {elapsed:.2f}s, {elapsed / (shard_idx+1):.2f}s/shard"
                )

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg, task


def load_wav2vec2_for_finetune(ckpt_path: str, drop_layer: bool = False):

    state = load_checkpoint_to_cpu(ckpt_path, None)
    if state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    else:
        cfg = state["cfg"]

    # turn off the layer dropout for finetune
    if not drop_layer:
        logging.info("turn off the layer dropout")
        cfg.model["encoder_layerdrop"] = -1
    else:
        logging.info("turn on the layer dropout")
    cfg.model["mask_channel_length"] = 64
    cfg.model["mask_channel_prob"] = 0.2
    cfg.model["mask_prob"] = 0.2
    # cfg.model["quantize_targets"] = False
    model = Wav2Vec2Model.build_model(cfg.model)
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    return model, cfg, None


if __name__ == "__main__":
    load_wav2vec2_for_finetune(
        "/home/cc/workdir/code/wav2vec-exp/ckpts/wav2vec_small.pt"
    )
