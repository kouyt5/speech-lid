import argparse
import csv
import os, sys
from typing import List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torchmetrics
import torchaudio
import kenlm

from eer import EER2, EER

sys.path.append("..")
sys.path.append("./code/")
from lid.LidModule_Cross_Entropy import LidModuleCross
import logging


class MyResult:
    def __init__(
        self,
        ckpt_path: str,
        base_pt_path: str,
        folder: str,
        result_file: str,
        map_location: str = "cpu",
    ) -> None:

        self.folder = folder
        self.model = LidModuleCross.resume_from_checkpoint(
            ckpt_path, map_location=map_location, pt_path=base_pt_path
        )
        self.model.model.eval()
        self.device = torch.device(map_location)
        self.result_file = result_file
        self.langs = ["Persian", "Swahili", "Vietnamese"]
        self.scores = []
        self.labels = []

    @torch.no_grad()
    def predict_audio(self, audio_path: str):
        wav, sr = torchaudio.load(audio_path)
        wav = self.normalize_wav(wav).to(self.device)
        pre_lang, pre_score, pre_index = self.model.infer(wav, sr)
        self.scores.append(pre_score)
        return pre_lang

    def process(self, lang: str = None):
        results = []
        total = 0
        corr = 0
        langs = self.langs
        if lang is not None:
            langs = [lang]
        for lang in langs:
            wav_folder = os.path.join(self.folder, lang, "wav", "test")
            wavs = os.listdir(wav_folder)
            for wav_name in tqdm(wavs, total=len(wavs)):
                audio_path = os.path.join(wav_folder, wav_name)
                pre_lang = self.predict_audio(audio_path)
                if pre_lang == lang:
                    corr += 1
                total += 1
                results.append((wav_name, pre_lang))

        self.write_to_file(results)
        print("done!!!")
        print(f"正确率: {corr/total}, {corr}/{total}")

    def write_to_file(self, datas: List = None):
        writer = csv.DictWriter(
            open(self.result_file, "w"), fieldnames=["wav_name", "text"], delimiter="\t"
        )
        writer.writeheader()
        for data in datas:
            writer.writerow({"wav_name": data[0], "text": data[1]})

    def write_to_csv(
        self,
        trues: List = None,
        preds: List = None,
        probs: List = None,
        lang: str = "none",
    ):
        """写入本地验证集结果

        Args:
            [(true_text, pre_text, "Persian", "Swahili", "Vietnamese")].
        """
        result_file = "/".join(self.result_file.split("/")[:-1]) + "/" + lang + ".csv"
        writer = csv.DictWriter(
            open(result_file, "w"),
            fieldnames=["true", "pred", "Persian", "Swahili", "Vietnamese"],
            delimiter="\t",
        )
        writer.writeheader()
        for ground_true, pred, prob in zip(trues, preds, probs):
            writer.writerow(
                {
                    "true": ground_true,
                    "pred": pred,
                    "Persian": prob[0],
                    "Swahili": prob[1],
                    "Vietnamese": prob[2],
                }
            )
        pass

    def normalize_wav(self, wav: torch.Tensor):
        """对音频做归一化处理

        Args:
            wav (torch.Tensor): (1, T)
        """
        std, mean = torch.std_mean(wav, dim=-1)
        return torch.div(wav - mean, std + 1e-6)

    def test_val(self, manifest_path: str, lang: str = None):
        total = 0
        corr = 0

        dataset = self._get_dataset_xf(manifest_path)
        for data in tqdm(dataset, total=len(dataset)):
            pre_lang = self.predict_audio(data["path"])
            if pre_lang == data["locale"]:
                corr += 1
            total += 1
            self.labels.append(self.model.getIndexByLangName(data["locale"]))
        print(f"语种识别精度: {corr/total}, {corr}/{total}")

    def clear(self):
        self.labels.clear()
        self.scores.clear()
        
    def _get_dataset_xf(self, manifest_path: str = None):
        datasets = []
        with open(manifest_path, "r") as f:
            lang = manifest_path.split("/")[-2]
            base_path = "/".join(manifest_path.split("/")[:-1])
            base_path = os.path.join(base_path, "wav", "train")
            for line in f.readlines():
                data = {}
                name = line.split("\t")[0]
                text = line.split("\t")[1].strip()

                data["path"] = os.path.join(base_path, name)
                audio_info = torchaudio.info(data["path"])
                duration = audio_info.num_frames / audio_info.sample_rate
                data["duration"] = duration
                data["locale"] = lang
                data["sentence"] = text
                datasets.append(data)
        return datasets


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="WavLM 交叉熵语种判别")

    parse.add_argument(
        "--base_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/data/xf/"
    )
    parse.add_argument("--test_path", type=str, default="/workspace/xfdata/data/")
    parse.add_argument(
        "--pt_path",
        type=str,
        default="/hdd/1/chenc/lid/speech-lid/lid/outputs/2022-12-13/22-14-lid_lr_0.001_dr_0.1_bs_8_model_xvector/ckpt/last.pt",
    )
    parse.add_argument(
        "--base_pt_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/wavlm/ckpts/WavLM-Base-plus.pt"
    )
    arg = parse.parse_args()

    base_path = arg.base_path
    # final 0.9166
    ckpt_path = arg.pt_path

    module = MyResult(
        ckpt_path=ckpt_path,
        base_pt_path=arg.base_pt_path,  # "wavlm/ckpts/WavLM-Large.pt"
        folder=arg.base_path
        + "data/",  # test集文件夹，解压后的文件夹，下面分别是 Persian, Swahili, Vietnamese三个文件夹
        result_file="/tmp/result.csv",
        map_location="cuda:0",
    )
    module.test_val("/data/chenc/lid/xfdata/Persian/dev1.label")
    module.test_val("/data/chenc/lid/xfdata/Swahili/dev1.label")
    module.test_val(
        "/data/chenc/lid/xfdata/Vietnamese/dev1.label"
    )


    eer_score = EER(num_class=len(module.scores[0]))(
            module.scores, module.labels
        )
    print(f"eer: {eer_score}")
    module.clear()
    
    module.test_val(
        "/data/chenc/lid/xfdata/Persian/cv_test.label"
    )
    module.test_val(
        "/data/chenc/lid/xfdata/Swahili/cv_test.label"
    )
    module.test_val(
        "/data/chenc/lid/xfdata/Vietnamese/cv_test.label"
    )
    eer_score = EER(num_class=len(module.scores[0]))(
            module.scores, module.labels
        )
    print(f"eer: {eer_score}")
    module.clear()
    # 生成结果
    # module.process()
