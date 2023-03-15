import argparse
import csv
import os, sys
import time
from typing import List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torchmetrics
import torchaudio
import kenlm
import random

from eer import EER, EER2, CAvg

sys.path.append("..")
sys.path.append("./code/")
from lid.LidModule_Cross_Entropy import LidModuleCross
import logging

SNR = 0
ENHANCE_FACTOR=0.3

NOISE = "factory1"

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
        self.eer = EER2()
        self.cavg = CAvg(num_class=len(self.model.lang2index_dict.keys()))
        
        self.noises = {}

    @torch.no_grad()
    def predict_audio(self, audio_path: str):
        wav, sr = torchaudio.load(audio_path)
        if SNR < 50:  # 50以下的snr才有测试的必要
            wav = self.add_noise(wav, sr, SNR, NOISE)
        if ENHANCE_FACTOR != 1:
            enhance = self.enhance(wav)
            wav = (1 - ENHANCE_FACTOR) * enhance + ENHANCE_FACTOR * wav
        wav = self.normalize_wav(wav).to(self.device)
        pre_lang, pre_score, pre_index = self.model.infer(wav, sr)
        return pre_lang, pre_score

    def add_noise(self, clean, sr, snr:int = 0, noise_type:str = "white"):
        import random
        noise = self._get_noise(noise_type, sr)
        if noise.shape[1] < clean.shape[1] * 3:
            noise = noise.repeat((1, (int(len(clean) * 3 / len(noise)) - 1)))
        start = random.randint(0, noise.shape[0] - clean.shape[0])
        noise_b = noise[:, start:start+clean.shape[1]]
        sum_clean = torch.sum(clean ** 2)
        sum_noise = torch.sum(noise_b ** 2)

        x = torch.sqrt(sum_clean / (sum_noise * pow(10, snr / 10.0)))
        noise_c = x * noise_b
        noisy = clean + noise_c
        # torchaudio.save("/hdd/1/chenc/lid/speech-lid/lid/results/test.wav", noisy, sr)
        return noisy
    
    def _get_noise(self, noise_type, target_sr):
        if noise_type == "white":
            if noise_type in self.noises.keys():
                noise, noise_sr = self.noises[noise_type]
            else:
                noise, noise_sr = torchaudio.load("/hdd/1/chenc/lid/speech-lid/lid/noise/white.wav")
                noise = torchaudio.functional.resample(noise, noise_sr, target_sr)
                self.noises[noise_type] = noise, noise_sr
        if noise_type == "factory1":
            if noise_type in self.noises.keys():
                noise, noise_sr = self.noises[noise_type]
            else:
                noise, noise_sr = torchaudio.load("/hdd/1/chenc/lid/EHNet/dataset/noise92/factory1.wav")
                noise = torchaudio.functional.resample(noise, noise_sr, target_sr)
                self.noises[noise_type] = noise, noise_sr
        if noise_type == "factory2":
            if noise_type in self.noises.keys():
                noise, noise_sr = self.noises[noise_type]
            else:
                noise, noise_sr = torchaudio.load("/hdd/1/chenc/lid/EHNet/dataset/noise92/factory2.wav")
                noise = torchaudio.functional.resample(noise, noise_sr, target_sr)
                self.noises[noise_type] = noise, noise_sr
        if noise_type == "babble":
            if noise_type in self.noises.keys():
                noise, noise_sr = self.noises[noise_type]
            else:
                noise, noise_sr = torchaudio.load("/hdd/1/chenc/lid/EHNet/dataset/noise92/babble.wav")
                noise = torchaudio.functional.resample(noise, noise_sr, target_sr)
                self.noises[noise_type] = noise, noise_sr
        return noise
    
    def enhance(self, x:torch.Tensor):
        import soundfile as sf
        import io, requests
        
        to_send = io.BytesIO()
        sf.write(to_send, x.squeeze(0).numpy(), 16000, format="WAV")
        to_send.seek(0)
        res = requests.post("http://127.0.0.1:8080/se", 
              files={"file": to_send})
        clean_io = io.BytesIO(res.content)
        clean_io.seek(0)
        enhance, sr = torchaudio.load(clean_io)
        # torchaudio.save("/hdd/1/chenc/lid/speech-lid/lid/results/enhance.wav", enhance, sr)
        return enhance

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
        return torch.div(wav - mean, std + 1e-9)  # 1e-6

    def test_val(self, manifest_path: str, lang: str = None):
        total = 0
        corr = 0
        total_cost_time = 0
        dataset = self._get_dataset_xf(manifest_path)
        for data in tqdm(dataset, total=len(dataset)):
            pre_time = time.time()
            pre_lang, pre_score = self.predict_audio(data["path"])
            total_cost_time += (time.time() - pre_time)
            self.eer.update([pre_score], [self.model.getIndexByLangName(data["locale"])])
            self.cavg.update([pre_score], [self.model.getIndexByLangName(data["locale"])])
            if pre_lang == data["locale"]:
                corr += 1
            total += 1
        print(f"语种识别精度: {corr/total}, {corr}/{total}, avg_time: {total_cost_time/len(dataset)}")

        
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
        default="/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-06/01-35-lid_lr_0.001_adam_bs_8_model_resnet2_aug_speedTrue/ckpt/last.pt",
    )
    parse.add_argument(
        "--base_pt_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/wavlm/ckpts/WavLM-Base-plus.pt"
    )
    parse.add_argument("--snr", type=float, default=20)
    parse.add_argument("--factor", type=float, default=1)
    parse.add_argument("--noise", type=str, default="factory1")
    parse.add_argument("--test_range", type=str, default="xf",help="xf, all")
    arg = parse.parse_args()
    
    SNR = arg.snr
    ENHANCE_FACTOR = arg.factor
    NOISE = arg.noise  # white factor1 factor2 babble

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
    print("--------------------------------------")
    print(f"eer2: {module.eer.compute()}")
    print(f"cavg: {module.cavg.compute()}")
    print("--------------------------------------")
    module.eer.reset()
    module.cavg.reset()
    
    if arg.test_range == "all":
        module.test_val(
            "/data/chenc/lid/xfdata/Persian/cv_test.label"
        )
        module.test_val(
            "/data/chenc/lid/xfdata/Swahili/cv_test.label"
        )
        module.test_val(
            "/data/chenc/lid/xfdata/Vietnamese/cv_test.label"
        )
        print("--------------------------------------")
        print(f"eer2: {module.eer.compute()}")
        print(f"cavg: {module.cavg.compute()}")
        print("--------------------------------------")
        module.cavg.reset()
    # 生成结果
    # module.process()
