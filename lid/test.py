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
sys.path.append("..")
sys.path.append("./code/")

from lid.lm_decoder import BeamSearchDecoderWithLM
from lid.LidModule_ASR import LidModule
import logging

KENLM_THRESHOLD = 0.012 * 1


class XFResult:
    def __init__(
        self,
        ckpt_path: str,
        base_pt_path: str, 
        folder: str,
        result_file: str,
        map_location: str = "cpu",
        lang_bind: bool = True,
        persian_lm_path: str = None,
        swahili_lm_path: str = None,
        vietnamese_lm_path: str = None,
        per_lm_decoder: BeamSearchDecoderWithLM = None,
        swa_lm_decoder: BeamSearchDecoderWithLM = None,
        vie_lm_decoder: BeamSearchDecoderWithLM = None,
    ) -> None:

        if not os.path.exists(folder):
            logging.error(f"{folder} 不存在")
            exit()
        self.folder = folder
        self.model = LidModule.resume_from_checkpoint(
            ckpt_path, map_location=map_location, pt_path=base_pt_path
        )
        self.model.model.eval()
        self.device = torch.device(map_location)
        self.result_file = result_file
        self.lang_bind = lang_bind
        self.langs = ["Persian", "Swahili", "Vietnamese"]
        self.per_lm = kenlm.Model(persian_lm_path)
        self.swa_lm = kenlm.Model(swahili_lm_path)
        self.vie_lm = kenlm.Model(vietnamese_lm_path)
        self.per_lm_decoder = per_lm_decoder
        self.swa_lm_decoder = swa_lm_decoder
        self.vie_lm_decoder = vie_lm_decoder

    @torch.no_grad()
    def predict_audio(self, audio_path: str, lang: str = None):
        wav, sr = torchaudio.load(audio_path)
        wav = self.normalize_wav(wav).to(self.device)
        pre_lang = lang
        if lang is None:
            out = self.model.infer(wav, sr)
            index = torch.argmax(out[1], dim=-1)
            prob = out[1].squeeze(0).detach().cpu().numpy().tolist()
            index = index[0].item()
            # 如果区分度不明显，使用语言模型区分
            if (
                prob[index] - prob[index - 1] < KENLM_THRESHOLD
                and prob[index] - prob[index - 1] > -KENLM_THRESHOLD
            ) or (
                prob[index] - prob[index - 2] < KENLM_THRESHOLD
                and prob[index] - prob[index - 2] > -KENLM_THRESHOLD
            ):
                pre_text, pre_lang = self.lm_select(
                    out[0]["Persian"][0], out[0]["Swahili"][0], out[0]["Vietnamese"][0]
                )
            else:
                pre_lang = self.langs[index]
                pre_text = out[0][pre_lang][0]
        else:
            out = self.model.infer(wav, sr, lang)
            pre_text = out[0][lang][0]

        # lm
        lm_predict_texts = []
        if pre_lang == "Persian" and self.per_lm_decoder is not None:
            lm_predict_texts = self.per_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
            return lm_predict_texts[0], pre_lang, out
        if pre_lang == "Swahili" and self.swa_lm_decoder is not None:
            lm_predict_texts = self.swa_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
            return lm_predict_texts[0], pre_lang, out
        if pre_lang == "Vietnamese" and self.vie_lm_decoder is not None:
            lm_predict_texts = self.vie_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
            return lm_predict_texts[0], pre_lang, out
        return pre_text, pre_lang, out

    def lm_select(self, persian_text, swahili_text, vietnamese_text):
        """使用语言模型判断内容语种归属

        Returns:
            _type_: 文本, 语种
        """
        per_score = self.per_lm.perplexity(persian_text)
        swa_score = self.swa_lm.perplexity(swahili_text)
        vie_score = self.vie_lm.perplexity(vietnamese_text)
        if per_score <= swa_score and per_score <= vie_score:
            return persian_text, "Persian"
        if swa_score <= per_score and swa_score <= vie_score:
            return swahili_text, "Swahili"
        if vie_score <= per_score and vie_score <= swa_score:
            return vietnamese_text, "Vietnamese"

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
                pre_text, pre_lang, out = self.predict_audio(
                    audio_path, lang if self.lang_bind else None
                )
                if pre_lang == lang:
                    corr += 1
                total += 1
                results.append((wav_name, pre_text))

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
        ground_trues = []
        pre_texts = []
        probs = []
        total = 0
        corr = 0

        dataset = self._get_dataset_xf(manifest_path)
        for data in tqdm(dataset, total=len(dataset)):
            pre_text, pre_lang, out = self.predict_audio(data["path"], lang=lang)
            ground_trues.append(data["sentence"])
            pre_texts.append(pre_text)
            if pre_lang == data["locale"]:
                corr += 1
            total += 1
            # probs.append(out[1].squeeze(0).detach().cpu().numpy().tolist()) 
        wer_fn = torchmetrics.CharErrorRate()
        wer_fn2 = torchmetrics.WordErrorRate()
        # self.write_to_csv(ground_trues, pre_texts, probs, lang=lang)
        print(f"语种识别精度: {corr/total}, {corr}/{total}")
        cer = wer_fn(ground_trues, pre_texts).item()
        wer = wer_fn2(ground_trues, pre_texts).item()
        print(f"cer: {cer}, wer: {wer}")
        return cer, ground_trues, pre_texts

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
    parse = argparse.ArgumentParser(description="WavLM muti language ASR for 讯飞比赛")
    parse.add_argument("--alpha", type=float, default=2)
    parse.add_argument("--beta", type=float, default=1)
    parse.add_argument("--cutoff_top_n", type=int, default=40)
    parse.add_argument("--beam_width", type=int, default=1000)
    parse.add_argument("--base_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/data/xf/")
    parse.add_argument("--test_path", type=str, default="/workspace/xfdata/data/")
    parse.add_argument("--pt_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/outputs/2022-09-11/16-14-limit_Wav2vec_Large_lr_0.0001_dr_0.1_bs_4_conform_False/ckpt/last.pt")
    parse.add_argument("--base_pt_path", type=str, default="wavlm/ckpts/WavLM-Base-plus.pt")
    arg = parse.parse_args()
    
    base_path = arg.base_path
    # final 0.9166
    ckpt_path = arg.pt_path
    
    per_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            base_path + "data/Persian-vocab.txt", "r"
        ).readlines()
    ]
    swa_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            base_path + "data/Swahili-vocab.txt", "r"
        ).readlines()
    ]
    vie_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            base_path + "data/Vietnamese-vocab.txt", "r"
        ).readlines()
    ]
    
    pe_lm_path = base_path + "lm/per_train9lm3.arpa"
    sw_lm_path = base_path + "lm/swa_train9lm3.arpa"
    vi_lm_path = base_path + "lm/vie_train9lm3.arpa"
    
    # lm 提升 wenet greedy12.18 ->(wav2vec lstm)0.1215 -> 0.1167(wavlm conformer)
    # -> 0.1066(train) -> (train+cv) 0.0994 -> 0.921 (train+cv+other)
    beam_width = arg.beam_width
    alpha = arg.alpha
    beta = arg.alpha
    cutoff_top_n = arg.cutoff_top_n
    per_lm_decoder = BeamSearchDecoderWithLM(
        vocab=per_vocab,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=pe_lm_path,
        num_cpus=12,
        cutoff_prob=1,
        cutoff_top_n=cutoff_top_n,
    )
    swa_lm_decoder = BeamSearchDecoderWithLM(
        vocab=swa_vocab,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=sw_lm_path,
        num_cpus=12,
        cutoff_prob=1,
        cutoff_top_n=25,
    )
    vie_lm_decoder = BeamSearchDecoderWithLM(
        vocab=vie_vocab,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=vi_lm_path,
        num_cpus=12,
        cutoff_prob=1,
        cutoff_top_n=70,
    )
    module = XFResult(
        ckpt_path=ckpt_path,
        base_pt_path=arg.base_pt_path,  # "wavlm/ckpts/WavLM-Large.pt"
        folder=arg.base_path + "data/",  # test集文件夹，解压后的文件夹，下面分别是 Persian, Swahili, Vietnamese三个文件夹
        result_file="/tmp/result.csv",
        map_location="cuda:0",
        persian_lm_path=pe_lm_path,
        swahili_lm_path=sw_lm_path,
        vietnamese_lm_path=vi_lm_path,
        lang_bind=False,
        # per_lm_decoder=per_lm_decoder,
        # swa_lm_decoder=swa_lm_decoder,
        # vie_lm_decoder=vie_lm_decoder,
    )
    all_ground_trues = []
    all_pre_texts = []
    cer, ground_trues, pre_texts = module.test_val(
        base_path + "data/Persian/dev1.label"#, lang="Persian"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        base_path + "data/Swahili/dev1.label"#, lang="Swahili"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        base_path + "data/Vietnamese/dev1.label"#, lang="Vietnamese"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer = torchmetrics.CharErrorRate()(all_ground_trues, all_pre_texts)
    wer = torchmetrics.WordErrorRate()(all_ground_trues, all_pre_texts)
    print(f"total cer: {cer}, total wer: {wer}")
    # common voice
    all_ground_trues.clear()
    all_pre_texts.clear()
    cer, ground_trues, pre_texts = module.test_val(
        "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Persian/cv_test.label"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Swahili/cv_test.label"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Vietnamese/cv_test.label"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer = torchmetrics.CharErrorRate()(all_ground_trues, all_pre_texts)
    wer = torchmetrics.WordErrorRate()(all_ground_trues, all_pre_texts)
    print(f"total cer: {cer}, total wer: {wer}")
    # 生成结果
    # module.process()
