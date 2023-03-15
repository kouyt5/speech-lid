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
sys.path.append("..")
sys.path.append("./code/")

from lid.eer import EER2, CAvg
from lid.lm_decoder import BeamSearchDecoderWithLM
from lid.LidModule_ASR_Supervised import LidSuperviseModule
import logging

KENLM_THRESHOLD = 0.012 * 0.2

SNR = 0
ENHANCE_FACTOR=0.3

NOISE = "factor1"

class XFResult:
    def __init__(
        self,
        ckpt_path: str,
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
        use_lm: bool = False,
    ) -> None:

        if not os.path.exists(folder):
            logging.error(f"{folder} 不存在")
            exit()
        self.folder = folder
        self.use_lm = use_lm
        self.model = LidSuperviseModule.resume_from_checkpoint(
            ckpt_path, map_location=map_location
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
        self.eer = EER2()
        self.cavg = CAvg(num_class=len(self.model.lang2index_dict.keys()))
        self.noises = {}
        self.TOTAL_TIME = 0
        self.total_audio_len = 0

    def _need_lm(self, prob:List[float], index:int):
        result = False
        i = 1
        while (i<len(prob)):
            curr_ok = prob[index] - prob[index - i] < KENLM_THRESHOLD \
                and prob[index] - prob[index - i] > -KENLM_THRESHOLD
            result = curr_ok or result
            i += 1
        return result
    
    def __lm_select(self, pre_lang, out):
        lm_predict_texts = []
        if pre_lang == "Persian" and self.per_lm_decoder is not None:
            lm_predict_texts = self.per_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
        if pre_lang == "Swahili" and self.swa_lm_decoder is not None:
            lm_predict_texts = self.swa_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
        if pre_lang == "Vietnamese" and self.vie_lm_decoder is not None:
            lm_predict_texts = self.vie_lm_decoder.forward(
                torch.softmax(out[2][pre_lang], dim=-1).detach().cpu().numpy(),
                [out[2][pre_lang][0].size(0)],
            )
        return lm_predict_texts[0], pre_lang
    
    @torch.no_grad()
    def predict_audio(self, audio_path: str, lang: str = None):
        pre_lang = lang
        final_prob = []
        origin_prob = []
        if lang is None:
            wav, sr = torchaudio.load(audio_path)
            self.total_audio_len += (wav.shape[1] / sr)
            temp_audio = audio_path
            if SNR < 50:  # 50以下的snr才有测试的必要
                wav = self.add_noise(wav, sr, SNR, NOISE)
            if ENHANCE_FACTOR != 1:
                enhance = self.enhance(wav)
                wav = (1 - ENHANCE_FACTOR) * enhance + ENHANCE_FACTOR * wav
            # temp_audio = "/tmp/conformer_test_temp2.wav"
            # torchaudio.save(temp_audio, wav, sr)
            pre_time = time.time()
            # out = self.model.infer(temp_audio, device=self.device)
            out = self.model.infer_tensor(wav, sr, device=self.device)
            self.TOTAL_TIME += (time.time() - pre_time)
            index = torch.argmax(out[1], dim=-1)
            prob = out[1].squeeze(0).detach().cpu().numpy().tolist()
            origin_prob = prob
            index = index[0].item()
            # 如果区分度不明显，使用语言模型区分
            if self._need_lm(prob, index):
                pre_text, pre_lang, lm_prob = self.lm_select(
                    out[0]["Persian"][0], out[0]["Swahili"][0], out[0]["Vietnamese"][0]
                )
                final_prob = lm_prob
            else:
                pre_lang = self.langs[index]
                pre_text = out[0][pre_lang][0]
                prob = [(-1/(item-1e-9)) for item in prob]
                prob = [item/sum(prob) for item in prob]
                final_prob = prob
        else:
            out = self.model.infer(wav, sr, lang)
            pre_text = out[0][lang][0]

        # 语言模型解码ASR
        if self.use_lm:
            lm_predict_texts, pre_lang = self.__lm_select(pre_lang, out)
            return lm_predict_texts, pre_lang, final_prob, origin_prob
        
        return pre_text, pre_lang, final_prob, origin_prob
    
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
    
    def lm_select(self, persian_text, swahili_text, vietnamese_text):
        """使用语言模型判断内容语种归属

        Returns:
            _type_: 文本, 语种
        """
        per_score = self.per_lm.perplexity(persian_text)
        swa_score = self.swa_lm.perplexity(swahili_text)
        vie_score = self.vie_lm.perplexity(vietnamese_text)
        total_score = 1/per_score + 1/swa_score + 1/vie_score
        prob_per = (1/per_score) / total_score
        prob_swa = (1/swa_score) / total_score
        prob_vie = (1/vie_score) / total_score
        if per_score <= swa_score and per_score <= vie_score:
            return persian_text, "Persian", [prob_per, prob_swa, prob_vie]
        if swa_score <= per_score and swa_score <= vie_score:
            return swahili_text, "Swahili", [prob_per, prob_swa, prob_vie]
        if vie_score <= per_score and vie_score <= swa_score:
            return vietnamese_text, "Vietnamese", [prob_per, prob_swa, prob_vie]

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
                pre_text, pre_lang, out, _ = self.predict_audio(
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
        origin_probs = []
        total = 0
        corr = 0
        total_cost_time = 0
        dataset = self._get_dataset_xf(manifest_path)
        for data in tqdm(dataset, total=len(dataset)):
            pre_time = time.time()
            pre_text, pre_lang, prob, origin_prob = self.predict_audio(data["path"], lang=None)
            total_cost_time += (time.time() - pre_time)
            ground_trues.append(data["sentence"])
            pre_texts.append(pre_text)
            if pre_lang == data["locale"]:
                corr += 1
            total += 1
            # prob = torch.softmax(torch.FloatTensor(prob), dim = -1).tolist()
            # print(prob)
            # prob = [-1/item for item in prob]
            # prob = [item/sum(prob) for item in prob]
            # print(prob)
            self.eer.update([prob], [self.model.lang2index_dict[data["locale"]]])
            self.cavg.update([prob], [self.model.lang2index_dict[data["locale"]]])
            probs.append(prob)
            origin_probs.append(origin_prob)
        wer_fn = torchmetrics.CharErrorRate()
        wer_fn2 = torchmetrics.WordErrorRate()
        self.write_to_csv(ground_trues, pre_texts, origin_probs, lang=lang)
        print(f"语种识别精度: {corr/total}, {corr}/{total}, avg time: {total_cost_time/len(dataset)}, avg act:{self.TOTAL_TIME/len(dataset)}")
        print(f"平均音频长度 {self.total_audio_len/len(dataset)}")
        self.TOTAL_TIME = 0
        self.total_audio_len = 0
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
    parse.add_argument("--pt_path", type=str, default="/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-12/00-18-lr_0.01_dr_0.1_bs_16_conform_True/ckpt/last.pt")
    parse.add_argument("--snr", type=float, default=100)
    parse.add_argument("--factor", type=float, default=1)
    parse.add_argument("--kenlm_factor", type=int, default=0.012 * 0.2)
    parse.add_argument("--noise", type=str, default="factory1")
    parse.add_argument("--test_range", type=str, default="xf",help="xf, all")
    arg = parse.parse_args()
    
    SNR = arg.snr
    ENHANCE_FACTOR = arg.factor
    KENLM_THRESHOLD = arg.kenlm_factor
    NOISE = arg.noise  # white factor1 factor2 babble
    
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
    
    pe_lm_path = "/hdd/1/chenc/lid/speech-lid/lid/data/xf/lm/github/all/v5" + "/outv5pe3gram.arpa"
    sw_lm_path = "/hdd/1/chenc/lid/speech-lid/lid/data/xf/lm/github/all/v3/outv3sw3gram.arpa"
    vi_lm_path = "/hdd/1/chenc/lid/speech-lid/lid/data/xf/lm/github/all/v5/outv5vi3gram.arpa"
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
        folder=arg.base_path + "data/",  # test集文件夹，解压后的文件夹，下面分别是 Persian, Swahili, Vietnamese三个文件夹
        result_file="/hdd/1/chenc/lid/speech-lid/result.csv",
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
        base_path + "data/Persian/dev1.label", lang="Persian"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        base_path + "data/Swahili/dev1.label", lang="Swahili"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer, ground_trues, pre_texts = module.test_val(
        base_path + "data/Vietnamese/dev1.label", lang="Vietnamese"
    )
    all_ground_trues.extend(ground_trues)
    all_pre_texts.extend(pre_texts)
    cer = torchmetrics.CharErrorRate()(all_ground_trues, all_pre_texts)
    wer = torchmetrics.WordErrorRate()(all_ground_trues, all_pre_texts)
    print(f"total cer: {cer}, total wer: {wer}")
    print("--------------------------------------")
    print(f"eer: {module.eer.compute()}")
    print(f"cavg: {module.cavg.compute()}")
    print("--------------------------------------\n")
    module.eer.reset()
    module.cavg.reset()
    if arg.test_range == "all":
        # common voice
        all_ground_trues.clear()
        all_pre_texts.clear()
        cer, ground_trues, pre_texts = module.test_val(
            "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Persian/cv_test.label", "Persian"
        )
        all_ground_trues.extend(ground_trues)
        all_pre_texts.extend(pre_texts)
        cer, ground_trues, pre_texts = module.test_val(
            "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Swahili/cv_test.label", "Swahili"
        )
        all_ground_trues.extend(ground_trues)
        all_pre_texts.extend(pre_texts)
        cer, ground_trues, pre_texts = module.test_val(
            "/hdd/1/chenc/lid/speech-lid/lid/data/xf/data/Vietnamese/cv_test.label", "Vietnamese"
        )
        all_ground_trues.extend(ground_trues)
        all_pre_texts.extend(pre_texts)
        cer = torchmetrics.CharErrorRate()(all_ground_trues, all_pre_texts)
        wer = torchmetrics.WordErrorRate()(all_ground_trues, all_pre_texts)
        print(f"total cer: {cer}, total wer: {wer}")
        print("--------------------------------------")
        print(f"eer: {module.eer.compute()}")
        print(f"cavg: {module.cavg.compute()}")
        print("--------------------------------------")
        module.eer.reset()
        module.cavg.reset()
    # 生成结果
    # module.process()
