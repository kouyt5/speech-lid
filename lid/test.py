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
import wandb
sys.path.append("..")

from ccml.loggers.wandb_logger import WandbLogger
from lid.lm_decoder import BeamSearchDecoderWithLM
from lid.LidModule_ASR import LidModule
import logging

KENLM_THRESHOLD = 0.012


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
    ) -> None:

        if not os.path.exists(folder):
            logging.error(f"{folder} 不存在")
            exit()
        self.folder = folder
        self.model = LidModule.resume_from_checkpoint(
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

    def process(self):
        results = []
        total = 0
        corr = 0
        langs = self.langs
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
            pre_text, pre_lang, out = self.predict_audio(data["path"])
            ground_trues.append(data["sentence"])
            pre_texts.append(pre_text)
            if pre_lang == data["locale"]:
                corr += 1
            total += 1
            probs.append(out[1].squeeze(0).detach().cpu().numpy().tolist())
        wer_fn = torchmetrics.CharErrorRate()
        self.write_to_csv(ground_trues, pre_texts, probs, lang=lang)
        print(f"语种识别精度: {corr/total}, {corr}/{total}")
        cer = wer_fn(ground_trues, pre_texts).item()
        print(f"cer: {cer}")
        return cer

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
    # 0.58 acc 0.9966
    # ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-10/04-50-lr_0.0001_dr_0.1_opti_adam_bs_4_conform_True/ckpt/last.pt"
    # 0.52 common-voice per(train)+vi(train+test) acc 0.99708
    ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-11/01-44-lr_0.0001_dr_0.1_opti_adam_bs_4_conform_True/ckpt/last.pt"
    # 0.487 cv per(train)+swa(train)+vi(train+test)  0.0895 0.0579 0.0239 acc 0.99757 2gram
    # 3gram asr 0.891 0.058 0.02367 acc 0.9957 1.0 1.0 0.9966
    # 2gram asr 0.882 0.0605 0.2306 acc 0.997 0.995 1.0 0.9975
    # 2gram alpha2 beta1 asr 0.0837 0.058 0.237 acc 0.997 0.998 1.0 
    # 3gram alpha2 beta1 asr 0.0858 0.0575 0.227 acc 0.998 0.998 
    # 3gram a2 b1 400 asr 0.0845 0.0552 0.0236 acc 0.998 0.998 1.0
    # 3gram a2 b1 beam 1000 0.0828 0.0554 0.0228 
    ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-11/13-17-lr_0.0001_dr_0.1_opti_adam_bs_4_conform_True/ckpt/last.pt"
    # ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-10/18-43-lr_0.0001_dr_0.1_opti_adam_bs_4_conform_True/ckpt/last.pt"

    # 3gram a2 b1 1000 0.0714 0.0507 0.0188 acc 0.998 0.9968 1.0
    # 3gram a2 b1 1k other_txt1 0.0736 0.0455 0.0157
    ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-13/11-55-lr_0.0001_dr_0.1_opti_adam_bs_4_conform_True/ckpt/last.pt"

    # None 0.066 0.0328 0.024
    # (3gram-o a2 b1 1k) None (3gram-o a2 b1 1k) 0.057 0.0327 0.012
    # 3gram-o2(cv) 0.05126 0.0325 0.0117
    ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-16/16-14-lr_0.0001_dr_0.1_mask_0.1_cmask_0.1_bs_4_conform_True/ckpt/last.pt"
    
    # 3gram-o2(cv) 0.0539 0.03208 0.0124
    ckpt_path = "/home/cc/workdir/code/lid/outputs/2022-07-17/15-30-lr_0.0001_dr_0.1_mask_0.1_cmask_0.1_bs_4_conform_True/ckpt/last.pt"
    
    per_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            "/home/cc/workdir/code/lid/data/xf/data/Persian-vocab.txt", "r"
        ).readlines()
    ]
    swa_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            "/home/cc/workdir/code/lid/data/xf/data/Swahili-vocab.txt", "r"
        ).readlines()
    ]
    vie_vocab = [
        c.strip() if not c.replace("\n", "") == " " else c.replace("\n", "")
        for c in open(
            "/home/cc/workdir/code/lid/data/xf/data/Vietnamese-vocab.txt", "r"
        ).readlines()
    ]

    parse = argparse.ArgumentParser(description="WavLM muti language ASR for 讯飞比赛")
    parse.add_argument(
        "--pe_lm_path",
        type=str,
        default="/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1pe3gram.arpa"
    )
    parse.add_argument(
        "--sw_lm_path",
        type=str,
        default="/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1sw3gram.arpa"
    )
    parse.add_argument(
        "--vi_lm_path",
        type=str,
        default="/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1vi3gram.arpa"
    )
    parse.add_argument("--alpha", type=float, default=2)
    parse.add_argument("--beta", type=float, default=1)
    parse.add_argument("--cutoff_top_n", type=int, default=40)
    parse.add_argument("--beam_width", type=int, default=1000)
    arg = parse.parse_args()
    
    logging.info("使用wandb 进行搜索")
    exp_name = f"a{arg.alpha}_b{arg.beta}_cut{arg.cutoff_top_n}_beam{arg.beam_width}"
    wandb_logger = WandbLogger(
        project="xf_lm",
        entity="kouyt5",
        name=exp_name,
        wandb_id=None,
        config=arg,
    )
    arg = wandb.config
    print("使用参数: "+str(arg))
    vi_lm_path = arg.vi_lm_path
    sw_lm_path = arg.sw_lm_path
    pe_lm_path = arg.pe_lm_path
    # vi_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/vie3gram.arpa"
    # vi_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/old/outvi3gram.arpa"
    # sw_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/swa3gram.arpa"
    # sw_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/old/outsw3gram.arpa"
    # pe_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/per3gram.arpa"
    # pe_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/old/outpe3gram.arpa"
    
    pe_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1pe3gram.arpa"
    sw_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1sw2gram.arpa"
    vi_lm_path = "/home/cc/workdir/code/lid/data/xf/lm/github/all/v1/outv1vi3gram.arpa"
    
    beam_width = arg.beam_width
    alpha = arg.alpha
    beta = arg.alpha
    cutoff_top_n=arg.cutoff_top_n
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
        beam_width=1000,
        alpha=2,
        beta=1,
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
        cutoff_top_n=60,
    )
    module = XFResult(
        ckpt_path=ckpt_path,
        folder="/home/cc/workdir/code/lid/data/xf/data",
        result_file="/home/cc/workdir/code/lid/data/xf/result/result.txt",
        map_location="cpu",
        persian_lm_path=pe_lm_path,
        swahili_lm_path=sw_lm_path,
        vietnamese_lm_path=vi_lm_path,
        lang_bind=True,
        per_lm_decoder=per_lm_decoder,
        # swa_lm_decoder=swa_lm_decoder,
        vie_lm_decoder=vie_lm_decoder,
    )

    cer = module.test_val("/home/cc/workdir/code/lid/data/xf/data/Persian/dev1.label", "Persian")
    cer = module.test_val("/home/cc/workdir/code/lid/data/xf/data/Swahili/dev1.label", "Swahili")
    cer = module.test_val(
        "/home/cc/workdir/code/lid/data/xf/data/Vietnamese/dev1.label", "Vietnamese"
    )
    wandb_logger.log(data={"test_cer": cer})

    # 生成结果
    module.process()


    # TODO
    # 2gram lm