from collections import defaultdict
from functools import partial
import logging
import math
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

class CTCTokenizer:
    """CTC Tokenizer: convert number to string with CTC beam search"""

    def __init__(self, vocab: Union[str, list]) -> None:
        if isinstance(vocab, str):
            with open(vocab) as f:
                all_lines = [s.replace("\n", "") for s in f.readlines()]
                self.labels_map = dict((i, all_lines[i]) for i in range(len(all_lines)))
        elif isinstance(vocab, list):
            self.labels_map = dict((i, vocab[i]) for i in range(len(vocab)))
        else:
            raise Exception("vocab is neither str or list, please check")
        # labels_map: {0, '_'}
        self.s2labels_map = dict(
            (self.labels_map[key], key) for key in self.labels_map.keys()
        )
        self.blank_id = len(self.labels_map)
        
        
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def ctc_decode(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None
    ) -> List[str]:
        """
        Decodes a sequence of labels to words
        Args:
            predictions: A torch.Tensor of shape [Batch, Time] of integer indices that correspond
                to the index of some character in the label set.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        hypotheses = []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[0]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = "".join([self.labels_map[c] for c in decoded_prediction])

            hypotheses.append(hypothesis)
        return hypotheses

    def decoder(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> List[str]:
        """
        解码非CTC预测值，用于对label解码

        Args:
            targets (torch.Tensor): 真实值label向量
            target_lengths (torch.Tensor): 长度

        Returns:
            List[str]: [description]
        """
        references = []
        with torch.no_grad():
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()
            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = ""
                for c in target:
                    if c in self.labels_map.keys():
                        reference += self.labels_map[c]
                    else:
                        reference += "_"  # ctc blank
                # reference = "".join([self.labels_map[c] for c in target])
                references.append(reference)
        return references

    def _ctc_prefix_beam_search(
        self,
        predictions: torch.Tensor,
        beam_size: int,
    ) -> List[Tuple[str, float]]:
        """CTC prefix beam search inner implementation
        https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py
        prefix 搜索 ctc 路径上所有满足当前候选解码字符串，并使用他们的概率和作为搜索的权重值，
        和 greedy 的区别在于 greedy 直接使用最大概率作为输出。而 prefix 的方式更加符合 CTC 损失函数计算方式。
        使用这种方式能够提升cer一点，大概0.07%(9.35%->9.28%)左右。

        Args:
            predictions: 模型预测输出, torch.tensor [T, C]
            beam_size: 集束宽度

        Returns:
            results: List[(str, float),....] length=beam_size
        """

        def log_add(args: List[int]) -> float:
            """
            Stable log add
            """
            if all(a == -float("inf") for a in args):
                return -float("inf")
            a_max = max(args)
            lsp = math.log(sum(math.exp(a - a_max) for a in args))
            return a_max + lsp

        ctc_probs = F.log_softmax(predictions, dim=-1)  # L*C
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float("inf")))]
        # 2. CTC beam search step by step
        for t in range(0, ctc_probs.shape[0]):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float("inf"), -float("inf")))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == self.blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(), key=lambda x: log_add(list(x[1])), reverse=True
            )
            cur_hyps = next_hyps[:beam_size]
        hyps = [
            (
                self.decoder(
                    torch.Tensor(list(y[0])).unsqueeze(0),
                    torch.Tensor([len(y[0])]).unsqueeze(0),
                )[0],
                log_add([y[1][0], y[1][1]]),
            )
            for y in cur_hyps
        ]
        return hyps

    def encoder(self, s: str) -> torch.Tensor:
        """对字符串进行直接编码
            1. 先进行大写转小写
            2. 未在词典中的词语忽略

        Args:
            s (str): to be converted string

        Returns:
            torch.Tensor: torch.LongTensor shape: [len(s), ]
        """
        ids = []
        # 1. 大写转小写
        s = s.lower()
        s_new = ""
        for c in s:
            # 2. 未在词典中的直接忽略
            if c not in self.s2labels_map.keys():
                # logging.debug(f"char: {c} is not in dict and will be replace as _")
                # ids.append(self.blank_id)
                pass
            else:
                s_new += c
        # 3. 字符串去掉多余空格和首字母去掉空格
        s_new = s_new.replace("  ", " ").strip()
        for c in s_new:
                ids.append(self.s2labels_map[c])
        return torch.LongTensor(ids)

    def export_vocab(self):
        """导出词典

        Example:

        >>> vocab = ['_', ' ', 'a', 'b', 'c']
        >>> tokenizer = CTCTokenizer(vocab=vocab)
        >>> tokenizer.export_vocab()
        ['_', ' ', 'a', 'b', 'c']
        """

        return [self.labels_map[i] for i in range(len(self.labels_map))]


if __name__ == "__main__":
    vocab = ["_", " ", "a", "b", "c"]
    # vocab = "/home/cc/workdir/code/wav2vec-exp/s3prl-example/vocab.txt"
    tokenizer = CTCTokenizer(vocab=vocab)
    predict_tensor = torch.randn((8, 200, 6), dtype=torch.float32)
    predict_len = torch.IntTensor([100, 20, 40, 80, 120, 150, 180, 200])
    out = tokenizer.ctc_decode(torch.argmax(predict_tensor, dim=-1, keepdim=False))
    vocab = tokenizer.export_vocab()
    out_beam_search = tokenizer.parallel_ctc_prefix_search(predict_tensor, predict_len, 10)
    out2 = tokenizer._ctc_prefix_beam_search(predictions=predict_tensor[0], beam_size=4)

    print()
