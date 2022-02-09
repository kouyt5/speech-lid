from collections import defaultdict
import os
from typing import List
import logging
import torch
import numpy


def read_and_filter(data_path: str) -> List[str]:
    """read the wiki data and remove blank or meaningless line

    Args:
        data_path (str): path to the wiki txt data.
            egs: /path/to/your/wiki.test.raw

    Returns:
        List[str]: text list
            egs: ["Robert Boulter is an English film", "..."]
    """
    if not os.path.exists(data_path):
        raise Exception(f"data path {data_path} is not exist!")

    all_lines = open(data_path).readlines()

    filter_lines = []

    for line in all_lines:
        # filter 1 remove blank line
        if len(line.strip()) == 0:
            continue
        if line.startswith(" ="):
            continue
        if len(line.strip().split(' ')) < 2:
            continue
        if len(line.split()) > 500:
            continue
        filter_lines.append(line.strip())
    return sorted(filter_lines)


def build_vocab(data_path: str, word_level: bool = False, min_count: int = 1) -> List:
    """from train txt to build a sorted vocab

    Args:
        txt_list (List): list contain txt

    Returns:
        List: list container vocab
    """
    txt_list = read_and_filter(data_path)
    vocab = defaultdict()
    if not word_level:  # char level
        for item in txt_list:
            for c in item:
                if c in vocab.keys():
                    vocab[c] += 1
                else:
                    vocab[c] = 1
    else:
        for item in txt_list:
            words = item.split()
            for word in words:
                if word not in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    vocab["<unk>"] = min_count+1  # ensure that <unk> not be filtered
    sorted_vocab = sorted(vocab.items(), key=lambda kv: (kv[1], kv[0]))
    vocab.clear()
    # filter vocab < MIN_COUNT
    for item in sorted_vocab:
        if item[1] < min_count:
            continue
        vocab[item[0]] = item[1]
    return list(vocab.keys())


class Tokenizer:
    def __init__(self, vocab: List):
        self.vocab2num = dict([(vocab[i], i) for i in range(len(vocab))])
        self.num2vocab = dict([(i, vocab[i]) for i in range(len(vocab))])

    def encoder(self, s: str, word_level:bool = True) -> torch.Tensor:
        x = []
        # char level
        if not word_level:
            for c in s:
                if c not in self.vocab2num.keys():
                    x.append(self.vocab2num['<unk>'])
                    continue
                x.append(self.vocab2num[c])
        else:
            word_list = s.split(' ')
            for word in word_list:
                if word not in self.vocab2num.keys():
                    x.append(self.vocab2num['<unk>'])
                    continue
                x.append(self.vocab2num[word])
        x = torch.LongTensor(x)
        return x

    def decoder(self, x: torch.Tensor, mask: torch.Tensor = None) -> List[str]:
        x = x.detach().cpu().numpy()
        result = []
        if mask is not None:
            raise NotImplementedError()
        for i in range(len(x)):
            result.append(" ".join([self.num2vocab[c] for c in x[i].tolist()]))
        return result
