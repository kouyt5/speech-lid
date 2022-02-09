from collections import defaultdict
from typing import Any, List
import logging
from torch.utils.data import Dataset, DataLoader
import os
import torch
import random

from tokenizer import Tokenizer, read_and_filter, build_vocab
from torch.nn.utils.rnn import pad_sequence

class Wiki102Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_len: int = 500,
        tokenizer: Tokenizer = None,
        mask: bool = False,
        mask_prob: float = 0.01,
    ) -> None:
        super().__init__()
        #  ["I like LM...", "He hate LM...."]
        self.datasets = read_and_filter(data_path)
        self.max_len = max_len
        self.mask = mask
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer
                
        logging.info(f"datasets size: {len(self.datasets)}")
        

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index) -> Any:
        s = self.datasets[index]
        origin_s = s  # for no mask
        if self.mask:
            choice_len = int(self.mask_prob * len(s.split(' ')))
            choice_len = random.randint(0, choice_len)
            replace_list = random.sample(range(len(s.split(' '))), choice_len)
            replace_s = s.split(' ')
            for index in replace_list:
                replaced_word = self.tokenizer.num2vocab[random.randint(0, len(self.tokenizer.vocab2num.keys()) - 1)]
                replace_s[index] = replaced_word
            s = ' '.join(replace_s)
        return self.tokenizer.encoder(s), self.tokenizer.encoder(s)

    def collate_fn(self, batch):
        x = [batch[i][0] for i in range(len(batch))]
        length = torch.Tensor([len(item) for item in x]).long()
        target = [batch[i][1] for i in range(len(batch))]
        target = pad_sequence(target, batch_first=True)
        x = pad_sequence(x, batch_first=True)
        return x, length, target

if __name__ == "__main__":

    data_path = "/home/cc/workdir/code/lm/data/wikitext-2-raw/wiki.train.raw"
    vocab = build_vocab(data_path)
    tokenizer = Tokenizer(vocab)
    
    dataset = Wiki102Dataset(data_path=data_path, tokenizer=tokenizer)
    item = dataset.__getitem__(0)
    print(tokenizer.decoder(item.unsqueeze(0))[0])
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        batch
