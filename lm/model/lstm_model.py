from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LM_LSTM(torch.nn.Module):
    def __init__(
        self,
        vocab: List,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        lstm_dropout: float = 0.,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.emb = nn.Embedding(len(vocab), embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            batch_first=True,  # input = (batch, seq, feature)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout = lstm_dropout,
            bidirectional = bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_dim, len(vocab))
        self.dropout = nn.Dropout(lstm_dropout)
        
    def forward(self, x, length):
        x = self.emb(x)
        
        x = nn.utils.rnn.pack_padded_sequence(x, enforce_sorted=False,
                                              lengths=length, batch_first=True)
        x, h = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        #x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    import os,sys
    os.chdir("lm")
    sys.path.append(".")
    from tokenizer import build_vocab
    data_path = "/home/cc/workdir/code/lm/data/wikitext-2-raw/wiki.train.raw"
    vocab = build_vocab(data_path)
    model = LM_LSTM(
        vocab=vocab,
        embedding_dim=32,
        hidden_size=128,
        num_layers=1,
        lstm_dropout=0.1,
    )
    x = torch.randint(0, len(vocab),size=(4,512)).long()
    length = torch.randint(100, 512, size=(4,))
    out = model(x, length)
    print(out.shape)
