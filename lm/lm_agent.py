from typing import Any, List, Tuple

from tokenizer import Tokenizer
from ccml.ccml_module import CCMLModule

import torch
from lm.tokenizer import build_vocab

from model.lstm_model import LM_LSTM
import torch.nn.functional as F


class LMModule(CCMLModule):
    def __init__(
        self,
        lr: float = 0.01,
        wd: float = 0.,
        vocab: List = None,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = LM_LSTM(
            vocab=vocab,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.save_hyper_parameters(
            {
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "lstm_dropout": lstm_dropout,
                "bidirectional": bidirectional,
                "vocab": vocab,
            }
        )
        self.lr = lr
        self.wd = wd

    def common_loop(self, batch):
        x, length, target = batch
        target = target[:, 1:]
        out = self.model(x, length.cpu())
        p_predict = []
        bpc = []
        ppl = []
        loss = 0.
        for i in range(out.shape[0]):
            i_loss = F.cross_entropy(out[i,:length[i]-1, :], target[i, :length[i]-1])
            loss += i_loss
            p_predict.append(F.softmax(out[i,:length[i], :].detach(), dim=-1))
            pred_label_prob = F.softmax(out[i,:length[i] - 1, :].detach(), dim=-1)[torch.arange(length[i] - 1),  target[i, :length[i]-1]]
            bpc.append(-torch.sum(torch.log2(pred_label_prob))/(length[i] - 1))
            ppl.append(torch.exp(i_loss))
        # loss = F.cross_entropy(out[:, :-1, :].transpose(1, 2), target)
        loss = loss / out.shape[0]
        ppl = torch.mean(torch.as_tensor(ppl))
        bpc = torch.mean(torch.as_tensor(bpc))
        return loss, torch.exp(loss), bpc

    def next_char_infer(self, batch):
        x, length,target = batch
        out = self.model(x, length.cpu())[0, -1, :]
        idx = torch.argmax(out, dim=-1)
        return idx

    def train_loop(self, batch):
        loss, ppl, bpc = self.common_loop(batch)

        self.trainer.logger.log(
            data={"loss": loss, "train_ppl": torch.mean(ppl), "bpc": torch.mean(bpc)},
            progress=True,
            stage="train",
        )
        return {"loss": loss, "ppl": torch.mean(ppl), "bpc": torch.mean(bpc)}

    def val_loop(self, batch):
        loss, ppl, bpc = self.common_loop(batch)
        return {
            "val_loss": loss,
            "val_ppl": torch.mean(ppl),
            "val_bpc": torch.mean(bpc),
        }

    def val_loop_end(self, outputs: List[Any] = None):
        total_val_loss = 0.0
        total_val_ppl = 0.0
        total_val_bpc = 0.0
        for item in outputs:
            total_val_loss += item["val_loss"]
            total_val_ppl += item["val_ppl"]
            total_val_bpc += item["val_bpc"]
        self.trainer.logger.log(
            data={
                "val_avg_loss": total_val_loss / len(outputs),
                "val_ppl": total_val_ppl / len(outputs),
                "val_bpc": total_val_bpc / len(outputs),
                "epoch": self.trainer.current_epoch,
            },
            progress=True,
            stage="val",
            commit=False,
        )
        self.trainer.logger.remove_key(["val_avg_loss"])

    def config_optim(
        self, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        # )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            cooldown=3,
            min_lr=1e-4,
        )
        return optimizer, scheduler, {"monitor": "val_loss", "interval": "epoch"}


if __name__ == "__main__":
    ckpt_path = "/home/cc/workdir/code/lm/exp/lr1e3_wd1e4_dr5_em128/last.pt"
    train_manifest = "/home/cc/workdir/code/lm/data/wikitext-2/wiki.train.tokens"
    vocab = build_vocab(train_manifest, word_level=True, min_count=2)
    tokenizer = Tokenizer(vocab=vocab)
    text = "As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical"
    tensor = tokenizer.encoder(text)
    print("decoder: "+str(tokenizer.decoder(tensor.unsqueeze(0))))
    length = torch.LongTensor([tensor.shape[0]])
    tensor = tensor.unsqueeze(0)

    ccml_module = LMModule.resume_from_checkpoint(ckpt_path, map_location="cpu")
    loss, ppl, bpc = ccml_module.common_loop((tensor, length))
    next_idx = ccml_module.next_char_infer((tensor, length))
    print("next: " + str(tokenizer.num2vocab[next_idx.item()]))
    print()
