import logging
from typing import List, Union
import torch



class CTCTokenizer:
    """CTC Tokenizer: convert number to string with CTC beam search
    """
    def __init__(self,
                vocab:Union[str, list]) -> None:
        if isinstance(vocab, str):
            with open(vocab) as f:
                all_lines = [s.replace('\n', '') for s in f.readlines()]
                self.labels_map = dict((i, all_lines[i]) for i in range(len(all_lines)))
        elif isinstance(vocab, list):
            self.labels_map = dict((i, vocab[i]) for i in range(len(vocab)))
        else:
            raise Exception("vocab is neither str or list, please check")
        # labels_map: {0, '_'}
        self.s2labels_map = dict((self.labels_map[key], key) for key in self.labels_map.keys())
        self.blank_id = len(self.labels_map)
        
        
    def ctc_decode(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None) -> List[str]:
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
            hypothesis = ''.join([self.labels_map[c] for c in decoded_prediction])

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
                reference = ''.join([self.labels_map[c] for c in target])
                references.append(reference)
        return references
    def encoder(self, s:str) -> torch.Tensor:
        """encoder string to numbers Tensor

        Args:
            s (str): to be converted string

        Returns:
            torch.Tensor: torch.LongTensor shape: [len(s), ]
        """
        ids = []
        for c in s:
            if c not in self.s2labels_map.keys():
                logging.warning(f'char: {c} is not in dict and will be ignore')
                continue
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
    vocab = ['_', ' ', 'a', 'b', 'c']
    # vocab = "/home/cc/workdir/code/wav2vec-exp/s3prl-example/vocab.txt"
    tokenizer = CTCTokenizer(vocab=vocab)
    predict_tensor = torch.randn((2, 20, 6), dtype=torch.float32)
    out = tokenizer.ctc_decode(torch.argmax(predict_tensor, dim=-1, keepdim=False))
    out = tokenizer.export_vocab()
    print()
    
    import time
    query = [1,2,6,-2,5555,268,-54, 465,1576, 44444, -443, 4534, -43,-25765, -424233,554354,5325,76465755]
    d = dict((i, 1) for i in range(1000000))
    pre_time = time.time()
    for k in query:
        if k in d.keys():
            print(f"find {k}")
            continue
    print(f"speed time {time.time() - pre_time}")
    
    l = [i for i in range(1000000)]
    pre_time = time.time()
    for k in query:
        if k in l:
            print(f"find {k}")
            continue
    print(f"speed time {time.time() - pre_time}")
            