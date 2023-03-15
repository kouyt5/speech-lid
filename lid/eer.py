import torchmetrics
import torchmetrics.functional.classification as F
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
from sklearn.metrics import roc_curve


class EER(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, num_class=3):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_on_step=True,
            full_state_update=False,
        )
        self.num_class = num_class

    def update(self, predict: list, target: list) -> None:
        predict_score_list = []
        pre_index = np.argmax(predict, axis=-1)
        pos_list = []
        for i in range(len(predict)):
            for j in range(len(predict[i])):
                predict_score_list.append(predict[i][j])
                pos_list.append(int(j == target[i]))
        with torch.no_grad():
            fpr, tpr, thresholds = roc_curve(pos_list, predict_score_list)
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds

    def compute(self):
        fpr = list(self.fpr)
        tpr = list(self.tpr)
        return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0, 1.0)


class EER2(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, num_class=3):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_on_step=True,
            full_state_update=False,
        )
        self.num_class = num_class
        self.add_state("pos_list", default=[], dist_reduce_fx="cat")
        self.add_state("score_list", default=[], dist_reduce_fx="cat")

    def update(self, predict: list, target: list) -> None:
        predict_score_list = []
        pos_list = []
        for i in range(len(predict)):
            for j in range(len(predict[i])):
                predict_score_list.append(predict[i][j])
                pos_list.append(int(j == target[i]))
        self.pos_list.extend(pos_list)
        self.score_list.extend(predict_score_list)

    def compute(self):
        fpr, tpr, thresholds = roc_curve(self.pos_list, self.score_list)
        fpr = list(fpr)
        tpr = list(tpr)
        return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0, 1.0)
    
class CAvg(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, num_class=3):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_on_step=True,
            full_state_update=False,
        )
        self.num_class = num_class
        self.add_state("pairs", default=[], dist_reduce_fx="cat")  # [(0, 1, 0.44)]
        
    def update(self, predict: list, target: list) -> None:
        for i in range(len(predict)):
            for j in range(len(predict[i])):
                self.pairs.append((j, target[i], predict[i][j]))
    
    def compute(self):
        min_score = min([pair[2] for pair in self.pairs])
        max_score = max([pair[2] for pair in self.pairs])
        cavgs, min_cavg = self.get_cavg(self.pairs, self.num_class, min_score, max_score, 20, 0.5)
        res = round(min_cavg, 4)
        return res
        
    def get_cavg(self, pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
        ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
        '''
        cavgs = [0.0] * (bins + 1)
        precision = (max_score - min_score) / bins
        for section in range(bins + 1):
            threshold = min_score + section * precision
            # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
            target_cavg = [0.0] * lang_num
            for lang in range(lang_num):
                p_miss = 0.0 # prob of missing target pairs
                LTa = 0.0 # num of all target pairs
                LTm = 0.0 # num of missing pairs
                p_fa = [0.0] * lang_num # prob of false alarm, respect to all other langs
                LNa = [0.0] * lang_num # num of all nontarget pairs, respect to all other langs
                LNf = [0.0] * lang_num # num of false alarm pairs, respect to all other langs
                for line in pairs:
                    if line[0] == lang:
                        if line[1] == lang:
                            LTa += 1
                            if line[2] < threshold:
                                LTm += 1
                        else:
                            LNa[line[1]] += 1
                            if line[2] >= threshold:
                                LNf[line[1]] += 1
                if LTa != 0.0:
                    p_miss = LTm / LTa
                for i in range(lang_num):
                    if LNa[i] != 0.0:
                        p_fa[i] = LNf[i] / LNa[i]
                p_nontarget = (1 - p_target) / (lang_num - 1)
                target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
            cavgs[section] = sum(target_cavg) / lang_num

        return cavgs, min(cavgs)    
    
if __name__ == "__main__":
    eer = EER2()
    eer.update([[0.1,0.2,0.7]], [0])
    eer.update([[0.2,0.1,0.7]], [1])
    print(eer.compute())
    eer.reset()
    eer.update([[0.1,0.2,0.7]], [0])
    eer.update([[0.2,0.1,0.7]], [1])
    eer.compute()
