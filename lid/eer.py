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
            fpr, tpr, thresholds = roc_curve(
                pos_list, predict_score_list
            )
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds

    def compute(self):
        fpr = list(self.fpr)
        tpr = list(self.tpr)
        return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0, 1.)
