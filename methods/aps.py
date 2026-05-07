import torch, random, os, ipdb, logging
from .base_cp import BaseCP
import torch.nn.functional as F
import numpy as np
from abc import ABC

logger = logging.getLogger("global_logger")



def aps_transform(scores, self_prob_coef="random", **kwargs):
    scores = np.asarray(scores)

    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    cls_scores = []

    for cls_id in range(scores.shape[1]):
        y_rank = ranks[:, cls_id].reshape(-1, 1)
        larger_scores = (scores * (ranks > y_rank).astype(int)).sum(axis=1)

        if self_prob_coef == "random":
            u_vec = np.random.rand(scores.shape[0]).astype(scores.dtype, copy=False)
        elif self_prob_coef == "ones":
            u_vec = np.ones_like(scores[:, cls_id])
        elif self_prob_coef == "zeros":
            u_vec = np.zeros_like(scores[:, cls_id])
        else:
            raise ValueError(f"Unknown self_prob_coef: {self_prob_coef}")

        cls_result = scores[:, cls_id] * u_vec + larger_scores
        cls_scores.append(cls_result.reshape(-1, 1))

    return np.hstack(cls_scores)



class APS(BaseCP):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.task == "cls", "Only supports classification."
        self.cp_method = self.aps
    
    
    def aps(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        pre_weights = None,
        **kwargs
    ):
        cal_scores, cal_labels = all_outs["scores"][cal_idxs], all_outs["labels"][cal_idxs]
        test_scores, test_labels = all_outs["scores"][test_idxs], all_outs["labels"][test_idxs]
        n_cal, n_cls = cal_scores.shape
        n_test = test_scores.shape[0]

        cal_nc_scores = aps_transform(cal_scores)[np.arange(n_cal), cal_labels]
        test_nc_scores = aps_transform(test_scores)

        if pre_weights is None:
            pre_weights = np.ones_like(cal_nc_scores) / n_cal
        else:
            pre_weights = pre_weights / pre_weights.sum()

        ord_idxs = np.argsort(cal_nc_scores)
        sorted_nc_scores = cal_nc_scores[ord_idxs]
        
        quantile_level = (1 - self.alpha) * (1 + 1/n_cal)

        cumulative_probs = np.cumsum(pre_weights[ord_idxs])
        quantile_idx = np.min(np.where(cumulative_probs >= quantile_level))
        
        quantile_thresh = sorted_nc_scores[quantile_idx]

        test_conf_pred = test_nc_scores <= quantile_thresh

        outputs = {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
        
        return outputs