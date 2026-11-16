import torch, random, os, ipdb, logging
from .base_cp import BaseCP
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("global_logger")



class SplitCP(BaseCP):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cp_method = self.split_cp
    
    
    def split_cp(
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

        if self.task == "reg":
            cal_nc_scores = np.abs(cal_labels - cal_scores).reshape(-1)
        else:
            cal_nc_scores = 1 - cal_scores[np.arange(n_cal), cal_labels]
            test_nc_scores = 1 - test_scores

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

        if self.task == "reg":
            test_conf_pred = np.zeros((n_test, n_cls, 2))
            test_conf_pred[:, 0] = np.concatenate(
                (test_scores - quantile_thresh.reshape(-1, 1), test_scores + quantile_thresh.reshape(-1, 1)),
                axis=1
            )
        else:
            test_conf_pred = test_nc_scores <= quantile_thresh

        outputs = {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
        
        return outputs