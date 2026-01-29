import torch, random, os, ipdb, logging
import numpy as np
from .base_cp import BaseCP
from .utils import compute_guassian_weights
from scipy.stats import multivariate_normal
from metrics import analyze_distribution
from tqdm import tqdm


logger = logging.getLogger("global_logger")



class CalibratedLocalizedCP(BaseCP):

    def __init__(
        self,
        bandwidth = 0.1,
        batch_size = 1024,
        weight_analysis_metrics = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.cp_method = self.cal_lcp
        self.batch_size = batch_size

        if len(weight_analysis_metrics.keys()) > 0:
            self.weighst_stats = {k:{"kwargs": weight_analysis_metrics[k], "stats": []} for k in weight_analysis_metrics}
        else:
            self.weighst_stats = {}


    def cal_lcp(
        self, 
        all_outs, 
        cal_idxs,
        test_idxs,
        pre_weights = None,
        **kwargs
    ):

        test_scores, test_x, test_labels = [all_outs[k][test_idxs.numpy()] for k in ["scores", "x", "labels"]]
        cal_scores, cal_x, cal_labels = [all_outs[k][cal_idxs.numpy()] for k in ["scores", "x", "labels"]]
        n_test = test_scores.shape[0]

        n_cal, n_cls = cal_scores.shape
        if self.task == "reg":
            cal_nc_scores = np.abs(cal_labels - cal_scores).reshape(-1)
        else:
            test_nc_scores = 1 - test_scores
            cal_nc_scores = 1 - cal_scores[np.arange(n_cal), cal_labels]

        cal_nc_idxs = np.argsort(cal_nc_scores)
        cal_nc_scores = cal_nc_scores[cal_nc_idxs]
        cal_x = cal_x[cal_nc_idxs, :]

        nc_scores = np.concatenate((cal_nc_scores, [np.inf]))

        if pre_weights is not None:
            pre_weights = np.concatenate((pre_weights, np.ones((1,), dtype=cal_nc_scores.dtype)))

        cal_cal_weights = []
        for i in range(0, n_cal, self.batch_size):
            cal_cal_weights.append(
                compute_guassian_weights(
                    cal_x[i:i+self.batch_size], 
                    cal_x, 
                    bandwidth = self.bandwidth,
                )
            )

        cal_cal_weights = np.concatenate(cal_cal_weights, axis=0)
        denom_cal = np.sum(cal_cal_weights, axis=1)

        ext_cal_scores = np.tile(cal_nc_scores[None, :], (n_cal, 1))
        num_cal = np.sum(cal_cal_weights * (ext_cal_scores < cal_nc_scores[:, None]).astype(np.float64), axis=1)

        test_pred_sets = []
        quantile_thresh = []

        for i in tqdm(range(0, n_test, self.batch_size)):
            start, end = i, min(i + self.batch_size, n_test)
            b_n_test = end - start

            b_test_x = test_x[start:end]
            b_test_scores = test_scores[start:end]
            b_quantile_thresh = np.zeros_like(b_test_scores)

            if self.task == "reg":
                b_test_conf_pred = np.zeros((b_test_scores.shape[0], n_cls, 2), dtype=b_test_scores.dtype)
            else:
                b_test_conf_pred = np.zeros_like(b_test_scores)

            test_cal_weights = compute_guassian_weights(
                b_test_x, 
                cal_x, 
                test_x = b_test_x,  
                bandwidth = self.bandwidth, 
                normalize = False, 
            )
            
            if self.weighst_stats is not None:
                for k in self.weighst_stats.keys():
                    self.weighst_stats[k]["stats"].append(
                        analyze_distribution(test_cal_weights, metric_name=k, **self.weighst_stats[k]["kwargs"])
                    )

            test_cal_weights_sum = np.sum(test_cal_weights, axis=1)

            #NOTE: The following line doesn't match with the equations in the paper.
            extended_nc_scores = np.tile(nc_scores[None, :], (b_n_test, 1))

            extended_num_cal = np.tile(num_cal[None, :], (b_n_test, 1))
            extended_denom_cal = np.tile(denom_cal[None, :], (b_n_test, 1))
            T_values = np.zeros((b_n_test, n_cal+1))
            T_values[:, :-1] = (extended_num_cal + test_cal_weights[:, :-1]) / (extended_denom_cal + test_cal_weights[:, :-1])
            #NOTE: The following line doesn't match with the equations in the paper.
            T_values[:, -1] = np.sum(test_cal_weights * (extended_nc_scores < extended_nc_scores[:, 0][:, None]).astype(np.float64), axis=1) / test_cal_weights_sum

            extended_score_mask = (np.tile(nc_scores[None , None, :], (b_n_test, n_cal+1, 1)) < np.tile(nc_scores[None, :], (b_n_test, 1))[:, :, None]).astype(np.float64)
            extended_weights = np.tile(test_cal_weights[:, None, :], (1, n_cal+1, 1))
            extended_weight_sum = np.tile(test_cal_weights_sum[:, None], (1, n_cal+1))
            extended_T_values = np.tile(T_values[:, None, :], (1, n_cal+1, 1))
            extended_T_values[:, :, -1] = np.sum(extended_weights * extended_score_mask, axis=-1) / extended_weight_sum
            extended_T_values[:, np.arange(1, n_cal+1), np.arange(n_cal)] = extended_num_cal / (extended_denom_cal + test_cal_weights[:, :-1])

            uniform_muls = np.random.uniform(0, 1, (b_n_test, n_cls))
            p_values = np.zeros((b_n_test, n_cal+1))

            for j in range(n_cls):

                #NOTE: The following line doesn't match with the euquaions in the paper. Also, the uniform_mults must be modified to capture all classes.
                p_values[:, 0] = (np.sum(
                    (T_values > T_values[:, -1][:, None]), axis=-1) + \
                    uniform_muls[:, j] * np.sum((T_values == T_values[:, -1][:, None]), axis=1)
                ) / (n_cal+1)

                p_values[:, 1:] = ((
                    np.sum(extended_T_values > extended_T_values[:, :, -1][:, :, None], axis=-1) + \
                    uniform_muls[:, j][:, None] * np.sum(extended_T_values == extended_T_values[:, :, -1][:, :, None], axis=-1)
                ) / (n_cal+1))[:, 1:]

                p_mask = np.zeros((p_values.shape[0],), dtype=np.bool)
                idx0, idx1 = np.nonzero(p_values > self.alpha)
                certain_idxs = np.unique(idx0)
                p_mask[certain_idxs] = True
                smallest_idxs = (np.roll(idx0, shift=-1) != idx0).nonzero()[0]

                b_quantile_thresh[p_mask, j] = extended_nc_scores[p_mask, idx1[smallest_idxs]]

                if self.task == "reg":
                    b_quantile_thresh[~p_mask, j] = 0.0
                    b_lower_bounds = b_test_scores[:, j] - b_quantile_thresh[:, j]
                    b_upper_bounds = b_test_scores[:, j] + b_quantile_thresh[:, j]
                    
                    b_test_conf_pred[:, j, :] = np.stack([b_lower_bounds, b_upper_bounds], axis=1)
                else:
                    closed_val = np.zeros((certain_idxs.shape[0],))
                    e_mask = (idx1[smallest_idxs] == n_cal)
                    e_idx0 = certain_idxs[e_mask]
                    e_idx1 = idx1[smallest_idxs[e_mask]]

                    closed_mask = np.zeros((b_n_test,), dtype=np.bool)
                    closed_mask[p_mask][e_mask] = False

                    ne_idx0 = certain_idxs[~e_mask]
                    ne_idx1 = idx1[smallest_idxs[~e_mask]]

                    T_values[ne_idx0, -1] = np.sum(
                        test_cal_weights[ne_idx0] * \
                        (extended_nc_scores[ne_idx0] < extended_nc_scores[ne_idx0, ne_idx1][:, None]).astype(np.float64), axis=1
                    ) / test_cal_weights_sum[ne_idx0]

                    idx_mat = np.tile(np.arange(0, n_cal+1), (ne_idx0.shape[0], 1))
                    temp_idx0, temp_idx1 = np.nonzero(idx_mat < ne_idx1[:, None])
                    T_values[ne_idx0[temp_idx0], temp_idx1] = extended_num_cal[ne_idx0[temp_idx0], temp_idx1] / (
                        extended_denom_cal[ne_idx0[temp_idx0], temp_idx1] + \
                        test_cal_weights[ne_idx0[temp_idx0], temp_idx1]
                    )

                    le_mask = (idx1[smallest_idxs] < n_cal)
                    le_idx0 = certain_idxs[le_mask]
                    le_idx1 = idx1[smallest_idxs[le_mask]]
                    idx_mat = np.tile(np.arange(0, n_cal), (le_idx0.shape[0], 1))
                    temp_idx0, temp_idx1 = np.nonzero(idx_mat >= le_idx1[:, None])
                    T_values[le_idx0[temp_idx0], temp_idx1] = extended_num_cal[le_idx0[temp_idx0], temp_idx1] / (
                        extended_denom_cal[le_idx0[temp_idx0], temp_idx1] + \
                        test_cal_weights[le_idx0[temp_idx0], temp_idx1]
                    )
                    temp_p_values = (
                        np.sum((T_values > T_values[:, -1][:, None]), axis=-1) + \
                        uniform_muls[:, j] * np.sum((T_values == T_values[:, -1][:, None]), axis=1)
                    ) / (n_cal+1)

                    closed_mask[p_mask][~e_mask] = temp_p_values[p_mask][~e_mask] > self.alpha

                    b_test_nc_scores = test_nc_scores[start:end]
                    b_test_conf_pred[closed_mask, j] = b_test_nc_scores[closed_mask, j] < b_quantile_thresh[closed_mask, j]
                    b_test_conf_pred[~closed_mask, j] = b_test_nc_scores[~closed_mask, j] <= b_quantile_thresh[~closed_mask, j]

            test_pred_sets.append(b_test_conf_pred)
            quantile_thresh.append(b_quantile_thresh)

        quantile_thresh = np.concatenate(quantile_thresh, axis=0)
        test_pred_sets = np.concatenate(test_pred_sets, axis=0)

        if self.weighst_stats is not None:
             self.weighst_stats = {
                k: {"kwargs": v["kwargs"], "stats": [np.concatenate(v["stats"], axis=0)]} for k, v in self.weighst_stats.items()
            }

        return test_pred_sets, quantile_thresh