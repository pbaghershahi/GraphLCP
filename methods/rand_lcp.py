import torch, random, os, ipdb, logging
import numpy as np
from .base_cp import BaseCP
from .utils import compute_guassian_weights, sample_from_guassian_kernel, sample_ppr_anchor, compute_ppr_weights
from scipy.stats import multivariate_normal
from metrics import analyze_distribution
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix, degree



logger = logging.getLogger("global_logger")



class RandomlyLocalizedCP(BaseCP):

    def __init__(
        self,
        bandwidth = 0.1,
        batch_size = 1024,
        weight_analysis_metrics = {},
        base_method = "split",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.cp_method = self.rand_lcp
        self.batch_size = batch_size
        self.base_method = base_method

        if base_method == "aps":
            assert self.task == "cls", "APS-based weighting only supports classification tasks."
        
        if len(weight_analysis_metrics.keys()) > 0:
            self.weighst_stats = {k:{"kwargs": weight_analysis_metrics[k], "stats": []} for k in weight_analysis_metrics}
        else:
            self.weighst_stats = {}
    

    def _compute_guassian_weights(self, center_x, cal_x, test_x):
        """
        inputs:
        - center_x: embeddings of the sampled points in the proximity of test points -> (n, d)
        - cal_x: embeddings of the calibration set -> (m, d)
        - test_x: embeddings of the test points -> (n, d)

        outputs:
        - weights: normallized weights for all the calibration points and the test points -> (n, m+1)
        """
        weights = compute_guassian_weights(
            center_x, cal_x, test_x, bandwidth=self.bandwidth, normalize=True, eigenvectors=None
        )
        return weights


    def rand_lcp(
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
            test_nc_scores = np.zeros_like(test_scores)
            cal_nc_scores = np.abs(cal_labels - cal_scores).reshape(-1)
        elif self.task == "cls" and self.base_method == "split":
            test_nc_scores = 1 - test_scores
            cal_nc_scores = 1 - cal_scores[np.arange(n_cal), cal_labels]
        elif self.task == "cls" and self.base_method == "aps":
            test_nc_scores = aps_transform(test_scores)
            cal_nc_scores = aps_transform(cal_scores)[np.arange(n_cal), cal_labels]
        else:
            raise ValueError(f"Unsupported task ({self.task}) and base method ({self.base_method}) combination.")

        cal_nc_idxs = np.argsort(cal_nc_scores)
        cal_nc_scores = cal_nc_scores[cal_nc_idxs]
        cal_x = cal_x[cal_nc_idxs, :]

        nc_scores = np.concatenate((cal_nc_scores, [np.inf])) # nc_scores: (n_val+1)
        unique_scores, unique_scores_inv = np.unique(nc_scores, return_inverse=True)
        n_unique = unique_scores.shape[0]

        if pre_weights is not None:
            pre_weights = np.concatenate((pre_weights[cal_nc_idxs], np.ones((1,), dtype=test_nc_scores.dtype)))

        test_conf_pred = []
        quantile_thresh = []

        for i in tqdm(range(0, n_test, self.batch_size)):

            start, end = i, min(i + self.batch_size, n_test)
            b_n_test = end - start

            b_test_x = test_x[start:end]
            b_test_scores = test_scores[start:end]
            b_quantile_thresh = np.full_like(b_test_scores, -np.inf)

            if self.task == "reg":
                b_test_conf_pred = np.zeros((b_test_scores.shape[0], n_cls, 2), dtype=b_test_scores.dtype)
            else:
                b_test_conf_pred = np.zeros_like(b_test_scores)

            b_test_tilde_x = sample_from_guassian_kernel(
                b_test_x, 
                sampling_scale = self.bandwidth,
            )

            weights = self._compute_guassian_weights(
                b_test_tilde_x, 
                cal_x, 
                b_test_x, 
            )

            if self.weighst_stats is not None:
                for k in self.weighst_stats.keys():
                    self.weighst_stats[k]["stats"].append(
                        analyze_distribution(weights, metric_name=k, **self.weighst_stats[k]["kwargs"])
                    )

            if pre_weights is not None:
                weights *= pre_weights.reshape(1, -1)
                weights /= (weights.sum(axis=1) + 1e-9).reshape(-1, 1)

            uniform_muls = np.random.uniform(0, 1, (b_n_test, n_cls, 1))

            w_tilde = np.zeros((b_n_test, n_unique))
            row_idxs = np.tile(np.arange(b_n_test)[:, None], (1, unique_scores_inv.shape[0])) # row_idxs: n_test x (n_cal+1)
            col_idxs = np.tile(unique_scores_inv[None, :], (b_n_test, 1)) # col_idxs: n_test x (n_cal+1)
            np.add.at(w_tilde, (row_idxs, col_idxs), weights)

            for j in range(n_cls):
                
                """
                Computing conformal p-value. 
                For every s_k (nonconformity score of a calibration sample) we have:
                    p_k = (1 + \sum_{j=1}^n_cal 1(s_j >= s_k))/(1 + n_cal)
                """

                uniform_col = uniform_muls[:, j] * w_tilde[:, -1][:, None]
                main_cols = (np.cumsum(w_tilde[:, :-1][:, ::-1], axis=1))[:, ::-1] + uniform_col
                p_values = np.concatenate((main_cols, uniform_col), axis=1)
        
                """
                Computing the quntile thresholds per test sample.
                    threshold_i = p_value of the calibration sample which has the lowest p_value above alpha
                """

                p_mask_all = p_values > self.alpha
                p_mask = p_mask_all.any(axis=1)
                idx0, idx1 = np.nonzero(p_mask_all)

                if idx0.shape[0] == 0:
                    smallest_idxs = np.array([], dtype=int)
                    certain_idxs = np.array([], dtype=int)
                elif idx0.shape[0] == 1:
                    smallest_idxs = np.array([idx1.max()])
                    certain_idxs = np.unique(idx0)
                else:
                    smallest_idxs_mask = (np.roll(idx0, shift=-1) != idx0)
                    smallest_idxs_mask[-1] = True
                    smallest_idxs = idx1[smallest_idxs_mask]
                    certain_idxs = idx0[smallest_idxs_mask]

                extended_unique_scores = np.tile(unique_scores[None, :], (n_test, 1)) # extended_unique_scores: n_test x n_unique

                b_quantile_thresh[p_mask, j] = extended_unique_scores[certain_idxs, smallest_idxs]

                if self.task == "reg":
                    b_quantile_thresh[~p_mask, j] = 0.0
                
                    # --- 9. Compute Prediction Interval ---
                    # The interval is [f(x) - q, f(x) + q]
                    b_lower_bounds = b_test_scores[:, j] - b_quantile_thresh[:, j]
                    b_upper_bounds = b_test_scores[:, j] + b_quantile_thresh[:, j]
                    
                    b_test_conf_pred[:, j, :] = np.stack([b_lower_bounds, b_upper_bounds], axis=1)
                else:
                    """
                    Check which intervals are closed and which ones are not. 
                    """
                    closed_val = np.zeros((certain_idxs.shape[0],))

                    # Mask for indices that are not the last unique score (i.e., not infinity)
                    le_mask = (smallest_idxs < n_unique -1)
                    le_idx0 = certain_idxs[le_mask]
                    le_idx1 = smallest_idxs[le_mask]
                    le_val = p_values[le_idx0, le_idx1+1] + uniform_muls[le_idx0, j].reshape(-1) * w_tilde[le_idx0, le_idx1]
                    closed_val[le_mask] = le_val

                    ge_idx0 = certain_idxs[~le_mask]
                    ge_idx1 = smallest_idxs[~le_mask]
                    ge_val = p_values[ge_idx0, -1] + uniform_muls[ge_idx0, j].reshape(-1) * w_tilde[ge_idx0, ge_idx1]
                    closed_val[~le_mask] = ge_val
                
                    closed_mask = np.zeros((b_n_test,), dtype=np.bool)
                    closed_mask[p_mask] = closed_val > self.alpha

                    b_test_nc_scores = test_nc_scores[start:end]
                    b_test_conf_pred[closed_mask, j] = b_test_nc_scores[closed_mask, j] < b_quantile_thresh[closed_mask, j]
                    b_test_conf_pred[~closed_mask, j] = b_test_nc_scores[~closed_mask, j] <= b_quantile_thresh[~closed_mask, j]
            
            test_conf_pred.append(b_test_conf_pred)
            quantile_thresh.append(b_quantile_thresh)

        quantile_thresh = np.concatenate(quantile_thresh, axis=0)
        test_conf_pred = np.concatenate(test_conf_pred, axis=0)

        if self.weighst_stats is not None:
            self.weighst_stats = {
                k: {"kwargs": v["kwargs"], "stats": [np.concatenate(v["stats"], axis=0)]} for k, v in self.weighst_stats.items()
            }
        
        outputs = {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
        
        return outputs