import torch, random, os, ipdb, logging
import numpy as np
from .base_cp import BaseCP
from .utils import *
from .aps import aps_transform
from scipy.stats import multivariate_normal
from torch_geometric.utils import homophily, subgraph
from metrics import analyze_distribution
from tqdm import tqdm
from .plot_homophily_scp import compute_test_node_homophily


logger = logging.getLogger("global_logger")



class GraphRLCP(BaseCP):

    def __init__(
        self,
        weighting_method = "convex",
        bandwidth = 0.1,
        batch_size = 1024,
        weight_analysis_metrics = {},
        kernel_type = "gaussian",
        ppr_alpha = 0.15,
        ppr_k = 10,
        convex_dist_coeff = None,
        convex_coeff_method = "degree",
        densification_config = dict(
            threshold = 1.0,
            method = "cosine",
            adaptive = False,
            max_edge_ratio = None
        ),
        report_homophily_ratio = True,
        base_method = "split",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cp_method = self.graph_rlcp
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.kernel_type = kernel_type
        self.ppr_alpha = ppr_alpha
        self.ppr_k = ppr_k
        self.convex_dist_coeff = convex_dist_coeff if convex_dist_coeff >= 0.0 else None
        self.convex_coeff_method = convex_coeff_method
        self.densification_config = densification_config
        self.report_homophily_ratio = report_homophily_ratio
        self.base_method = base_method

        if base_method == "aps":
            assert self.task == "cls", "APS-based weighting only supports classification tasks."

        if weighting_method == "convex":
            self.weighting_fn = self.get_weights_convex
        elif weighting_method == "biased_ppr":
            self.weighting_fn = self.get_weights_biased_ppr
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")
        
        if len(weight_analysis_metrics.keys()) > 0:
            self.weighst_stats = {k:{"kwargs": weight_analysis_metrics[k], "stats": []} for k in weight_analysis_metrics}
        else:
            self.weighst_stats = {}
        

    def compute_convex_coeffs(
        self, 
        method="degree",
        x = None,
        edge_index = None, 
        hops = 0, 
        bayeisan_multiplier = 0, 
        node_degrees = None, 
        num_nodes = None, 
    ):

        if method == "degree":
            convex_coeffs = (node_degrees - node_degrees.mean()) / (node_degrees.std() + 1e-9)

        elif method == "homophily":
            # lambda_i = sigmoid( (C * mean_sim + sum_{j in N(i)} sim(x_i, x_j)) / (C + |N(i)|) )
            k_hop_edges = get_all_khop_neighbors(num_nodes, edge_index, hops, drop_self_loops=True)
            row, col = k_hop_edges[0], k_hop_edges[1]

            x_src = torch.nn.functional.normalize(x[row], dim=-1)
            x_dst = torch.nn.functional.normalize(x[col], dim=-1)
            x_sim_vec = (x_src * x_dst).sum(dim=-1)
            src_idxs, src_counts = row.unique(dim=0, return_counts=True)

            mean_sim = x_sim_vec.mean() 
            convex_coeffs = torch.zeros((num_nodes,), device=x.device) + mean_sim
            bayeisan_multiplier = max(bayeisan_multiplier, 1.0)

            convex_coeffs[src_idxs] *= bayeisan_multiplier
            convex_coeffs.scatter_reduce_(0, row, x_sim_vec.to(convex_coeffs.dtype), reduce='sum', include_self=True)
            convex_coeffs[src_idxs] = convex_coeffs[src_idxs] / (src_counts + bayeisan_multiplier)

        convex_coeffs = torch.sigmoid(convex_coeffs) 
        convex_coeffs = torch.clamp(convex_coeffs, 0.02, 0.98)
        
        return convex_coeffs

    
    def _compute_guassian_weights(self, center_x, cal_x, test_x=None, bandwidth=1.0, eigenvectors=None):
        """
        inputs:
        - center_x: embeddings of the sampled points in the proximity of test points -> (n, d)
        - cal_x: embeddings of the calibration set -> (m, d)
        - test_x: embeddings of the test points -> (n, d)

        outputs:
        - weights: normallized weights for all the calibration points and the test points -> (n, m+1)
        """
        weights = compute_guassian_weights(
            center_x, cal_x, test_x, bandwidth=bandwidth, normalize=True, eigenvectors=eigenvectors
        )
        return weights


    def _compute_ppr_weights(
        self, 
        edge_index, 
        edge_weight, 
        node_degrees, 
        anchor_idxs, 
        cal_idxs, 
        test_idxs, 
        num_nodes, 
        position_mat = None
    ):
        
        # Compute weights based on graph topology (PPR)
        # Returns (Batch_Size, Num_Total_Nodes)
        n_anchors = anchor_idxs.shape[0]
        all_weights = compute_ppr_weights(
            edge_index, 
            edge_weight,
            num_nodes, 
            anchor_idxs,
            position_mat,
            alpha=self.ppr_alpha, 
            k=self.ppr_k
        ).cpu()

        # Apply Reciprocity Correction: w_cal = P(Anchor -> Cal) / deg(Cal)
        # NOTE: The full theoretical correction is P(Anchor -> Cal) * deg(Anchor) / deg(Cal).
        # However, deg(Anchor) is constant for a given test point (row).
        # It cancels out perfectly during the final normalization step (weights /= sum).
        # Therefore, we simply divide by the destination node's degree.

        target_idxs = torch.cat([cal_idxs[None, :].tile(n_anchors, 1), test_idxs[:, None]], dim=1).cpu()
        target_degrees = node_degrees[target_idxs].cpu()
        target_weights = torch.gather(all_weights, dim=1, index=target_idxs) / target_degrees
        
        target_weights[target_weights == float('inf')] = 0
        target_weights[target_weights.isnan()] = 0
        target_weights += 1e-9
        target_weights /= target_weights.sum(dim=1, keepdims=True)
        target_weights = target_weights.cpu().numpy()

        return target_weights


    def get_weights_convex(
        self, 
        test_idxs,
        cal_idxs,
        test_x,
        cal_x,
        all_x,
        edge_index, 
        edge_weight,
        node_degrees,
        bandwidth,
        eigenvectors,
        sampling_scale,
        **kwargs
    ):

        num_nodes = all_x.shape[0]
        n_test = test_idxs.shape[0]
        n_cal = cal_idxs.shape[0]

        if not hasattr(self, "convex_coeffs"):
            if self.convex_dist_coeff is not None:
                self.convex_coeffs = torch.tensor([self.convex_dist_coeff] * num_nodes, dtype=torch.float32, device=test_idxs.device)
            else:
                convex_coeff_args = dict(method = self.convex_coeff_method)
                if self.convex_coeff_method == "degree":
                    convex_coeff_args |= dict(node_degrees = node_degrees)

                elif self.convex_coeff_method == "homophily":
                    convex_coeff_args |= dict(
                        x =  torch.from_numpy(all_x).to(self.device),
                        edge_index = edge_index.to(self.device), 
                        hops = 2, 
                        bayeisan_multiplier = 5,
                        num_nodes = num_nodes,
                    )

                else:
                    convex_coeff_args |= dict(num_nodes = num_nodes)
                
                self.convex_coeffs = self.compute_convex_coeffs(**convex_coeff_args).to(test_idxs.device)

        test_convex_coeffs = self.convex_coeffs[test_idxs]

        mask = torch.rand(n_test) <= test_convex_coeffs
        weights_ppr = np.zeros((n_test, n_cal + 1), dtype=np.float32)
        weights_rbf = np.zeros((n_test, n_cal + 1), dtype=np.float32)

        if mask.any():

            test_tilde_idxs = sample_ppr_anchor(
                edge_index, 
                test_idxs[mask].to(self.device), 
                alpha=self.ppr_alpha, 
                k=self.ppr_k, 
                num_nodes=num_nodes
            )

            weights_ppr[mask, :] = self._compute_ppr_weights(
                edge_index, 
                edge_weight,
                node_degrees,
                test_tilde_idxs,
                cal_idxs.to(self.device),
                test_idxs[mask].to(self.device), 
                num_nodes,
            )

            test_tilde_x = all_x[test_tilde_idxs.cpu().numpy(), :]
            weights_rbf[mask, :] = self._compute_guassian_weights(
                test_tilde_x, 
                cal_x, 
                test_x[mask], 
                bandwidth = bandwidth,
                eigenvectors = eigenvectors
            )

        if (~mask).any():

            test_tilde_x = sample_from_guassian_kernel(
                test_x[~mask], 
                sampling_scale = sampling_scale,
                eigenvectors = eigenvectors
            )

            weights_rbf[~mask] = self._compute_guassian_weights(
                test_tilde_x, 
                cal_x, 
                test_x[~mask], 
                bandwidth = bandwidth,
                eigenvectors = eigenvectors
            )
        
        if self.convex_dist_coeff is None:
            cal_convex_coeffs = self.convex_coeffs[cal_idxs]
            weights_ppr[:, :-1] =  weights_ppr[:, :-1] * cal_convex_coeffs[None, :].numpy()
            weights_ppr[:, -1] = weights_ppr[:, -1] * test_convex_coeffs.numpy()
            weights_rbf[:, :-1] =  weights_rbf[:, :-1] * (1 - cal_convex_coeffs[None, :].numpy())
            weights_rbf[:, -1] = weights_rbf[:, -1] * (1 - test_convex_coeffs.numpy())
            weights = weights_ppr + weights_rbf + 1e-6
        else:
            weights = test_convex_coeffs[:, None] * weights_ppr + (1 - test_convex_coeffs[:, None]) * weights_rbf + 1e-6

        weights /= weights.sum(axis=1, keepdims=True)
        
        return weights


    def get_weights_biased_ppr(
        self, 
        test_idxs,
        cal_idxs,
        test_x,
        all_x,
        edge_index, 
        edge_weight,
        node_degrees,
        bandwidth,
        eigenvectors,
        **kwargs
    ):
        num_nodes = all_x.shape[0]
        sim_mat = self._compute_guassian_weights(
            test_x, 
            all_x, 
            bandwidth = bandwidth,
            eigenvectors = eigenvectors
        )

        test_tilde_idxs = sample_ppr_anchor(
            edge_index,
            torch.from_numpy(sim_mat), 
            alpha=self.ppr_alpha, 
            k=self.ppr_k, 
            num_nodes=num_nodes
        )
        
        weights = self._compute_ppr_weights(
            edge_index.to(self.device), 
            edge_weight,
            node_degrees,
            test_tilde_idxs,
            cal_idxs.to(self.device),
            test_idxs.to(self.device), 
            num_nodes,
            torch.from_numpy(sim_mat).T
        )

        weights /= weights.sum(axis=1, keepdims=True)
        
        return weights


    def weighted_cp(
        self,
        weights,
        nc_scores,
        test_scores,
        test_nc_scores,
    ):

        n_unique = nc_scores.shape[0]
        n_samples, n_clsses = test_scores.shape
        uniform_muls = np.random.uniform(0, 1, (n_samples, n_clsses, 1))

        quantile_thresh = np.full_like(test_scores, -np.inf)
        if self.task == "reg":
            test_conf_pred = np.zeros((test_scores.shape[0], n_clsses, 2), dtype=test_scores.dtype)
        else:
            test_conf_pred = np.zeros_like(test_scores)

        for j in range(n_clsses):
            
            """
            Computing conformal p-value. 
            For every s_k (nonconformity score of a calibration sample) we have:
                p_k = (1 + \sum_{j=1}^n_cal 1(s_j >= s_k))/(1 + n_cal)
            """

            uniform_col = uniform_muls[:, j] * weights[:, -1][:, None]
            main_cols = (np.cumsum(weights[:, :-1][:, ::-1], axis=1))[:, ::-1] + uniform_col
            p_values = np.concatenate((main_cols, uniform_col), axis=1) # p_values: n_samples x n_unique

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

            extended_scores = np.tile(nc_scores[None, :], (n_samples, 1)) # extended_scores: n_samples x n_unique

            quantile_thresh[p_mask, j] = extended_scores[certain_idxs, smallest_idxs]
            if self.task == "reg":
                quantile_thresh[~p_mask, j] = 0.0
            
                # --- 9. Compute Prediction Interval ---
                # The interval is [f(x) - q, f(x) + q]
                lower_bounds = test_scores[:, j] - quantile_thresh[:, j]
                upper_bounds = test_scores[:, j] + quantile_thresh[:, j]
                
                test_conf_pred[:, j, :] = np.stack([lower_bounds, upper_bounds], axis=1)
            else:
                """
                Check which intervals are closed and which ones are not. 
                """
                closed_val = np.zeros((certain_idxs.shape[0],))

                # Mask for indices that are not the last unique score (i.e., not infinity)
                le_mask = (smallest_idxs < n_unique -1)
                le_idx0 = certain_idxs[le_mask]
                le_idx1 = smallest_idxs[le_mask]
                le_val = p_values[le_idx0, le_idx1+1] + uniform_muls[le_idx0, j].reshape(-1) * weights[le_idx0, le_idx1]
                closed_val[le_mask] = le_val

                ge_idx0 = certain_idxs[~le_mask]
                ge_idx1 = smallest_idxs[~le_mask]
                ge_val = p_values[ge_idx0, -1] + uniform_muls[ge_idx0, j].reshape(-1) * weights[ge_idx0, ge_idx1]
                closed_val[~le_mask] = ge_val
            
                closed_mask = np.zeros((n_samples,), dtype=np.bool)
                closed_mask[p_mask] = closed_val > self.alpha

                test_conf_pred[closed_mask, j] = test_nc_scores[closed_mask, j] < quantile_thresh[closed_mask, j]
                test_conf_pred[~closed_mask, j] = test_nc_scores[~closed_mask, j] <= quantile_thresh[~closed_mask, j]

        return test_conf_pred, quantile_thresh


    def graph_rlcp(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        pre_weights = None,
        pca_config = None,
        n_quantiles = 20,
        dataset = None,
    ):

        assert dataset is not None, "dataset must be provided for kernel_type='ppr'"
        num_nodes = dataset.num_nsamples
        train_idxs = dataset.train_idxs

        all_x = all_outs["x"]

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
        
        sampling_scale = self.bandwidth
        eigenvectors = None
        bandwidth = self.bandwidth
        if pca_config is not None:
            pca_eigenvalues = pca_config["pca"].explained_variance_ # the eigenvalues of PCA are exactly equivalent to the variances
            pca_eigenvectors = pca_config["pca"].components_
            if pca_config["with_dynamic_bandwidth"]:
                sampling_scale = self.bandwidth * np.sqrt(pca_eigenvalues)
                bandwidth = self.bandwidth**2 * (pca_eigenvalues + 1e-8)
                if pca_config["sample_original_space"]:
                    eigenvectors = pca_eigenvectors

        if pre_weights is not None:
            pre_weights = np.concatenate((pre_weights[cal_nc_idxs], np.ones((1,), dtype=test_nc_scores.dtype)))

        if self.densification_config["threshold"] < 1.0:
            n_init_edges = dataset._data.edge_index.shape[1]
            # dense_cutoff = 3 - 2 * self.densification_threshold  # Simplified (1 + (1-x)*2)
            last_dense_thresh = self.densification_config["threshold"]
            max_nodes_sq = num_nodes * (num_nodes - 1)
            max_adapt_itrs = 10
            min_dense_threshold = 0.01
            max_dense_threshold = 0.99

            cal_edge_index, _ = subgraph(
                subset=cal_idxs,
                edge_index=dataset._data.edge_index.cpu(),
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            homophily_ratio = graph_homophily(cal_edge_index, dataset._data.y[cal_idxs], task=self.task)

            if self.densification_config["max_edge_ratio"] is None:
                max_edge_ratio = 2 * min(20, 1 / (1 - homophily_ratio + 1e-9))
            else:
                max_edge_ratio = float(self.densification_config["max_edge_ratio"])

            min_edge_ratio = max(1 + homophily_ratio, 3 - 2 * self.densification_config["threshold"]) # 3-2*x: (1 + (1-x)*2)

            sim_mat = None
            for adapt_itr in range(max_adapt_itrs):
                # ipdb.set_trace()
                edge_index, edge_weight, sim_mat = get_dense_graph(
                    all_x, 
                    dataset._data.edge_index.cpu().clone(),
                    method=self.densification_config["method"],
                    bandwidth=bandwidth, 
                    threshold=last_dense_thresh, 
                    num_nodes=num_nodes, 
                    sim_mat=sim_mat
                )
                
                new_edge_ratio = edge_index.shape[1] / n_init_edges
                is_fully_connected = (edge_index.shape[1] == max_nodes_sq)
                stat_log =  f"(edges: {new_edge_ratio:.2f}x). Homophily: {homophily_ratio:.2f}"

                if not self.densification_config["adaptive"]:
                    logger.info(f"Densification threshold fixed at {last_dense_thresh:.2f} {stat_log}")
                    break
                
                if max_edge_ratio >= new_edge_ratio >= min_edge_ratio or is_fully_connected or last_dense_thresh < min_dense_threshold:
                    logger.info(f"Final threshold: {last_dense_thresh:.2f} {stat_log}")
                    break
                    
                if new_edge_ratio > max_edge_ratio:
                    # Too dense: Increase threshold to restrict edges
                    last_dense_thresh = min(max_dense_threshold, last_dense_thresh * 1.1)
                    action = "Increasing"
                else:
                    # Too sparse: Decrease threshold to allow more edges
                    last_dense_thresh = max(min_dense_threshold, last_dense_thresh * 0.9)
                    action = "Decreasing"
                
                if last_dense_thresh >= max_dense_threshold:
                    # logger.info(f"Reached threshold limit ({min_dense_threshold} - {max_dense_threshold}). Exiting.")
                    break
            else:
                logger.info(f"Max attempts reached. Final threshold: {last_dense_thresh:.3f} {stat_log}")
        else:
            edge_weight = 1.0
            self.report_homophily_ratio = False
        
        if self.report_homophily_ratio:
            org_homophily_ratio = graph_homophily(dataset._data.edge_index.clone(), dataset._data.y, task=self.task)
            next_homophily_ratio = graph_homophily(edge_index.clone(), dataset._data.y, task=self.task)

        node_degrees = degree(edge_index[0], num_nodes, dtype=torch.float)

        test_conf_pred = []
        quantile_thresh = []

        for i in range(0, n_test, self.batch_size):

            start, end = i, min(i + self.batch_size, n_test)
            b_n_test = end - start
            b_test_scores = test_scores[start:end]

            if self.task == "reg":
                b_test_conf_pred = np.zeros((b_test_scores.shape[0], n_cls, 2), dtype=b_test_scores.dtype)
            else:
                b_test_conf_pred = np.zeros_like(b_test_scores)

            weights = self.weighting_fn(
                test_idxs = test_idxs[start:end],
                cal_idxs = cal_idxs,
                test_x = test_x[start:end],
                cal_x = cal_x,
                all_x = all_x,
                edge_index = edge_index, 
                edge_weight = edge_weight,
                node_degrees = node_degrees,
                bandwidth = bandwidth,
                eigenvectors = eigenvectors,
                sampling_scale = sampling_scale
            )

            if pre_weights is not None:
                weights *= pre_weights.reshape(1, -1)
                weights /= (weights.sum(axis=1) + 1e-9).reshape(-1, 1)

            if self.weighst_stats is not None:
                for k in self.weighst_stats.keys():
                    if len(self.weighst_stats[k]["stats"]) == 0:
                        self.weighst_stats[k]["stats"].append(
                            analyze_distribution(weights, metric_name=k, **self.weighst_stats[k]["kwargs"])
                        )

            w_tilde = np.zeros((b_n_test, n_unique))
            row_idxs = np.tile(np.arange(b_n_test)[:, None], (1, unique_scores_inv.shape[0])) # row_idxs: n_test x (n_cal+1)
            col_idxs = np.tile(unique_scores_inv[None, :], (b_n_test, 1)) # col_idxs: n_test x (n_cal+1)
            np.add.at(w_tilde, (row_idxs, col_idxs), weights)

            b_test_conf_pred, b_quantile_thresh = self.weighted_cp(
                w_tilde,
                unique_scores,
                b_test_scores,
                test_nc_scores[start:end]
            )
            
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
        outputs = outputs | {"dense_homophily_ratio": next_homophily_ratio / org_homophily_ratio} if self.report_homophily_ratio else outputs

        return outputs
