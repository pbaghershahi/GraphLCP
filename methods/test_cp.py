import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch_geometric.utils import to_undirected, remove_self_loops, k_hop_subgraph, to_torch_coo_tensor
from torch_scatter import scatter
from .split_cp import SplitCP
import os
import sys
from pathlib import Path
from scipy import stats
import matplotlib.ticker as ticker
from .utils import cosine_similarity, get_rbf_similarity, graph_homophily

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "covmetrics" / "src"))

import torch, random, os, ipdb, logging
from covmetrics import WSC, FSC, SSC, EOC, KMeansGrouping
import torch.nn.functional as F
import numpy as np
from .plot_homophily_scp import compute_test_node_homophily
from torch_geometric.utils import homophily, subgraph

logger = logging.getLogger("global_logger")



def get_dense_graph(x, edge_index, method="rbf", bandwidth=1.0, threshold=0.1, num_nodes=None, sim_mat=None):

    if method == "cosine":
        sim_mat = cosine_similarity(torch.from_numpy(x))
    else:
        sim_mat = get_rbf_similarity(x, bandwidth)
        sim_mat = torch.from_numpy(sim_mat)

    if num_nodes is None:
        num_nodes = x.shape[0]

    sim_mat.fill_diagonal_(0.0)

    sim_mask = sim_mat > threshold

    sim_edge_index = sim_mask.nonzero().T

    sim_mat[edge_index[0, :], edge_index[1, :]] = 1.0
        
    mask = sim_mat > threshold

    edge_index = mask.nonzero().T
    edge_weight = sim_mat[edge_index[0, :], edge_index[1, :]]
    
    return edge_index, edge_weight, sim_mat, sim_edge_index



class TestCP(SplitCP):

    def __init__(
        self,
        bandwidth = 0.1,
        densification_config = dict(
            threshold = 1.0,
            method = "cosine",
            adaptive = False,
            max_edge_ratio = None
        ),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.densification_config = densification_config
        self.bandwidth = bandwidth


    def compute_metrics(
        self, 
        test_predictions, 
        test_labels, 
        test_x, 
        partitions=None, 
        eval_config=None, 
    ):

        n_test = test_predictions.shape[0]
        eval_method = "marginal" if eval_config is None else eval_config.get("method", "marginal")

        if self.task == "reg":
            lower_bounds = test_predictions[:, :, 0]
            upper_bounds = test_predictions[:, :, 1]

            coverage_per_sample = (test_labels >= lower_bounds) & (test_labels <= upper_bounds)
            length_per_sample = upper_bounds - lower_bounds
            length_per_sample[length_per_sample < 0] = 0.0 

            coverage_per_sample = coverage_per_sample.reshape(-1)
            length_per_sample = length_per_sample.reshape(-1)

        else:
            coverage_per_sample = test_predictions[np.arange(n_test), test_labels]
            length_per_sample = test_predictions.sum(axis=-1)
        
        coverages = {}
        pred_lengths = {}

        if isinstance(eval_method, (str, set)):
            eval_method = list(eval_method) if isinstance(eval_method, set) else [eval_method]

        methods_to_run = ["marginal"] + [m for m in eval_method if m != "marginal"]

        for eval_method in methods_to_run:
            if eval_method in ["marginal", "wsc"]:
                if eval_method == "marginal":
                    mask = np.ones(len(coverage_per_sample), dtype=bool)
                else:
                    mask = WSC(seed=1234).evaluate(test_x, coverage_per_sample, unbiased=False, 
                                                M=eval_config.get("wsc_num_vectors", 1000))
                
                cov, pl = self._compute_metrics(mask, coverage_per_sample, length_per_sample)
                coverages[eval_method], pred_lengths[eval_method] = cov, pl

            elif eval_method in ["ssc", "eoc"]:
                n_partitions = dataset.n_partitions if hasattr(dataset, "n_partitions") else 2
                metric = SSC() if eval_method == "ssc" else EOC()
                
                if eval_method == "ssc":
                    assert self.task != "reg", "SSC is not supported for regression tasks."
                    target_data = length_per_sample
                else:
                    target_data = test_labels.reshape(-1)

                cov, pl = metric.evaluate_FSC(target_data, coverage_per_sample, number_max_groups=n_partitions)
                coverages[eval_method], pred_lengths[eval_method] = cov, pl

            elif eval_method == "fsc":
                assert partitions is not None, "Partitions must be provided for FSC evaluation."
                partition_methods, partition_mat = partitions

                for i, p_method in enumerate(partition_methods):
                    c_list, l_list = [], []
                    n_partitions = partition_mat[:, i].max() + 1
                    for p_idx in range(n_partitions):
                        mask = partition_mat[:, i] == p_idx
                        cov, pl = self._compute_metrics(mask, coverage_per_sample, length_per_sample)
                        c_list.append(cov)
                        l_list.append(pl)
                    
                    coverages[f"fsc_{p_method}"] = np.array(c_list)
                    pred_lengths[f"fsc_{p_method}"] = np.array(l_list)

            else:
                raise ValueError(f"Unsupported method: {eval_method}")

        return coverage_per_sample, length_per_sample, coverages, pred_lengths


    def densify(
        self,
        all_outs,
        cal_idxs,
        dataset,
        pca_config,
    ):

        assert dataset is not None, "dataset must be provided for kernel_type='ppr'"
        num_nodes = dataset.num_nsamples
        all_x = all_outs["x"]

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

        edge_index = dataset._data.edge_index.clone()
        
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
                edge_index, edge_weight, sim_mat, new_edge_index = get_dense_graph(
                    all_x, 
                    dataset._data.edge_index.cpu(),
                    method=self.densification_config["method"],
                    bandwidth=bandwidth, 
                    threshold=last_dense_thresh, 
                    num_nodes=num_nodes, 
                    sim_mat=sim_mat
                )
                
                new_edge_ratio = edge_index.shape[1] / n_init_edges
                is_fully_connected = (edge_index.shape[1] == max_nodes_sq)
                stat_log = f"(edges: {new_edge_ratio:.2f}x). Homophily: {homophily_ratio:.2f}"
                # print(max_edge_ratio, min_edge_ratio, stat_log)

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

                # print(f"Edges: {new_edge_ratio:.2f}x. {action} threshold to {last_dense_thresh:.3f} and retrying.")
                
                if last_dense_thresh >= max_dense_threshold:
                    # logger.info(f"Reached threshold limit ({min_dense_threshold} - {max_dense_threshold}). Exiting.")
                    break
            else:
                logger.info(f"Max attempts reached. Final threshold: {last_dense_thresh:.3f} {stat_log}")
        else:
            edge_weight = 1.0

        return edge_index, new_edge_index


    def predict(
        self,
        cal_outs,
        test_outs,
        dataset,
        all_outs = None,
        pre_weights = None,
        pca_config = None,
        save_dir = "/workspace/cp_files",
        **kwargs
    ):

        if all_outs is not None:
            for k, v in all_outs.items():
                if isinstance(v, torch.Tensor):
                    if k == "labels":
                        all_outs[k] = v.to(torch.int64)
                    else:
                        all_outs[k] = v.to(torch.float32)
                    all_outs[k] = all_outs[k].numpy()
                elif isinstance(v, np.ndarray):
                    if k == "labels":
                        all_outs[k] = v.astype(np.int64)
                    else:
                        all_outs[k] = v.astype(np.float32)

        preds = {}
        all_homophilies = []
        all_edge_homophily = []
        all_org_edge_homophily = []
        all_new_edge_homophily = []
        all_coverages = []
        all_preds = []

        for i in range(1):
            cal_test_idxs = (~dataset._data.train_mask).nonzero().squeeze().cpu()
            perm = torch.randperm(cal_test_idxs.shape[0])
            cal_test_idxs = cal_test_idxs[perm]
            cal_idxs = cal_test_idxs[:dataset.n_cal]
            test_idxs = cal_test_idxs[dataset.n_cal:]

            test_predictions, quantile_thresholds = self.cp_method(
                all_outs = all_outs,
                cal_idxs = cal_idxs,
                test_idxs = test_idxs,
                pre_weights = pre_weights,
                pca_config = pca_config,
                dataset = dataset,
                **kwargs,
            )

            edge_index, new_edge_index = self.densify(all_outs, cal_idxs, dataset, pca_config)
            all_idxs = np.arange(dataset._data.num_nodes)

            org_edge_homophily = compute_test_node_homophily(dataset._data, all_idxs, task=self.task, k=1, edge_index=dataset._data.edge_index)
            edge_homophily = compute_test_node_homophily(dataset._data, all_idxs, task=self.task, k=1, edge_index=edge_index)
            new_edge_homophily = compute_test_node_homophily(dataset._data, all_idxs, task=self.task, k=1, edge_index=new_edge_index)

            test_x, test_labels = [all_outs[k][test_idxs.numpy()] for k in ["x", "labels"]]
            coverage_per_sample, length_per_sample, coverage, pred_length = self.compute_metrics(
                test_predictions,
                test_labels,
                test_x,
                partitions = (dataset.partition_methods, dataset._data.partitions[test_idxs]) if hasattr(dataset._data, "partitions") else None,
                eval_config = self.eval_config,
            )

            all_coverages.append(coverage_per_sample)
            all_preds.append(length_per_sample)
            all_org_edge_homophily.append(org_edge_homophily)
            all_edge_homophily.append(edge_homophily)
            all_new_edge_homophily.append(new_edge_homophily)

            for k, v in coverage.items():
                out_key = f"{k}_coverage"
                if out_key not in preds:
                    preds[f"{k}_coverage"] = [v]
                    preds[f"{k}_pred_length"] = [pred_length[k]]
                else:
                    preds[f"{k}_coverage"].append(v)
                    preds[f"{k}_pred_length"].append(pred_length[k])

        all_coverages = np.concatenate(all_coverages)
        all_preds = np.concatenate(all_preds)
        all_org_edge_homophily = np.concatenate(all_org_edge_homophily)
        all_edge_homophily = np.concatenate(all_edge_homophily)
        all_new_edge_homophily = np.concatenate(all_new_edge_homophily)

        preds = {k: np.array(v).mean(axis=0).astype(np.float32) for k, v in preds.items()}
        preds |= {
            "org_edge_homophily": all_org_edge_homophily.mean(),
            "edge_homophily": all_edge_homophily.mean(),
            "new_edge_homophily": all_new_edge_homophily.mean(),
        }

        return preds
