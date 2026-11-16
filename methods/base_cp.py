import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "covmetrics" / "src"))

import torch, random, os, ipdb, logging
from covmetrics import WSC, FSC, SSC, EOC, KMeansGrouping
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("global_logger")



class BaseCP():

    def __init__(
        self,
        alpha = 0.05,
        task = "cls",
        device = "cpu",
        eval_config = {"method": "marginal"},
        **kwargs
    ):
    
        self.alpha = alpha
        self.task = task
        self.device = device
        self.eval_config = eval_config

        if self.task == "cls":
            self.pred_len_func = getattr(np, "mean", None)
        else:
            self.pred_len_func = getattr(np, "median", None)


    def _compute_metrics(self, mask, coverage_per_sample, length_per_sample):
        coverage = coverage_per_sample[mask].mean()
        pred_len = self.pred_len_func(length_per_sample[mask])

        if pred_len == np.inf:
            finite_vals = length_per_sample[mask][np.isfinite(length_per_sample[mask])]
            pred_len = 1.5 * finite_vals.max() if len(finite_vals) > 0 else 1e6
                
        return coverage, pred_len


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

        return coverages, pred_lengths



    def predict(
        self,
        cal_outs,
        test_outs,
        dataset,
        all_outs = None,
        pre_weights = None,
        pca_config = None,
        num_runs = 10,
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

        preds = {"dense_homophily_ratio": []}
        for i in range(num_runs):
            cal_test_idxs = (~dataset._data.train_mask).nonzero().squeeze().cpu()
            perm = torch.randperm(cal_test_idxs.shape[0])
            cal_test_idxs = cal_test_idxs[perm]
            cal_idxs = cal_test_idxs[:dataset.n_cal]
            test_idxs = cal_test_idxs[dataset.n_cal:]

            cp_out_dict = self.cp_method(
                all_outs = all_outs,
                cal_idxs = cal_idxs,
                test_idxs = test_idxs,
                pre_weights = pre_weights,
                pca_config = pca_config,
                dataset = dataset,
                **kwargs,
            )
            test_predictions, quantile_thresholds = cp_out_dict["test_predictions"], cp_out_dict["quantile_thresholds"]

            test_x, test_labels = [all_outs[k][test_idxs.numpy()] for k in ["x", "labels"]]
            coverage, pred_length = self.compute_metrics(
                test_predictions,
                test_labels,
                test_x,
                partitions = (dataset.partition_methods, dataset._data.partitions[test_idxs]) if hasattr(dataset._data, "partitions") else None,
                eval_config = self.eval_config,
            )

            for k, v in coverage.items():
                out_key = f"{k}_coverage"
                if out_key not in preds:
                    preds[f"{k}_coverage"] = [v]
                    preds[f"{k}_pred_length"] = [pred_length[k]]
                else:
                    preds[f"{k}_coverage"].append(v)
                    preds[f"{k}_pred_length"].append(pred_length[k])

            if "dense_homophily_ratio" in cp_out_dict:
                preds["dense_homophily_ratio"].append(cp_out_dict["dense_homophily_ratio"])

        preds = {k: np.array(v).mean(axis=0).astype(np.float32) for k, v in preds.items() if len(v) > 0}

        return preds