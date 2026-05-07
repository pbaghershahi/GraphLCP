import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch_geometric.utils import is_undirected, to_undirected, remove_self_loops, k_hop_subgraph, to_torch_coo_tensor
from torch_scatter import scatter
from .split_cp import SplitCP
import os
import sys
from pathlib import Path
from scipy import stats
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "covmetrics" / "src"))

import torch, random, os, ipdb, logging
from covmetrics import WSC, FSC, SSC, EOC, KMeansGrouping
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("global_logger")


# Plot styling knobs.
FS_TITLE = 20
FS_AXIS_LABELS = 20
FS_TICKS = 14
FS_LEGEND = 20
FIG_METRICS_WIDTH = 10
FIG_METRICS_HEIGHT = 6
FIG_COVERED_WIDTH = 8
FIG_COVERED_HEIGHT = 6
COLOR_COVERAGE = "tab:green"
COLOR_EFFICIENCY = "tab:red"
COLOR_COVERED_EFFICIENCY = "tab:purple"
COLOR_TARGET = "black"
COLOR_TARGET_TEXT = "gray"
TARGET_LEGEND_FONT_SIZE = 10



def get_k_hop_edge_index(edge_index, num_nodes, k=1):

    if k == 1:
        return edge_index

    adj = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes)).coalesce()
    accum_adj = adj.clone()
    pow_adj = adj.clone()

    for _ in range(k - 1):
        pow_adj = torch.sparse.mm(pow_adj, adj).coalesce()
        accum_adj = accum_adj + pow_adj
        
    accum_adj = accum_adj.coalesce()
    indices = accum_adj.indices()
    
    indices, _ = remove_self_loops(indices)
    
    return indices



def compute_test_node_homophily(data, test_indices, task='cls', k=1, edge_index=None):
    num_nodes = data.num_nodes
    y = data.y
    
    if edge_index is None:
        edge_index = data.edge_index

    if is_undirected(edge_index):
        base_edge_index = edge_index
    else:
        base_edge_index = to_undirected(edge_index)
        
    base_edge_index, _ = remove_self_loops(base_edge_index)
    
    # Get k-hop connectivity
    edge_index = get_k_hop_edge_index(base_edge_index, num_nodes, k)
    row, col = edge_index

    if task == 'cls':
        # Classification: 1 if same, 0 if different
        is_same = (y[row] == y[col]).float()
        node_metric = scatter(is_same, row, dim=0, dim_size=num_nodes, reduce='mean')

    elif task == 'reg':
        # Regression: Normalized Distance
        y = y.float()
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze()
        
        # 1. Calculate absolute difference for ALL edges
        diff = torch.abs(y[row] - y[col])
        
        # 2. Find the maximum difference observed in the graph structure
        # (This avoids squashing scores if the global y_range is huge but edges are short)
        max_diff = diff.max()
        
        if max_diff == 0:
            # If max difference is 0, it means every single neighbor has the exact same value.
            # Homophily is perfect (1.0) everywhere.
            node_metric = torch.ones(num_nodes, device=y.device)
        else:
            # 3. Normalize difference relative to the worst edge in the graph
            # normalized_diff is 0 (identical) to 1 (worst observed edge)
            normalized_diff = diff / max_diff
            
            # 4. Aggregate neighbors
            mean_norm_diff = scatter(normalized_diff, row, dim=0, dim_size=num_nodes, reduce='mean')
            
            # 5. Invert: 1.0 = Perfect Homophily, 0.0 = Worst observed local heterogeneity
            node_metric = 1.0 - mean_norm_diff

    else:
        raise ValueError("task must be 'cls' or 'reg'")

    metric_np = node_metric.cpu().numpy().flatten()
    metric_np = np.nan_to_num(metric_np, nan=0.0)
    
    if isinstance(test_indices, torch.Tensor):
        test_indices = test_indices.cpu().numpy()
    
    return metric_np[test_indices]



def plot_robust_conformal_metrics(homophilies, coverages, lengths, save_path, task='cls', n_bins=10, cp_alpha=0.1):
    """
    Plots Coverage and Efficiency using Quantile Binning to ensure equal sample sizes per bin.
    Includes error bars (Standard Error) to visualize statistical significance.
    """
    
    # 1. Create DataFrame
    df = pd.DataFrame({
        'Homophily': homophilies,
        'Coverage': coverages,
        'Size': lengths 
    })
    # 2. QUANTILE BINNING (The Fix)
    # pd.qcut tries to divide data so each bin has the same number of samples.
    # duplicates='drop' is needed because many nodes might have identical homophily (e.g., 1.0)
    try:
        df['Bin_Cat'] = pd.qcut(df['Homophily'], q=n_bins, duplicates='drop')
    except ValueError:
        # Fallback if too few unique values for qcut
        print("Warning: Too few unique homophily values for qcut. Using cut.")
        df['Bin_Cat'] = pd.cut(df['Homophily'], bins=n_bins)

    # 3. Aggregation with Standard Error
    # We calculate Mean AND Standard Error (sem) to draw error bars
    grouped = df.groupby('Bin_Cat', observed=True).agg({
        'Homophily': 'mean', 
        'Coverage': ['mean', 'sem'],
        'Size': ['mean', 'sem'],
        'Bin_Cat': 'count' # Keep track of count just to be sure
    })
    
    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'Homophily_mean': 'Homophily', 'Bin_Cat_count': 'Count'})
    grouped = grouped.reset_index(drop=True)
    grouped = grouped.sort_values('Homophily')

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(FIG_METRICS_WIDTH, FIG_METRICS_HEIGHT))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # --- Left Axis: Coverage ---
    color_cov = COLOR_COVERAGE
    ax1.set_xlabel('Node Homophily (Average of Quantile)', fontsize=FS_AXIS_LABELS)
    ax1.set_ylabel('Marginal Coverage', color=color_cov, fontsize=FS_AXIS_LABELS)
    
    # Plot Mean Line
    coverage_line, = ax1.plot(grouped['Homophily'], grouped['Coverage_mean'], color=color_cov, marker='o', linewidth=2, label='Coverage')
    
    # ADD SHADED ERROR REGION (Confidence Interval)
    # Mean +/- 1.96 * SEM (approx 95% CI)
    ax1.fill_between(
        grouped['Homophily'], 
        grouped['Coverage_mean'] - 1.96 * grouped['Coverage_sem'],
        grouped['Coverage_mean'] + 1.96 * grouped['Coverage_sem'],
        color=color_cov, alpha=0.2
    )
    
    ax1.tick_params(axis='x', labelcolor="black", labelsize=FS_TICKS)
    ax1.tick_params(axis='y', labelcolor=color_cov, labelsize=FS_TICKS)
    target_coverage = 1 - cp_alpha
    target_line = ax1.axhline(y=target_coverage, color=COLOR_TARGET, linestyle='--', alpha=0.9, label='Target')
    ax1.text(
        0.02,
        target_coverage + 0.02,
        f"Target Cov. = {target_coverage:.2f}",
        transform=ax1.get_yaxis_transform(),
        fontsize=FS_LEGEND,
        color=COLOR_TARGET_TEXT,
        va="bottom",
    )
    ax1.set_ylim(0, 1.05)

    # --- Right Axis: Efficiency ---
    ax2 = ax1.twinx()  
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    color_eff = COLOR_EFFICIENCY
    y_label_right = 'Normalized Prediction Set Size' if task == 'cls' else 'Interval Width'
    
    ax2.set_ylabel(f'{y_label_right} (Lower is Better)', color=color_eff, fontsize=FS_AXIS_LABELS)
    
    # Plot Mean Line
    efficiency_line, = ax2.plot(grouped['Homophily'], grouped['Size_mean'], color=color_eff, marker='s', linestyle='-.', linewidth=2, label=y_label_right)
    
    # ADD SHADED ERROR REGION for Efficiency
    ax2.fill_between(
        grouped['Homophily'], 
        grouped['Size_mean'] - 1.96 * grouped['Size_sem'],
        grouped['Size_mean'] + 1.96 * grouped['Size_sem'],
        color=color_eff, alpha=0.2
    )
    
    ax2.tick_params(axis='y', labelcolor=color_eff, labelsize=FS_TICKS)

    # Title & Layout
    # plt.title('Conformal Prediction Metrics by Homophily (Quantile Binned)', fontsize=FS_TITLE)
    plt.grid(True, alpha=0.3)
    
    ax1.legend(handles=[target_line], loc='lower right', fontsize=TARGET_LEGEND_FONT_SIZE)
    fig.legend(
        [coverage_line, efficiency_line],
        ['Coverage', y_label_right],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=FS_LEGEND,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()
    


def plot_efficiency_conditioned_on_coverage(homophilies, coverages, lengths, save_path, task='cls', n_bins=10, cp_alpha=0.1):
    """
    Plots the Prediction Set Size (Efficiency) vs. Homophily ONLY for samples 
    where the true label was successfully covered.
    
    Args:
        homophilies (np.array): Homophily scores for test nodes.
        coverages (np.array): Boolean/Float (0/1) indicating if true label was in set.
        lengths (np.array): Size of prediction set or interval width.
        save_path (str): File path to save the plot.
        cp_alpha (float): The miscoverage rate used (e.g., 0.1).
        task (str): 'cls' or 'reg' for labeling.
        n_bins (int): Number of quantile bins.
    """
    
    # 1. FILTERING: Select only samples where Coverage == 1 (True)
    mask = (coverages == 1)

    # Safety check: if no samples are covered, we can't plot
    if np.sum(mask) == 0:
        print(f"Error: No samples have valid coverage at alpha={cp_alpha}. Cannot plot efficiency.")
        return

    filtered_hom = homophilies[mask]
    filtered_len = lengths[mask]
    
    print(f"Plotting using {len(filtered_hom)} covered samples (out of {len(homophilies)} total).")

    # 2. Create DataFrame with filtered data
    df = pd.DataFrame({
        'Homophily': filtered_hom,
        'Size': filtered_len
    })
    
    # 3. ROBUST BINNING (Quantile)
    # Using qcut ensures each point on the x-axis represents roughly the same amount of data
    try:
        df['Bin_Cat'] = pd.qcut(df['Homophily'], q=n_bins, duplicates='drop')
    except ValueError:
        print("Warning: Too few unique homophily values for qcut. Using cut.")
        df['Bin_Cat'] = pd.cut(df['Homophily'], bins=n_bins)

    # 4. Aggregation
    grouped = df.groupby('Bin_Cat', observed=True).agg({
        'Homophily': 'mean', 
        'Size': ['mean', 'sem'], # Mean and Standard Error
        'Bin_Cat': 'count'
    })
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'Homophily_mean': 'Homophily', 'Bin_Cat_count': 'Count'})
    grouped = grouped.reset_index(drop=True)
    grouped = grouped.sort_values('Homophily')

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(FIG_COVERED_WIDTH, FIG_COVERED_HEIGHT))
    
    ax.tick_params(axis='x', labelcolor="black", labelsize=FS_TICKS)
    ax.tick_params(axis='y', labelcolor="black", labelsize=FS_TICKS)

    # Color choice: Purple indicates "Efficiency of Success"
    color = COLOR_COVERED_EFFICIENCY
    
    # Labeling
    y_label = 'Normalized Prediction Set Size' if task == 'cls' else 'Interval Width'
    ax.set_xlabel('Node Homophily', fontsize=FS_AXIS_LABELS)
    ax.set_ylabel(f'{y_label} (Conditioned on Coverage)', fontsize=FS_AXIS_LABELS, color="black")
    
    # Plot Mean Line
    line_label = 'Mean Normalized Set Size' if task == 'cls' else 'Mean Interval Width'
    ax.plot(grouped['Homophily'], grouped['Size_mean'], color=color, marker='D', linewidth=2, label=line_label)
    
    # Add Shaded Confidence Interval (Mean +/- 1.96 * SEM)
    ax.fill_between(
        grouped['Homophily'], 
        grouped['Size_mean'] - 1.96 * grouped['Size_sem'],
        grouped['Size_mean'] + 1.96 * grouped['Size_sem'],
        color=color, alpha=0.2, label='95% Conf. Int.'
    )

    # Title & Grid
    target_cov = (1 - cp_alpha) * 100
    # plt.title(f'Efficiency for Valid Predictions Only ({task.upper()})\nTarget Coverage: {target_cov:.0f}% ($alpha={cp_alpha}$)', fontsize=FS_TITLE)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=FS_LEGEND)
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    plt.close()



class TestCP(SplitCP):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


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
        num_runs = kwargs.pop("num_runs", 10)

        if all_outs is not None:
            for k, v in all_outs.items():
                if isinstance(v, torch.Tensor):
                    if k == "labels":
                        all_outs[k] = v.to(torch.int64)
                    else:
                        all_outs[k] = v.to(torch.float32)
                    all_outs[k] = all_outs[k].cpu().numpy()
                elif isinstance(v, np.ndarray):
                    if k == "labels":
                        all_outs[k] = v.astype(np.int64)
                    else:
                        all_outs[k] = v.astype(np.float32)

        preds = {}
        all_homophilies = []
        all_coverages = []
        all_preds = []
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
            test_predictions = cp_out_dict["test_predictions"]
            quantile_thresholds = cp_out_dict["quantile_thresholds"]
            # preds["predictions"] = test_predictions
            # ipdb.set_trace()
            test_x, test_labels = [all_outs[k][test_idxs.numpy()] for k in ["x", "labels"]]
            coverage_per_sample, length_per_sample, coverage, pred_length = self.compute_metrics(
                test_predictions,
                test_labels,
                test_x,
                partitions = (dataset.partition_methods, dataset._data.partitions[test_idxs]) if hasattr(dataset._data, "partitions") else None,
                eval_config = self.eval_config,
            )

            if self.task == "cls":
                length_per_sample = length_per_sample / dataset.n_classes

            homophilies = compute_test_node_homophily(dataset._data, test_idxs, task=self.task, k=1)
            all_homophilies.append(homophilies)
            all_coverages.append(coverage_per_sample)
            all_preds.append(length_per_sample)

            for k, v in coverage.items():
                out_key = f"{k}_coverage"
                if out_key not in preds:
                    preds[f"{k}_coverage"] = [v]
                    preds[f"{k}_pred_length"] = [pred_length[k]]
                else:
                    preds[f"{k}_coverage"].append(v)
                    preds[f"{k}_pred_length"].append(pred_length[k])

        all_homophilies = np.concatenate(all_homophilies)
        all_coverages = np.concatenate(all_coverages)
        all_preds = np.concatenate(all_preds)

        os.makedirs(save_dir, exist_ok=True)

        plot_robust_conformal_metrics(
            homophilies=all_homophilies,
            coverages=all_coverages,
            lengths=all_preds,
            save_path=os.path.join(save_dir, f'cp_homophily_{dataset.name}.pdf'),
            task=self.task,
            n_bins=20,
            cp_alpha=self.alpha
        )

        plot_efficiency_conditioned_on_coverage(
            homophilies=all_homophilies,
            coverages=all_coverages,
            lengths=all_preds,
            save_path=os.path.join(save_dir, f'cp_homophily_covered_{dataset.name}.pdf'),
            task=self.task,
            n_bins=20,
            cp_alpha=self.alpha
        )

        save_payload = {
            'name': dataset.name,
            'task': self.task,
            'alpha': self.alpha,
            'n_classes': dataset.n_classes if self.task == "cls" else None,
            'preds_normalized': self.task == "cls",
            'homophily': all_homophilies,
            'coverage': all_coverages,
            'preds': all_preds
        }
        
        save_path = os.path.join(save_dir, f"{dataset.name}_results.pt")
        torch.save(save_payload, save_path)
        print(f"Saved results for {dataset.name} to {save_path}")

        preds = {k: np.array(v).mean(axis=0).astype(np.float32) for k, v in preds.items()}

        return preds
