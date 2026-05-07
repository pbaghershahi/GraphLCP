import torch, random, os, ipdb, logging
import numpy as np
from .base_cp import BaseCP
from .utils import compute_guassian_weights, sample_from_guassian_kernel, sample_ppr_anchor, compute_ppr_weights
from scipy.stats import multivariate_normal
from metrics import analyze_distribution
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix, degree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D


logger = logging.getLogger("global_logger")



# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
def generate_mock_data(n=200, m=50):
    """
    Generates dummy data where weights are positively correlated 
    with non-conformity scores to demonstrate the plots.
    """
    np.random.seed(42)
    
    # Non-conformity scores (S): fixed for all test points (length m)
    S = np.random.uniform(0, 10, m)
    
    # Weights (W): n x m matrix. 
    # Constructed to have a correlation with S, plus some random noise.
    # W_ij = slope * S_j + noise
    noise = np.random.normal(0, 2, size=(n, m))
    W = 0.5 * S[np.newaxis, :] + 1.0 + noise
    
    # Normalize weights roughly to looks like standard weights (optional)
    W = np.maximum(W, 0) # ensure non-negative
    
    return S, W


# ==========================================
# METHOD 1: Conditional Distribution (Mean + Std Band)
# ==========================================
def plot_conditional_distribution(S, W, filename="method1_conditional.pdf"):
    """
    Plots the mean weight vs score with a standard deviation band.
    """
    # 1. Sort indices based on S so the line plot is continuous
    n, m = W.shape
    sort_idx = np.argsort(S)
    S_sorted = S[sort_idx]
    W_sorted = W[:, sort_idx] # Reorder columns of W
    
    # 2. Calculate statistics across test points (axis 0)
    w_mean = np.mean(W_sorted, axis=0)
    w_std = np.std(W_sorted, axis=0)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the mean line
    plt.plot(S_sorted, w_mean, color='blue', linewidth=2, label='Mean Weight across n test points')
    
    # Shade the uncertainty (Mean +/- 1 Std Dev)
    plt.fill_between(S_sorted, w_mean - w_std, w_mean + w_std, 
                     color='blue', alpha=0.2, label='$\pm$ 1 Std. Dev.')

    uniform_baseline = 1.0 / m
    # === NEW: Plot the Uniform Weight Baseline ===
    plt.axhline(y=uniform_baseline, color='red', linestyle='--', linewidth=2, 
                label=f'Uniform Weight ($1/{m}$)')
    
    plt.xlabel('Non-Conformity Score ($S_j$)')
    plt.ylabel('Weight ($W_{ij}$)')
    plt.title(f'Method 1: Weight Distribution vs. Non-Conformity\n(Aggregated over {W.shape[0]} test points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# ==========================================
# METHOD 2: 2D Hexbin / Density Plot
# ==========================================
def plot_hexbin_density(S, W, filename="method2_hexbin.pdf"):
    """
    Flattens the data and creates a Hexbin plot to handle overplotting.
    """
    n, m = W.shape
    
    # 1. Flatten W (n*m) and Tile S to match
    # We repeat S for every test point row
    S_flat = np.tile(S, n)       # shape: (n*m, )
    W_flat = W.flatten()         # shape: (n*m, )
    
    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    # gridsize controls resolution. mincnt=1 hides empty cells
    hb = plt.hexbin(S_flat, W_flat, gridsize=30, cmap='inferno', mincnt=1)
    
    cb = plt.colorbar(hb)
    cb.set_label('Count (Density)')
    
    plt.xlabel('Non-Conformity Score ($S_j$)')
    plt.ylabel('Weight ($W_{ij}$)')
    plt.title(f'Method 2: Global Density of Weights vs Scores\n(Total points: {n*m})')
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# ==========================================
# METHOD 3: Histogram of Correlations
# ==========================================
def plot_correlation_histogram(S, W, filename="method3_hist_corr.pdf"):
    """
    Calculates the correlation coefficient for each test point (row)
    and plots the histogram of these correlations.
    """
    n = W.shape[0]
    correlations = []
    
    # 1. Calculate correlation for each row
    for i in range(n):
        # Pearson correlation matrix [2x2], we need off-diagonal
        rho = np.corrcoef(S, W[i, :])[0, 1]
        correlations.append(rho)
        
    correlations = np.array(correlations)
    
    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    sns.histplot(correlations, bins=20, kde=True, color='green')
    
    # Add a vertical line at 0
    plt.axvline(0, color='red', linestyle='--', label='Zero Correlation')
    plt.axvline(np.mean(correlations), color='black', linestyle='-', label=f'Mean Corr: {np.mean(correlations):.2f}')
    
    plt.xlabel('Pearson Correlation Coefficient ($\\rho$)')
    plt.ylabel('Number of Test Points')
    plt.title('Method 3: Distribution of Correlations per Test Point')
    plt.legend()
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# ==========================================
# METHOD 4: Binned Boxplot
# ==========================================
def plot_binned_boxplot(S, W, filename="method4_boxplot.pdf", n_bins=5):
    """
    Bins the continuous scores S and plots boxplots of weights for each bin.
    """
    n, m = W.shape
    
    # 1. Prepare Data in Long Format (Pandas is best for this)
    S_flat = np.tile(S, n)
    W_flat = W.flatten()
    
    df = pd.DataFrame({
        'Score': S_flat,
        'Weight': W_flat
    })
    
    # 2. Bin the scores
    # pd.cut creates categorical bins
    df['Score Bin'] = pd.cut(df['Score'], bins=n_bins)
    
    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x='Score Bin', y='Weight', data=df, palette="Blues")
    
    plt.xlabel('Non-Conformity Score Range')
    plt.ylabel('Weight Distribution')
    plt.title('Method 4: Boxplots of Weights by Score Bins')
    plt.xticks(rotation=15) # Rotate labels if they are long
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")



def plot_total_weight_distribution(W_matrix, filename="combined_weight_distribution.pdf"):
    """
    Plots the distribution of Calibration weights and Test point weights.
    
    Parameters:
    -----------
    W_matrix : np.array of shape (n, m + 1)
        The matrix where the first m columns are weights for calibration points,
        and the LAST column is the weight for the test point itself.
    """
    n, total_cols = W_matrix.shape
    m = total_cols - 1  # Number of calibration points
    
    # 1. Separate Calibration Weights and Test Weights
    W_cal = W_matrix[:, :-1]  # Shape (n, m)
    W_test = W_matrix[:, -1]  # Shape (n, )
    
    # Flatten calibration weights for global density
    w_cal_flat = W_cal.flatten()
    
    # 2. Define Baseline (Standard CP assumption: 1/m for calibration)
    # Note: Depending on your exact formulation, standard CP might imply 
    # uniform weights of 1/(m+1). We stick to 1/m as requested previously.
    uniform_cal_baseline = 1.0 / m
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # --- Distribution A: Calibration Weights ---
    # We use stat="density" to make them comparable regardless of count differences
    sns.kdeplot(w_cal_flat, color="teal", fill=True, alpha=0.3, 
                label='Calibration Points Weights', linewidth=2)
    
    # --- Distribution B: Test Points Weights ---
    # ipdb.set_trace()
    sns.kdeplot(W_test, color="orange", fill=True, alpha=0.3, 
                label='Test Points Weights', linewidth=2)
    
    # --- Reference Line: Uniform Calibration Weight ---
    plt.axvline(x=uniform_cal_baseline, color='red', linestyle='--', linewidth=2, 
                label=f'Uniform Calib. Weight ($1/{m}$)')
    
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title(f'Distribution of Weights: Calibration vs. Test Points\n($n={n}$ test points, $m={m}$ calibration points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")



def plot_hyperparameter_evolution(experiments_dict, filename="combined_ecdf_single.pdf"):
    """
    Plots both Calibration and Test weights on a single ECDF figure.
    - Legend 1: Hyperparameters (Colors).
    - Legend 2: Data Type (Solid vs Dashed).
    """
    
    FS_TITLE = 20
    FS_AXIS_LABELS = 16
    FS_TICKS = 14
    FS_LEGEND_TITLE = 14
    FS_LEGEND_TEXT = 12
    
    data_list = []
    
    # Get dimensions for baseline
    first_key = list(experiments_dict.keys())[0]
    m = experiments_dict[first_key].shape[1] - 1
    uniform_baseline = 1.0 / m
    
    print(f"Processing data for {len(experiments_dict)} bandwidth values...")
    
    for param_val, W_matrix in experiments_dict.items():
        # Flatten Calibration (limit size for speed if needed)
        W_cal = W_matrix[:, :-1].flatten()
        if len(W_cal) > 50000:
            W_cal = np.random.choice(W_cal, 50000, replace=False)
            
        W_test = W_matrix[:, -1]
        
        # Add to list
        for w in W_cal:
            data_list.append({'Weight': w, 'Type': 'Calibration', 'bandwidth': param_val})
        for w in W_test:
            data_list.append({'Weight': w, 'Type': 'Test', 'bandwidth': param_val})
            
    df = pd.DataFrame(data_list)

    # --- 2. FORMATTING ---
    # Convert bandwidth to string for consistent categorical coloring
    df['bandwidth'] = df['bandwidth'].astype(str)
    
    # Sort keys for the legend order
    sorted_keys = sorted(experiments_dict.keys())
    hue_order_str = [str(k) for k in sorted_keys]

    # Define a fixed palette so we can manually recreate the legend later
    unique_hues = len(hue_order_str)
    palette_name = "tab10"
    # Get the actual RGB values for the colors to ensure the legend matches the plot
    palette_colors = sns.color_palette(palette_name, n_colors=unique_hues)

    # --- 3. PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    # PLOT A: Calibration Weights (Dashed Lines)
    sns.ecdfplot(
        data=df[df['Type'] == 'Calibration'],
        x='Weight',
        hue='bandwidth',
        hue_order=hue_order_str,
        palette=palette_colors,
        linestyle='--',       # <--- DASHED
        linewidth=2.5,
        # alpha=0.8,
        legend=False,         # Disable auto-legend to avoid duplication
        zorder=2              # Plot in foreground
    )

    # PLOT B: Test Weights (Solid Lines + Markers)
    sns.ecdfplot(
        data=df[df['Type'] == 'Test'],
        x='Weight',
        hue='bandwidth',
        hue_order=hue_order_str,
        palette=palette_colors,
        linestyle='-',        # <--- SOLID
        linewidth=2,
        marker='*',           # <--- MARKER
        markersize=12,
        markevery=0.1,        # Space them out (10% intervals)
        alpha=0.8,
        legend=False,         # Disable auto-legend
        zorder=1              # Plot in background
    )

    # Reference Line
    plt.axvline(x=uniform_baseline, color='black', linestyle=':', linewidth=2, 
                label=f'Uniform ($1/{m}$)')

    # --- 4. MANUAL LEGENDS ---
    
    # --- Legend 1: Hyperparameters (Colors) ---
    color_handles = []
    for i, hue_label in enumerate(hue_order_str):
        # Create a handle with the correct color
        handle = Line2D([0], [0], color=palette_colors[i], linewidth=3, label=f'{hue_label}')
        color_handles.append(handle)
        
    # Add the first legend to the plot
    # bbox_to_anchor allows us to place it outside or precisely in the corner
    legend1 = plt.legend(
        handles=color_handles, 
        title="bandwidth", 
        loc='lower right', 
        bbox_to_anchor=(0.95, 0.25), 
        framealpha=0.9,
        fontsize=FS_LEGEND_TEXT,
        title_fontsize=FS_LEGEND_TITLE
    )
    
    # Crucial: Add this legend as an 'artist' so adding the second legend doesn't erase it
    plt.gca().add_artist(legend1)

    # --- Legend 2: Data Types (Styles) ---
    style_handles = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, 
               marker='*', markersize=10, label='Test Points'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, 
               label='Calib. Points'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5, 
               label=f'Identical Baseline')
    ]
    
    # Add the second legend (stacked below the first one)
    plt.legend(
        handles=style_handles, 
        title="Distributions", 
        loc='lower right', 
        bbox_to_anchor=(0.95, 0.02), 
        framealpha=0.9,
        fontsize=FS_LEGEND_TEXT,
        title_fontsize=FS_LEGEND_TITLE
    )

    # --- 4. TITLES AND LABELS WITH FONT SIZES ---
    # plt.title('Comparison of Weights: Test vs. Calibration', fontsize=FS_TITLE)
    plt.xlabel('Weight Value', fontsize=FS_AXIS_LABELS)
    plt.ylabel('Cumulative Probability (ECDF)', fontsize=FS_AXIS_LABELS)
    
    # Change Tick Label Size
    plt.tick_params(axis='both', which='major', labelsize=FS_TICKS)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")



class TestCP(BaseCP):

    def __init__(
        self,
        bandwidth = 0.1,
        batch_size = 1024,
        weight_analysis_metrics = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.cp_method = self.rand_lcp
        self.batch_size = batch_size
        
        if len(weight_analysis_metrics.keys()) > 0:
            self.weighst_stats = {k:{"kwargs": weight_analysis_metrics[k], "stats": []} for k in weight_analysis_metrics}
        else:
            self.weighst_stats = {}
        
        self.drew_the_figure = False


    def predict(
        self,
        cal_outs,
        test_outs,
        dataset,
        all_outs = None,
        pre_weights = None,
        pca_config = None,
        save_dir = ".",
        num_runs = 1,
        **kwargs
    ):
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

        os.makedirs(save_dir, exist_ok=True)

        for _ in range(num_runs):
            cal_test_idxs = (~dataset._data.train_mask).nonzero().squeeze().cpu()
            perm = torch.randperm(cal_test_idxs.shape[0])
            cal_test_idxs = cal_test_idxs[perm]
            cal_idxs = cal_test_idxs[:dataset.n_cal]
            test_idxs = cal_test_idxs[dataset.n_cal:]

            self.cp_method(
                all_outs = all_outs,
                cal_idxs = cal_idxs,
                test_idxs = test_idxs,
                pre_weights = pre_weights,
                pca_config = pca_config,
                dataset = dataset,
                save_dir = save_dir,
                **kwargs,
            )

            if self.drew_the_figure:
                break

        return {}
    

    def _compute_guassian_weights(self, center_x, cal_x, test_x, bandwidth=1.0):
        """
        inputs:
        - center_x: embeddings of the sampled points in the proximity of test points -> (n, d)
        - cal_x: embeddings of the calibration set -> (m, d)
        - test_x: embeddings of the test points -> (n, d)

        outputs:
        - weights: normallized weights for all the calibration points and the test points -> (n, m+1)
        """
        weights = compute_guassian_weights(
            center_x, cal_x, test_x, bandwidth=bandwidth, normalize=True, eigenvectors=None
        )
        
        return weights


    def rand_lcp(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        pre_weights = None,
        pca_config = None,
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

        nc_scores = np.concatenate((cal_nc_scores, [np.inf])) # nc_scores: (n_val+1)
        unique_scores, unique_scores_inv = np.unique(nc_scores, return_inverse=True)
        n_unique = unique_scores.shape[0]

        # ipdb.set_trace()
        if pre_weights is not None:
            pre_weights = np.concatenate((pre_weights[cal_nc_idxs], np.ones((1,), dtype=test_nc_scores.dtype)))

        param_weight_dict = {}
        for bandwidth in [0.01, 0.1, 1.0, 2.0, 10.0]:
            eigenvectors = None
            sampling_scale = bandwidth
            new_bandwidth = bandwidth
            if pca_config is not None:
                pca_eigenvalues = pca_config["pca"].explained_variance_ # the eigenvalues of PCA are exactly equivalent to the variances
                pca_eigenvectors = pca_config["pca"].components_
                if pca_config["with_dynamic_bandwidth"]:
                    sampling_scale = bandwidth * np.sqrt(pca_eigenvalues)
                    new_bandwidth = bandwidth**2 * (pca_eigenvalues + 1e-8)
                    if pca_config["sample_original_space"]:
                        eigenvectors = pca_eigenvectors

            test_conf_pred = []
            quantile_thresh = []
            all_weights = []
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
                    sampling_scale = new_bandwidth,
                )

                weights = self._compute_guassian_weights(
                    b_test_tilde_x, 
                    cal_x, 
                    b_test_x, 
                    bandwidth = new_bandwidth,
                )

                # weights /= weights.sum(axis=1, keepdims=True)

                # if self.weighst_stats is not None:
                #     for k in self.weighst_stats.keys():
                #         self.weighst_stats[k]["stats"].append(
                #             analyze_distribution(weights, metric_name=k, **self.weighst_stats[k]["kwargs"])
                #         )

                if pre_weights is not None:
                    weights *= pre_weights.reshape(1, -1)
                    weights /= (weights.sum(axis=1) + 1e-9).reshape(-1, 1)

                all_weights.append(weights)            

            param_weight_dict[bandwidth] = np.concatenate(all_weights, axis=0)

        # if self.weighst_stats is not None:
        #     self.weighst_stats = {
        #         k: {"kwargs": v["kwargs"], "stats": [np.concatenate(v["stats"], axis=0)]} for k, v in self.weighst_stats.items()
        #     }

        if not self.drew_the_figure:
            ds_name = kwargs["dataset"].name
            save_dir = kwargs.get("save_dir", ".")
            # plot_conditional_distribution(cal_nc_scores, all_weights[:, :-1], filename=f"{ds_name}_method1_conditional_{self.bandwidth}.pdf")
            # plot_hexbin_density(cal_nc_scores, all_weights[:, :-1], filename=f"{ds_name}_method2_hexbin_{self.bandwidth}.pdf")
            # plot_correlation_histogram(cal_nc_scores, all_weights[:, :-1], filename=f"{ds_name}_method3_hist_corr_{self.bandwidth}.pdf")
            # plot_binned_boxplot(cal_nc_scores, all_weights[:, :-1], filename=f"{ds_name}_method4_boxplot_{self.bandwidth}.pdf")
            # plot_total_weight_distribution(all_weights, filename=f"{ds_name}_method5_distribution_{self.bandwidth}.pdf")
            plot_hyperparameter_evolution(
                param_weight_dict,
                filename=os.path.join(save_dir, f"{ds_name}_method6_params_dists.pdf"),
            )
            self.drew_the_figure = True

        return {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
