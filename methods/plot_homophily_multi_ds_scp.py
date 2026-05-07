import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import argparse

# Plot styling knobs.
FS_TITLE = 20
FS_AXIS_LABELS = 20
FS_TICKS = 14
FS_LEGEND = 20
FIG_WIDTH = 13
FIG_HEIGHT = 6
DATASET_COLORS = [
    "tab:red",
    "tab:green",
    "tab:cyan",
    "tab:purple",
    "tab:blue",
    "tab:pink",
    "tab:olive",
    "tab:orange",
]
COLOR_TARGET = "black"
COLOR_TARGET_TEXT = "gray"
TARGET_LEGEND_FONT_SIZE = 12


def get_prediction_values(data, task_type):
    values = np.asarray(data['preds'])
    if task_type == 'cls' and not data.get('preds_normalized', False):
        n_classes = data.get('n_classes')
        if n_classes is not None:
            values = values / n_classes
    return values

def process_data_for_plotting(homophilies, values, n_bins=20):
    """
    Helper function to bin data using quantiles and return aggregated means/sems.
    """
    df = pd.DataFrame({'Homophily': homophilies, 'Value': values})
    
    # Quantile binning
    try:
        df['Bin_Cat'] = pd.qcut(df['Homophily'], q=n_bins, duplicates='drop')
    except ValueError:
        # Fallback if distribution is too sparse
        df['Bin_Cat'] = pd.cut(df['Homophily'], bins=n_bins)

    # Aggregate
    grouped = df.groupby('Bin_Cat', observed=True).agg({
        'Homophily': 'mean',
        'Value': ['mean', 'sem']
    })
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'Homophily_mean': 'Homophily'})
    grouped = grouped.sort_values('Homophily')
    
    return grouped

def plot_task_figure(datasets, target_dataset_name, task_type, save_path, plot_all_efficiency=False):
    """
    Generates a 1x2 figure for a specific task type (CLS or REG).
    Ax1: Coverage of ALL datasets.
    Ax2: Efficiency of the TARGET dataset, or all datasets when plot_all_efficiency=True.
    """
    if not datasets:
        print(f"No datasets found for task: {task_type}. Skipping...")
        return

    # Setup Figure (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Define Labels based on task
    if task_type == 'cls':
        title_cov = "Marginal Coverage"
        title_eff = "Normalized Prediction Set Size" if plot_all_efficiency else f"Normalized Prediction Set Size ({target_dataset_name})"
        ylabel_eff = "Normalized Set Size"
        color_eff = 'tab:red'
        marker_eff = 's'
    else:
        title_cov = "Marginal Coverage"
        title_eff = "Interval Width" if plot_all_efficiency else f"Interval Width ({target_dataset_name})"
        ylabel_eff = "Interval Width"
        color_eff = 'tab:red'
        marker_eff = 'D'

    # --- AXIS 1: Coverage (All Datasets) ---
    ax = axes[0]
    ax.set_title(title_cov, fontsize=FS_TITLE)
    ax.set_ylabel("Coverage", fontsize=FS_AXIS_LABELS, color='black')
    ax.set_xlabel("Node Homophily", fontsize=FS_AXIS_LABELS)
    
    colors = [DATASET_COLORS[i % len(DATASET_COLORS)] for i in range(len(datasets))]
    dataset_handles = []
    dataset_labels = []
    
    for idx, data in enumerate(datasets):
        df = process_data_for_plotting(data['homophily'], data['coverage'])
        
        # Plot Line
        line, = ax.plot(df['Homophily'], df['Value_mean'], marker='o', markersize=4, 
                        linewidth=2, label=data['name'], color=colors[idx])
        dataset_handles.append(line)
        dataset_labels.append(data['name'])
        
        # Shaded Error
        ax.fill_between(df['Homophily'], 
                        df['Value_mean'] - 1.96 * df['Value_sem'],
                        df['Value_mean'] + 1.96 * df['Value_sem'],
                        color=colors[idx], alpha=0.1)
    
    # Target Line
    target_cov = 1.0 - datasets[0]['alpha']
    target_line = ax.axhline(y=target_cov, color=COLOR_TARGET, linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    ax.text(
        0.02,
        target_cov + 0.02,
        f"Target Cov. = {target_cov:.2f}",
        transform=ax.get_yaxis_transform(),
        fontsize=FS_LEGEND,
        color=COLOR_TARGET_TEXT,
        va="bottom",
    )

    ax.set_ylim(0.0, 1.05)
    ax.legend(handles=[target_line], fontsize=TARGET_LEGEND_FONT_SIZE, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', colors='black', labelsize=FS_TICKS)


    # --- AXIS 2: Efficiency (Target Dataset Only) ---
    ax = axes[1]
    if plot_all_efficiency:
        ax.set_title(title_eff, fontsize=FS_TITLE)
        ax.set_ylabel(ylabel_eff, fontsize=FS_AXIS_LABELS, color='black')
        ax.set_xlabel("Node Homophily", fontsize=FS_AXIS_LABELS)

        for idx, data in enumerate(datasets):
            df = process_data_for_plotting(data['homophily'], get_prediction_values(data, task_type))
            ax.plot(df['Homophily'], df['Value_mean'], marker=marker_eff, markersize=4,
                    linewidth=2, color=colors[idx], label=data['name'])
            ax.fill_between(df['Homophily'],
                            df['Value_mean'] - 1.96 * df['Value_sem'],
                            df['Value_mean'] + 1.96 * df['Value_sem'],
                            color=colors[idx], alpha=0.1)

    else:
        target_data = next((d for d in datasets if d['name'] == target_dataset_name), None)
        
        if target_data:
            ax.set_title(title_eff, fontsize=FS_TITLE)
            ax.set_ylabel(ylabel_eff, fontsize=FS_AXIS_LABELS, color='black')
            ax.set_xlabel("Node Homophily", fontsize=FS_AXIS_LABELS)
            
            df = process_data_for_plotting(target_data['homophily'], get_prediction_values(target_data, task_type))
            
            ax.plot(df['Homophily'], df['Value_mean'], marker=marker_eff, markersize=5, 
                    linewidth=2, color=color_eff, label=ylabel_eff)
            
            ax.fill_between(df['Homophily'], 
                            df['Value_mean'] - 1.96 * df['Value_sem'],
                            df['Value_mean'] + 1.96 * df['Value_sem'],
                            color=color_eff, alpha=0.2)
            
        else:
            ax.text(0.5, 0.5, f"Dataset '{target_dataset_name}'\nnot found", 
                    ha='center', va='center', fontsize=FS_TITLE)

    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', colors='black', labelsize=FS_TICKS)

    fig.legend(
        dataset_handles,
        dataset_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(4, len(dataset_labels)),
        fontsize=FS_LEGEND,
    )

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

def main(data_dir, cls_target, reg_target, output_dir=".", plot_all_efficiency=False):
    # 1. Load Data
    file_pattern = os.path.join(data_dir, "*_results.pt")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No result files found in {data_dir}")
        return

    cls_datasets = []
    reg_datasets = []
    
    # Read files and segregate by task
    for f in files:
        data = torch.load(f, weights_only=False)
        if data['task'] == 'cls':
            cls_datasets.append(data)
        elif data['task'] == 'reg':
            reg_datasets.append(data)
    
    print(f"Found {len(cls_datasets)} CLS datasets and {len(reg_datasets)} REG datasets.")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Generate CLS Figure (1x2)
    plot_task_figure(
        datasets=cls_datasets, 
        target_dataset_name=cls_target, 
        task_type='cls', 
        save_path=os.path.join(output_dir, "homophily_analysis_CLS.pdf"),
        plot_all_efficiency=plot_all_efficiency,
    )

    # 3. Generate REG Figure (1x2)
    plot_task_figure(
        datasets=reg_datasets, 
        target_dataset_name=reg_target, 
        task_type='reg', 
        save_path=os.path.join(output_dir, "homophily_analysis_REG.pdf"),
        plot_all_efficiency=plot_all_efficiency,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SCP homophily analysis across datasets.")
    parser.add_argument("--data-dir", default="/workspace/cp_files")
    parser.add_argument("--cls-target", default="PubMed")
    parser.add_argument("--reg-target", default="Anaheim")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--plot-all-efficiency", action="store_true")
    args = parser.parse_args()
    
    main(args.data_dir, args.cls_target, args.reg_target, args.output_dir, args.plot_all_efficiency)
