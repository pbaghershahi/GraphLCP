"""Matched-score comparison of GraphLCP and residual-Mondrian graph CP.

This is a controlled Track-A reimplementation, not the RR-GNN authors' native
code.  A frozen base predictor supplies identical scores to all methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .base_cp import BaseCP
from .graph_rlcp import GraphRLCP


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample +infinity atom."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return math.inf
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def louvain_node_clusters(
    edge_index: torch.Tensor,
    num_nodes: int,
    resolution: float,
    seed: int,
) -> np.ndarray:
    """Compute a label-free Louvain community assignment for every node."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.detach().cpu().T.tolist())
    communities = nx.community.louvain_communities(
        graph, resolution=resolution, seed=seed
    )
    assignments = np.full(num_nodes, -1, dtype=np.int64)
    for cluster_id, nodes in enumerate(communities):
        assignments[np.fromiter(nodes, dtype=np.int64)] = cluster_id
    if np.any(assignments < 0):
        raise RuntimeError("Louvain did not assign every node to a community")
    return assignments


class ResidualScaleGNN(nn.Module):
    """Small positive-output GCN used only to predict base-score scale."""

    def __init__(self, in_channels: int, hidden_channels: int, depth: int, epsilon: float):
        super().__init__()
        if depth < 1:
            raise ValueError("Residual GNN depth must be positive")
        channels = [in_channels] + [hidden_channels] * depth
        self.layers = nn.ModuleList(
            GCNConv(channels[index], channels[index + 1]) for index in range(depth)
        )
        self.output = GCNConv(hidden_channels, 1)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return F.softplus(self.output(x, edge_index).reshape(-1)) + self.epsilon


@dataclass(frozen=True)
class SplitIndices:
    auxiliary: torch.Tensor
    calibration: torch.Tensor
    test: torch.Tensor


class MatchedRRMondrianComparison(BaseCP):
    """Run matched RR variants and GraphLCP on shared test splits."""

    def __init__(
        self,
        auxiliary_ratio: float = 0.30,
        calibration_ratio: float = 0.20,
        residual_hidden_dim: int = 64,
        residual_depth: int = 2,
        residual_epochs: int = 200,
        residual_lr: float = 1e-3,
        residual_weight_decay: float = 0.0,
        residual_epsilon: float = 1e-6,
        min_cluster_calibration: int = 20,
        louvain_resolution: float = 1.0,
        louvain_seed: int = 0,
        graphlcp_config: dict | None = None,
        base_method: str = "split",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if auxiliary_ratio <= 0 or calibration_ratio <= 0:
            raise ValueError("Auxiliary and calibration ratios must be positive")
        if auxiliary_ratio + calibration_ratio >= 1:
            raise ValueError("Auxiliary and calibration ratios must sum to less than one")
        if residual_epsilon <= 0:
            raise ValueError("residual_epsilon must be positive")
        self.auxiliary_ratio = auxiliary_ratio
        self.calibration_ratio = calibration_ratio
        self.residual_hidden_dim = residual_hidden_dim
        self.residual_depth = residual_depth
        self.residual_epochs = residual_epochs
        self.residual_lr = residual_lr
        self.residual_weight_decay = residual_weight_decay
        self.residual_epsilon = residual_epsilon
        self.min_cluster_calibration = min_cluster_calibration
        self.louvain_resolution = louvain_resolution
        self.louvain_seed = louvain_seed
        self.base_method = base_method
        self.graphlcp = GraphRLCP(
            alpha=self.alpha,
            task=self.task,
            device=self.device,
            eval_config=self.eval_config,
            base_method=base_method,
            report_homophily_ratio=False,
            weight_analysis_metrics={},
            **(graphlcp_config or {}),
        )

    def _draw_split(self, dataset) -> SplitIndices:
        # Match BaseCP's original conformal split: randomly divide all nodes
        # outside the fixed base-training mask into an original calibration
        # set of size dataset.n_cal and a final test set containing the rest.
        # Then split only that original calibration set for Residual-GNN
        # auxiliary training versus conformal calibration.
        nontraining = (~dataset._data.train_mask).nonzero().reshape(-1).cpu()
        nontraining = nontraining[torch.randperm(nontraining.numel())]
        if dataset.n_cal <= 1 or dataset.n_cal >= nontraining.numel():
            raise ValueError("Invalid original calibration/test allocation")
        original_calibration = nontraining[:dataset.n_cal]
        original_test = nontraining[dataset.n_cal:]
        allocation_total = self.auxiliary_ratio + self.calibration_ratio
        n_auxiliary = int(round(
            original_calibration.numel() * self.auxiliary_ratio / allocation_total
        ))
        n_calibration = original_calibration.numel() - n_auxiliary
        if n_auxiliary <= 0 or n_calibration <= 0:
            raise ValueError("Dataset is too small for auxiliary/calibration allocation")
        return SplitIndices(
            auxiliary=original_calibration[:n_auxiliary],
            calibration=original_calibration[n_auxiliary:],
            test=original_test,
        )

    def _true_base_scores(self, all_outs: dict, indices: torch.Tensor) -> np.ndarray:
        scores = all_outs["scores"][indices.numpy()]
        labels = all_outs["labels"][indices.numpy()]
        if self.task == "reg":
            return np.abs(labels - scores).reshape(-1)
        return 1.0 - scores[np.arange(indices.numel()), labels]

    def _train_residual_model(
        self,
        dataset,
        all_outs: dict,
        auxiliary_indices: torch.Tensor,
    ) -> tuple[np.ndarray, int]:
        device = torch.device(self.device)
        x = dataset._data.x.to(device=device, dtype=torch.float32)
        edge_index = dataset._data.edge_index.to(device)
        targets = torch.as_tensor(
            self._true_base_scores(all_outs, auxiliary_indices),
            device=device,
            dtype=torch.float32,
        )
        model = ResidualScaleGNN(
            x.shape[1],
            self.residual_hidden_dim,
            self.residual_depth,
            self.residual_epsilon,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.residual_lr,
            weight_decay=self.residual_weight_decay,
        )
        auxiliary_indices = auxiliary_indices.to(device)
        for _ in range(self.residual_epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(x, edge_index)
            loss = F.l1_loss(predictions[auxiliary_indices], targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scales = model(x, edge_index).detach().cpu().numpy()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        return scales, parameters

    def _prediction_sets(
        self,
        all_outs: dict,
        calibration_indices: torch.Tensor,
        test_indices: torch.Tensor,
        scales: np.ndarray,
        clusters: np.ndarray | None,
        use_residual: bool,
    ) -> tuple[np.ndarray, float]:
        calibration_scores = self._true_base_scores(all_outs, calibration_indices)
        calibration_scales = scales[calibration_indices.numpy()]
        normalized_scores = (
            calibration_scores / calibration_scales if use_residual else calibration_scores
        )
        global_quantile = finite_sample_quantile(normalized_scores, self.alpha)

        test_nodes = test_indices.numpy()
        test_quantiles = np.full(test_nodes.size, global_quantile, dtype=float)
        fallback = np.zeros(test_nodes.size, dtype=bool)
        if clusters is not None:
            calibration_nodes = calibration_indices.numpy()
            for cluster in np.unique(clusters[test_nodes]):
                calibration_mask = clusters[calibration_nodes] == cluster
                test_mask = clusters[test_nodes] == cluster
                if calibration_mask.sum() < self.min_cluster_calibration:
                    fallback[test_mask] = True
                    continue
                test_quantiles[test_mask] = finite_sample_quantile(
                    normalized_scores[calibration_mask], self.alpha
                )

        test_scores = all_outs["scores"][test_nodes]
        test_scales = scales[test_nodes] if use_residual else np.ones(test_nodes.size)
        if self.task == "reg":
            radii = test_quantiles * test_scales
            predictions = np.stack(
                [test_scores - radii[:, None], test_scores + radii[:, None]], axis=-1
            )
        else:
            candidate_scores = 1.0 - test_scores
            predictions = candidate_scores / test_scales[:, None] <= test_quantiles[:, None]
        return predictions, float(fallback.mean())

    def _record_metrics(
        self,
        collected: dict[str, list],
        prefix: str,
        predictions: np.ndarray,
        test_indices: torch.Tensor,
        all_outs: dict,
        dataset,
    ) -> None:
        test_nodes = test_indices.numpy()
        coverage, sizes = self.compute_metrics(
            predictions,
            all_outs["labels"][test_nodes],
            all_outs["x"][test_nodes],
            partitions=(
                dataset.partition_methods,
                dataset._data.partitions[test_indices],
            )
            if hasattr(dataset._data, "partitions")
            else None,
            eval_config=self.eval_config,
        )
        for metric, value in coverage.items():
            collected.setdefault(f"{prefix}_{metric}_coverage", []).append(value)
            collected.setdefault(f"{prefix}_{metric}_pred_length", []).append(
                sizes[metric]
            )

    def predict(
        self,
        cal_outs,
        test_outs,
        dataset,
        all_outs=None,
        pre_weights=None,
        pca_config=None,
        num_runs=10,
        **kwargs,
    ):
        del cal_outs, test_outs, kwargs
        if all_outs is None:
            raise ValueError("all_outs must be provided")
        if pre_weights is not None:
            raise ValueError("Matched comparison does not support importance weights")
        converted = {}
        for key, value in all_outs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                value = value.astype(np.int64 if key == "labels" else np.float32)
            converted[key] = value
        all_outs = converted
        clusters = louvain_node_clusters(
            dataset._data.edge_index,
            dataset.num_nsamples,
            self.louvain_resolution,
            self.louvain_seed,
        )
        collected: dict[str, list] = {}
        for _ in range(num_runs):
            split = self._draw_split(dataset)
            scales, parameter_count = self._train_residual_model(
                dataset, all_outs, split.auxiliary
            )
            full_calibration = torch.cat([split.auxiliary, split.calibration])
            unit_scales = np.ones(dataset.num_nsamples, dtype=np.float32)

            predictions, fallback = self._prediction_sets(
                all_outs,
                full_calibration,
                split.test,
                unit_scales,
                clusters,
                False,
            )
            self._record_metrics(
                collected, "mondrian_only", predictions, split.test, all_outs, dataset
            )
            collected.setdefault("mondrian_only_fallback_rate", []).append(fallback)

            predictions, fallback = self._prediction_sets(
                all_outs,
                split.calibration,
                split.test,
                scales,
                None,
                True,
            )
            self._record_metrics(
                collected, "residual_only", predictions, split.test, all_outs, dataset
            )
            collected.setdefault("residual_only_fallback_rate", []).append(fallback)

            predictions, fallback = self._prediction_sets(
                all_outs,
                split.calibration,
                split.test,
                scales,
                clusters,
                True,
            )
            self._record_metrics(
                collected, "rr_mondrian", predictions, split.test, all_outs, dataset
            )
            collected.setdefault("rr_mondrian_fallback_rate", []).append(fallback)

            graph_output = self.graphlcp.graph_rlcp(
                all_outs=all_outs,
                cal_idxs=full_calibration,
                test_idxs=split.test,
                pre_weights=pre_weights,
                pca_config=pca_config,
                dataset=dataset,
            )
            self._record_metrics(
                collected,
                "graphlcp",
                graph_output["test_predictions"],
                split.test,
                all_outs,
                dataset,
            )

            true_test_scores = self._true_base_scores(all_outs, split.test)
            predicted_test_scales = scales[split.test.numpy()]
            if np.std(true_test_scores) > 0 and np.std(predicted_test_scales) > 0:
                correlation = np.corrcoef(true_test_scores, predicted_test_scales)[0, 1]
            else:
                correlation = 0.0
            collected.setdefault("residual_scale_correlation", []).append(correlation)
            collected.setdefault("residual_scale_mae", []).append(
                np.abs(true_test_scores - predicted_test_scales).mean()
            )
            collected.setdefault("residual_model_parameters", []).append(parameter_count)
            collected.setdefault("num_louvain_clusters", []).append(
                np.unique(clusters).size
            )
            split_counts = {
                "num_base_train_nodes": dataset.train_idxs.numel(),
                "num_original_calibration_nodes": (
                    split.auxiliary.numel() + split.calibration.numel()
                ),
                "num_auxiliary_nodes": split.auxiliary.numel(),
                "num_calibration_nodes": split.calibration.numel(),
                "num_test_nodes": split.test.numel(),
            }
            for name, count in split_counts.items():
                collected.setdefault(name, []).append(count)
                collected.setdefault(name.replace("num_", "fraction_"), []).append(
                    count / dataset.num_nsamples
                )

        return {
            key: np.asarray(values).mean(axis=0).astype(np.float32)
            for key, values in collected.items()
        }
