"""GraphLCP with a consistent uniform--PPR mixture kernel.

The mixture parameter is used in both stages of randomized localized
conformal prediction: sampling the randomized anchor and weighting the
calibration/test observations.  This implementation is intentionally isolated
from the legacy GraphRLCP implementation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.utils import is_undirected, subgraph

from metrics import analyze_distribution

from .aps import aps_transform
from .graph_rlcp import GraphRLCP
from .utils import compute_ppr_weights, get_dense_graph, graph_homophily


logger = logging.getLogger("global_logger")


def weighted_node_degrees(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | float,
    num_nodes: int,
) -> torch.Tensor:
    """Return source degrees for the weighted random-walk transition matrix."""
    device = edge_index.device
    if isinstance(edge_weight, torch.Tensor):
        weights = edge_weight.to(device=device, dtype=torch.float32)
    else:
        weights = torch.full(
            (edge_index.shape[1],), float(edge_weight), device=device
        )
    degrees = torch.zeros(num_nodes, dtype=weights.dtype, device=device)
    degrees.scatter_add_(0, edge_index[0], weights)
    return degrees


def mixed_reverse_ppr_weights(
    anchor_ppr: torch.Tensor,
    anchor_indices: torch.Tensor,
    target_indices: torch.Tensor,
    node_degrees: torch.Tensor,
    mixture_lambda: float,
    num_nodes: int,
) -> torch.Tensor:
    """Compute normalized H_lambda(target, anchor) weights.

    ``anchor_ppr[b, i]`` is pi_{anchor_b}(i), and ``target_indices[b]``
    contains the calibration nodes followed by the test node for row ``b``.
    For an undirected weighted graph, reversibility gives
    pi_target(anchor) = d_anchor / d_target * pi_anchor(target).
    """
    if not 0.0 <= mixture_lambda <= 1.0:
        raise ValueError("mixture_lambda must lie in [0, 1]")
    if anchor_ppr.ndim != 2 or target_indices.ndim != 2:
        raise ValueError("anchor_ppr and target_indices must be matrices")
    if anchor_ppr.shape[0] != target_indices.shape[0]:
        raise ValueError("anchor and target batch dimensions do not match")

    device = anchor_ppr.device
    targets = target_indices.to(device)
    anchors = anchor_indices.to(device)
    degrees = node_degrees.to(device=device, dtype=anchor_ppr.dtype)

    forward = torch.gather(anchor_ppr, 1, targets)
    anchor_degree = degrees[anchors].unsqueeze(1)
    target_degree = degrees[targets]
    reverse = torch.where(
        target_degree > 0,
        forward * anchor_degree / target_degree.clamp_min(torch.finfo(forward.dtype).tiny),
        torch.zeros_like(forward),
    )
    unnormalized = (1.0 - mixture_lambda) / float(num_nodes) + mixture_lambda * reverse
    unnormalized = torch.nan_to_num(unnormalized, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = unnormalized.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("Mixed-kernel weights have a non-positive row sum")
    return unnormalized / row_sums


def mixed_direct_ppr_weights(
    source_ppr: torch.Tensor,
    anchor_indices: torch.Tensor,
    num_calibration: int,
    mixture_lambda: float,
    num_nodes: int,
) -> torch.Tensor:
    """Compute H_lambda(source, anchor) directly for a directed graph."""
    if not 0.0 <= mixture_lambda <= 1.0:
        raise ValueError("mixture_lambda must lie in [0, 1]")
    batch_size = source_ppr.shape[0] - num_calibration
    if batch_size != anchor_indices.numel():
        raise ValueError("Expected one test-centered PPR row per anchor")
    anchors = anchor_indices.to(source_ppr.device)
    calibration_values = source_ppr[:num_calibration, anchors].T
    test_rows = torch.arange(batch_size, device=source_ppr.device)
    test_values = source_ppr[num_calibration:][test_rows, anchors].unsqueeze(1)
    direct = torch.cat([calibration_values, test_values], dim=1)
    unnormalized = (1.0 - mixture_lambda) / float(num_nodes) + mixture_lambda * direct
    unnormalized = torch.nan_to_num(unnormalized, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = unnormalized.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("Mixed-kernel weights have a non-positive row sum")
    return unnormalized / row_sums


def sample_mixed_anchors_from_ppr(
    test_ppr: torch.Tensor,
    mixture_lambda: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coupled uniform/PPR anchors from a row-normalized PPR matrix.

    Both component anchors and the Bernoulli uniforms are drawn for every
    lambda, including endpoints.  Reusing the same generator seed in separate
    lambda runs therefore supplies common random numbers.
    """
    if not 0.0 <= mixture_lambda <= 1.0:
        raise ValueError("mixture_lambda must lie in [0, 1]")
    probabilities = test_ppr.detach().to(device="cpu", dtype=torch.float64)
    probabilities = torch.clamp(probabilities, min=0.0)
    row_sums = probabilities.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("PPR sampling distribution has a non-positive row sum")
    probabilities /= row_sums

    n_test, num_nodes = probabilities.shape
    uniforms = torch.rand(n_test, generator=generator)
    uniform_anchors = torch.randint(num_nodes, (n_test,), generator=generator)
    ppr_anchors = torch.multinomial(
        probabilities, 1, replacement=True, generator=generator
    ).squeeze(1)
    anchors = torch.where(uniforms <= mixture_lambda, ppr_anchors, uniform_anchors)
    return anchors, uniforms <= mixture_lambda


class GraphUniformLCP(GraphRLCP):
    """GraphLCP using H_lambda=(1-lambda)Uniform + lambda*PPR."""

    def __init__(
        self,
        mixture_lambda: float = 1.0,
        mixture_lambdas: list[float] | None = None,
        anchor_seed: int = 0,
        **kwargs,
    ) -> None:
        if not 0.0 <= mixture_lambda <= 1.0:
            raise ValueError("mixture_lambda must lie in [0, 1]")
        # The parent's Gaussian/PPR convex branch is not used.  Calling its
        # constructor retains the weighted-CP implementation and public API.
        super().__init__(
            weighting_method="convex",
            convex_dist_coeff=1.0,
            **kwargs,
        )
        self.mixture_lambda = float(mixture_lambda)
        self.mixture_lambdas = (
            [float(value) for value in mixture_lambdas]
            if mixture_lambdas is not None
            else [self.mixture_lambda]
        )
        if not self.mixture_lambdas or any(
            not 0.0 <= value <= 1.0 for value in self.mixture_lambdas
        ):
            raise ValueError("Every mixture lambda must lie in [0, 1]")
        self.anchor_seed = int(anchor_seed)
        self._anchor_generator = torch.Generator(device="cpu")
        self._anchor_generator.manual_seed(self.anchor_seed)
        self._dense_cache = None
        self._ppr_cache = {}
        self.cp_method = self.graph_uniform_lcp

    def _get_dense_artifacts(self, all_x, cal_idxs, dataset, bandwidth):
        """Build densification once for the current conformal split."""
        if self._dense_cache is not None:
            return self._dense_cache
        num_nodes = dataset.num_nsamples
        n_initial_edges = dataset._data.edge_index.shape[1]
        if self.densification_config["threshold"] < 1.0:
            last_threshold = self.densification_config["threshold"]
            maximum_edges = num_nodes * (num_nodes - 1)
            min_threshold, max_threshold = 0.01, 0.99
            calibration_edges, _ = subgraph(
                subset=cal_idxs,
                edge_index=dataset._data.edge_index.cpu(),
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            homophily_ratio = graph_homophily(
                calibration_edges, dataset._data.y[cal_idxs], task=self.task
            )
            if self.densification_config["max_edge_ratio"] is None:
                max_edge_ratio = 2 * min(20, 1 / (1 - homophily_ratio + 1e-9))
            else:
                max_edge_ratio = float(self.densification_config["max_edge_ratio"])
            min_edge_ratio = max(
                1 + homophily_ratio,
                3 - 2 * self.densification_config["threshold"],
            )
            similarity_matrix = None
            for _ in range(10):
                edge_index, edge_weight, similarity_matrix = get_dense_graph(
                    all_x,
                    dataset._data.edge_index.cpu().clone(),
                    method=self.densification_config["method"],
                    bandwidth=bandwidth,
                    threshold=last_threshold,
                    num_nodes=num_nodes,
                    sim_mat=similarity_matrix,
                )
                edge_ratio = edge_index.shape[1] / n_initial_edges
                fully_connected = edge_index.shape[1] == maximum_edges
                if (
                    not self.densification_config["adaptive"]
                    or max_edge_ratio >= edge_ratio >= min_edge_ratio
                    or fully_connected
                    or last_threshold < min_threshold
                ):
                    break
                if edge_ratio > max_edge_ratio:
                    last_threshold = min(max_threshold, last_threshold * 1.1)
                else:
                    last_threshold = max(min_threshold, last_threshold * 0.9)
                if last_threshold >= max_threshold:
                    break
        else:
            edge_index = dataset._data.edge_index.cpu().clone()
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        node_degrees = weighted_node_degrees(edge_index, edge_weight, num_nodes)
        homophily_ratio_output = None
        if self.report_homophily_ratio:
            original_homophily = graph_homophily(
                dataset._data.edge_index.clone(), dataset._data.y, task=self.task
            )
            dense_homophily = graph_homophily(
                edge_index.clone(), dataset._data.y, task=self.task
            )
            homophily_ratio_output = dense_homophily / original_homophily
        self._dense_cache = (
            edge_index,
            edge_weight,
            node_degrees,
            homophily_ratio_output,
        )
        return self._dense_cache

    def _compute_consistent_mixed_weights(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | float,
        node_degrees: torch.Tensor,
        test_indices: torch.Tensor,
        calibration_indices: torch.Tensor,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample anchors and compute weights from the same weighted PPR kernel."""
        device = edge_index.device
        test_indices = test_indices.to(device)
        calibration_indices = calibration_indices.to(device)

        cache_key = tuple(test_indices.detach().cpu().tolist())
        cached = self._ppr_cache.get(cache_key)
        if cached is None:
            test_ppr = compute_ppr_weights(
                edge_index,
                edge_weight,
                num_nodes,
                test_indices,
                alpha=self.ppr_alpha,
                k=self.ppr_k,
            )
            probabilities = test_ppr.detach().to(device="cpu", dtype=torch.float64)
            probabilities = torch.clamp(probabilities, min=0.0)
            probabilities /= probabilities.sum(dim=1, keepdim=True)
            cached = {
                "test_ppr": test_ppr,
                "uniforms": torch.rand(test_indices.numel(), generator=self._anchor_generator),
                "uniform_anchors": torch.randint(
                    num_nodes, (test_indices.numel(),), generator=self._anchor_generator
                ),
                "ppr_anchors": torch.multinomial(
                    probabilities, 1, replacement=True,
                    generator=self._anchor_generator,
                ).squeeze(1),
                "anchor_ppr": {},
                "source_ppr": None,
            }
            self._ppr_cache[cache_key] = cached
        used_ppr_cpu = cached["uniforms"] <= self.mixture_lambda
        anchor_indices_cpu = torch.where(
            used_ppr_cpu, cached["ppr_anchors"], cached["uniform_anchors"]
        )
        anchor_indices = anchor_indices_cpu.to(device)

        if is_undirected(edge_index):
            if self.mixture_lambda == 0.0:
                anchor_ppr = torch.zeros(
                    (anchor_indices.numel(), num_nodes),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                anchor_cache = cached["anchor_ppr"]
                if not anchor_cache:
                    grid_anchors = []
                    for value in self.mixture_lambdas:
                        if value == 0.0:
                            continue
                        grid_anchors.append(
                            torch.where(
                                cached["uniforms"] <= value,
                                cached["ppr_anchors"],
                                cached["uniform_anchors"],
                            )
                        )
                    unique_anchors = torch.unique(torch.cat(grid_anchors)).tolist()
                    unique_tensor = torch.tensor(
                        unique_anchors, device=device, dtype=torch.long
                    )
                    unique_ppr = compute_ppr_weights(
                        edge_index, edge_weight, num_nodes, unique_tensor,
                        alpha=self.ppr_alpha, k=self.ppr_k,
                    )
                    for node, row in zip(unique_anchors, unique_ppr):
                        anchor_cache[node] = row
                anchor_ppr = torch.stack(
                    [anchor_cache[node] for node in anchor_indices_cpu.tolist()]
                )
            targets = torch.cat(
                [
                    calibration_indices.unsqueeze(0).expand(test_indices.shape[0], -1),
                    test_indices.unsqueeze(1),
                ],
                dim=1,
            )
            weights = mixed_reverse_ppr_weights(
                anchor_ppr, anchor_indices, targets, node_degrees,
                self.mixture_lambda, num_nodes,
            )
        else:
            if cached["source_ppr"] is None:
                calibration_ppr = compute_ppr_weights(
                    edge_index, edge_weight, num_nodes, calibration_indices,
                    alpha=self.ppr_alpha, k=self.ppr_k,
                )
                cached["source_ppr"] = torch.cat(
                    [calibration_ppr, cached["test_ppr"]], dim=0
                )
            weights = mixed_direct_ppr_weights(
                cached["source_ppr"], anchor_indices, calibration_indices.numel(),
                self.mixture_lambda, num_nodes,
            )
        return (
            weights.detach().cpu().numpy().astype(np.float32),
            anchor_indices_cpu.numpy(),
            used_ppr_cpu.numpy(),
        )

    @staticmethod
    def _lambda_key(value: float) -> str:
        return f"lambda_{value:.2f}".replace(".", "p")

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
        """Evaluate the entire lambda grid while sharing split-local artifacts."""
        del cal_outs, test_outs, kwargs
        if all_outs is None:
            raise ValueError("all_outs must be provided")
        converted = {}
        for key, value in all_outs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                value = value.astype(np.int64 if key == "labels" else np.float32)
            converted[key] = value
        all_outs = converted

        collected: dict[str, list] = {}
        for run_index in range(num_runs):
            cal_test_idxs = (~dataset._data.train_mask).nonzero().squeeze().cpu()
            cal_test_idxs = cal_test_idxs[torch.randperm(cal_test_idxs.shape[0])]
            cal_idxs = cal_test_idxs[:dataset.n_cal]
            test_idxs = cal_test_idxs[dataset.n_cal:]

            self._dense_cache = None
            self._ppr_cache = {}
            self._anchor_generator.manual_seed(self.anchor_seed + run_index)
            test_nodes = test_idxs.numpy()
            test_x = all_outs["x"][test_nodes]
            test_labels = all_outs["labels"][test_nodes]
            partitions = (
                (dataset.partition_methods, dataset._data.partitions[test_idxs])
                if hasattr(dataset._data, "partitions")
                else None
            )

            for mixture_lambda in self.mixture_lambdas:
                self.mixture_lambda = mixture_lambda
                output = self.graph_uniform_lcp(
                    all_outs=all_outs,
                    cal_idxs=cal_idxs,
                    test_idxs=test_idxs,
                    pre_weights=pre_weights,
                    pca_config=pca_config,
                    dataset=dataset,
                )
                coverage, pred_length = self.compute_metrics(
                    output["test_predictions"],
                    test_labels,
                    test_x,
                    partitions=partitions,
                    eval_config=self.eval_config,
                )
                prefix = self._lambda_key(mixture_lambda)
                for metric, value in coverage.items():
                    collected.setdefault(f"{prefix}_{metric}_coverage", []).append(value)
                    collected.setdefault(f"{prefix}_{metric}_pred_length", []).append(
                        pred_length[metric]
                    )
                collected.setdefault(f"{prefix}_effective_sample_size", []).append(
                    output["effective_sample_size"].mean()
                )
                collected.setdefault(f"{prefix}_ppr_anchor_fraction", []).append(
                    output["used_ppr_component"].mean()
                )
                if "dense_homophily_ratio" in output:
                    collected.setdefault("dense_homophily_ratio", []).append(
                        output["dense_homophily_ratio"]
                    )

        return {
            key: np.asarray(values).mean(axis=0).astype(np.float32)
            for key, values in collected.items()
        }

    def graph_uniform_lcp(
        self,
        all_outs,
        cal_idxs,
        test_idxs,
        pre_weights=None,
        pca_config=None,
        n_quantiles=20,
        dataset=None,
    ):
        """Construct mixed-kernel prediction sets for one calibration split."""
        del n_quantiles  # Retained for interface compatibility.
        if dataset is None:
            raise ValueError("dataset must be provided")

        num_nodes = dataset.num_nsamples
        all_x = all_outs["x"]
        test_scores, test_x = [
            all_outs[key][test_idxs.numpy()] for key in ["scores", "x"]
        ]
        cal_scores, cal_x, cal_labels = [
            all_outs[key][cal_idxs.numpy()] for key in ["scores", "x", "labels"]
        ]
        n_test = test_scores.shape[0]
        n_cal, n_classes = cal_scores.shape

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
            raise ValueError(
                f"Unsupported task ({self.task}) and base method "
                f"({self.base_method}) combination"
            )

        score_order = np.argsort(cal_nc_scores)
        sorted_cal_indices = cal_idxs[torch.as_tensor(score_order, device=cal_idxs.device)]
        sorted_cal_scores = cal_nc_scores[score_order]
        cal_x = cal_x[score_order]
        nc_scores = np.concatenate((sorted_cal_scores, [np.inf]))
        unique_scores, unique_scores_inverse = np.unique(
            nc_scores, return_inverse=True
        )

        bandwidth = self.bandwidth
        if pca_config is not None and pca_config["with_dynamic_bandwidth"]:
            eigenvalues = pca_config["pca"].explained_variance_
            bandwidth = self.bandwidth**2 * (eigenvalues + 1e-8)

        edge_index, edge_weight, node_degrees, dense_homophily_ratio = (
            self._get_dense_artifacts(all_x, cal_idxs, dataset, bandwidth)
        )

        if pre_weights is not None:
            pre_weights = np.concatenate(
                (pre_weights[score_order], np.ones((1,), dtype=test_nc_scores.dtype))
            )

        predictions, thresholds = [], []
        sampled_anchors, used_ppr_components = [], []
        effective_sample_sizes = []
        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            batch_scores = test_scores[start:end]
            weights, anchors, used_ppr = self._compute_consistent_mixed_weights(
                edge_index.to(self.device),
                edge_weight.to(self.device)
                if isinstance(edge_weight, torch.Tensor)
                else edge_weight,
                node_degrees.to(self.device),
                test_idxs[start:end],
                sorted_cal_indices,
                num_nodes,
            )
            if pre_weights is not None:
                weights *= pre_weights.reshape(1, -1)
                weights /= weights.sum(axis=1, keepdims=True) + 1e-12

            effective_sample_sizes.append(1.0 / np.square(weights).sum(axis=1))
            sampled_anchors.append(anchors)
            used_ppr_components.append(used_ppr)

            if self.weighst_stats is not None:
                for key in self.weighst_stats:
                    if len(self.weighst_stats[key]["stats"]) == 0:
                        self.weighst_stats[key]["stats"].append(
                            analyze_distribution(
                                weights,
                                metric_name=key,
                                **self.weighst_stats[key]["kwargs"],
                            )
                        )

            n_unique = unique_scores.shape[0]
            batch_size = end - start
            aggregated_weights = np.zeros((batch_size, n_unique))
            rows = np.tile(
                np.arange(batch_size)[:, None],
                (1, unique_scores_inverse.shape[0]),
            )
            columns = np.tile(unique_scores_inverse[None, :], (batch_size, 1))
            np.add.at(aggregated_weights, (rows, columns), weights)
            batch_predictions, batch_thresholds = self.weighted_cp(
                aggregated_weights,
                unique_scores,
                batch_scores,
                test_nc_scores[start:end],
            )
            predictions.append(batch_predictions)
            thresholds.append(batch_thresholds)

        outputs = {
            "test_predictions": np.concatenate(predictions, axis=0),
            "quantile_thresholds": np.concatenate(thresholds, axis=0),
            "effective_sample_size": np.concatenate(effective_sample_sizes),
            "sampled_anchors": np.concatenate(sampled_anchors),
            "used_ppr_component": np.concatenate(used_ppr_components),
        }
        if dense_homophily_ratio is not None:
            outputs["dense_homophily_ratio"] = dense_homophily_ratio
        return outputs
