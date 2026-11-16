"""Sweep vanilla SplitCP miscoverage for comparison with reported GraphLCP."""

from __future__ import annotations

import math

import numpy as np
import torch
from covmetrics import WSC

from .aps import aps_transform
from .base_cp import BaseCP


class SCPAlphaSweep(BaseCP):
    """SCP-only alpha sweep; GraphLCP is not executed by this method."""

    def __init__(
        self,
        scp_alpha_grid: list[float] | None = None,
        base_method: str = "split",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not math.isclose(self.alpha, 0.1):
            raise ValueError("The SCP sweep must use alpha values at or below 0.1")
        grid = scp_alpha_grid or [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        self.scp_alpha_grid = sorted({float(value) for value in grid}, reverse=True)
        if any(value <= 0 or value > 0.1 for value in self.scp_alpha_grid):
            raise ValueError("Every SCP alpha must lie in (0, 0.1]")
        self.base_method = base_method

    @staticmethod
    def _alpha_key(alpha: float) -> str:
        return f"alpha_{alpha:.3f}".replace(".", "p")

    def _scores(self, all_outs: dict, cal_idxs: torch.Tensor, test_idxs: torch.Tensor):
        cal_nodes, test_nodes = cal_idxs.numpy(), test_idxs.numpy()
        cal_scores = all_outs["scores"][cal_nodes]
        test_scores = all_outs["scores"][test_nodes]
        cal_labels = all_outs["labels"][cal_nodes]
        if self.task == "reg":
            calibration = np.abs(cal_labels - cal_scores).reshape(-1)
            candidates = np.zeros_like(test_scores)
        elif self.base_method == "split":
            calibration = 1.0 - cal_scores[np.arange(cal_nodes.size), cal_labels]
            candidates = 1.0 - test_scores
        elif self.base_method == "aps":
            transformed_cal = aps_transform(cal_scores)
            calibration = transformed_cal[np.arange(cal_nodes.size), cal_labels]
            candidates = aps_transform(test_scores)
        else:
            raise ValueError(f"Unsupported base method: {self.base_method}")
        return np.sort(calibration), candidates, test_scores

    @staticmethod
    def _finite_sample_threshold(sorted_scores: np.ndarray, alpha: float) -> float:
        rank = int(math.ceil((sorted_scores.size + 1) * (1.0 - alpha)))
        return math.inf if rank > sorted_scores.size else float(sorted_scores[rank - 1])

    def _scp_predictions(self, threshold, candidate_scores, test_scores):
        if self.task == "reg":
            predictions = np.zeros((test_scores.shape[0], test_scores.shape[1], 2))
            predictions[:, :, 0] = test_scores - threshold
            predictions[:, :, 1] = test_scores + threshold
            return predictions
        return candidate_scores <= threshold

    def _sample_metrics(self, predictions, labels):
        if self.task == "reg":
            lower, upper = predictions[:, :, 0], predictions[:, :, 1]
            coverage = ((labels >= lower) & (labels <= upper)).reshape(-1)
            sizes = (upper - lower).reshape(-1)
        else:
            coverage = predictions[np.arange(labels.size), labels].astype(bool)
            sizes = predictions.sum(axis=1).astype(float)
        return coverage, sizes

    def _fixed_wsc_metrics(self, test_x, coverage, sizes):
        mask = WSC(seed=1234).evaluate(
            test_x,
            coverage,
            unbiased=False,
            M=self.eval_config.get("wsc_num_vectors", 1000),
        )
        return float(coverage[mask].mean()), float(sizes[mask].mean())

    def _evaluate_predictions(self, predictions, labels, test_x):
        coverage, sizes = self._sample_metrics(predictions, labels)
        wsc, wsc_size = self._fixed_wsc_metrics(test_x, coverage, sizes)
        return {
            "marginal_coverage": float(coverage.mean()),
            "marginal_size": float(sizes.mean()),
            "wsc": wsc,
            "wsc_size": wsc_size,
        }

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
        converted = {}
        for key, value in all_outs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                value = value.astype(np.int64 if key == "labels" else np.float32)
            converted[key] = value
        all_outs = converted

        collected: dict[str, list] = {}
        for _ in range(num_runs):
            available = (~dataset._data.train_mask).nonzero().squeeze().cpu()
            available = available[torch.randperm(available.numel())]
            cal_idxs = available[:dataset.n_cal]
            test_idxs = available[dataset.n_cal:]
            test_nodes = test_idxs.numpy()
            labels = all_outs["labels"][test_nodes]
            test_x = all_outs["x"][test_nodes]

            sorted_scores, candidate_scores, test_scores = self._scores(
                all_outs, cal_idxs, test_idxs
            )
            sweep = []
            baseline_size = None
            previous_predictions = None
            nesting_violations = 0
            marginal_monotonicity_violations = 0
            wsc_monotonicity_violations = 0
            previous_metrics = None
            for alpha in self.scp_alpha_grid:
                threshold = self._finite_sample_threshold(sorted_scores, alpha)
                predictions = self._scp_predictions(
                    threshold, candidate_scores, test_scores
                )
                metrics = self._evaluate_predictions(predictions, labels, test_x)
                if math.isclose(alpha, 0.1):
                    baseline_size = metrics["marginal_size"]
                if previous_predictions is not None:
                    if self.task == "reg":
                        nested = (
                            np.all(predictions[:, :, 0] <= previous_predictions[:, :, 0])
                            and np.all(predictions[:, :, 1] >= previous_predictions[:, :, 1])
                        )
                    else:
                        nested = np.all(predictions | ~previous_predictions)
                    nesting_violations += int(not nested)
                previous_predictions = predictions
                if previous_metrics is not None:
                    marginal_monotonicity_violations += int(
                        metrics["marginal_coverage"] + 1e-12
                        < previous_metrics["marginal_coverage"]
                    )
                    wsc_monotonicity_violations += int(
                        metrics["wsc"] + 1e-12 < previous_metrics["wsc"]
                    )
                previous_metrics = metrics
                sweep.append((alpha, threshold, metrics))

            if baseline_size is None:
                raise ValueError("SCP alpha grid must include 0.1")
            for alpha, threshold, metrics in sweep:
                prefix = f"scp_{self._alpha_key(alpha)}"
                for metric, value in metrics.items():
                    collected.setdefault(f"{prefix}_{metric}", []).append(value)
                collected.setdefault(f"{prefix}_threshold", []).append(threshold)
                collected.setdefault(f"{prefix}_relative_size", []).append(
                    metrics["marginal_size"] / baseline_size
                )
                collected.setdefault(f"{prefix}_nominal_coverage", []).append(1.0 - alpha)

            collected.setdefault("nesting_violations", []).append(nesting_violations)
            collected.setdefault("marginal_monotonicity_violations", []).append(
                marginal_monotonicity_violations
            )
            collected.setdefault("wsc_monotonicity_violations", []).append(
                wsc_monotonicity_violations
            )

        return {
            key: np.asarray(values).mean(axis=0).astype(np.float32)
            for key, values in collected.items()
        }
