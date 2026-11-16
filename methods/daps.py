import logging
import numpy as np
import torch

from .base_cp import BaseCP
from .aps import aps_transform
from .utils import compute_daps_scores

logger = logging.getLogger("global_logger")


class DAPS(BaseCP):
    """
    DAPS with iterative Score Propagation (SP), Section 5.2.

    Let H be the initial node-wise nonconformity score matrix and
    P = D^{-1} A. Starting with H^(0) = H, run:

        H^(t) = (1 - lambda) H + lambda P H^(t-1)

    for t = 1, ..., k.

    Important:
      - k=1 recovers the original 1-hop DAPS update exactly:
            H^(1) = (1-lambda)H + lambda D^{-1}AH
      - k>1 is the SP/PPR-style DAPS generalization.
      - In the k -> infinity limit:
            H_hat = (1-lambda)(I - lambda P)^(-1) H

    base_method:
      "aps"   : canonical DAPS/SP-DAPS using the project's aps_transform.
      "split" : simple score variant:
                  classification -> 1 - p(y|x)
                  regression     -> absolute residual.
    """

    def __init__(
        self,
        base_method="aps",
        diffusion_lambda=0.5,
        k=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not 0.0 <= diffusion_lambda < 1.0:
            raise ValueError("diffusion_lambda must be in [0, 1).")
        if k < 1:
            raise ValueError("k must be >= 1.")

        self.base_method = base_method
        self.diffusion_lambda = float(diffusion_lambda)
        self.k = int(k)

        if base_method == "aps":
            assert self.task == "cls", (
                "APS-based DAPS only supports classification."
            )
            self.cp_method = self.daps_aps
        elif base_method == "split":
            self.cp_method = self.daps_split
        else:
            raise ValueError(f"Unknown base_method: {base_method}")

    @staticmethod
    def _to_numpy_idxs(idxs):
        if isinstance(idxs, torch.Tensor):
            return idxs.detach().cpu().numpy()
        return np.asarray(idxs)

    def _propagate_scores(self, nc_scores, dataset):
        return compute_daps_scores(
            edge_index=dataset._data.edge_index.clone(),
            edge_weights=None,  # Paper baseline uses the original unweighted graph.
            num_nodes=dataset.num_nsamples,
            scores=torch.from_numpy(np.asarray(nc_scores)).to(torch.float32),
            diffusion_lambda=self.diffusion_lambda,
            k=self.k,
        ).cpu().numpy()

    def _calibrate_and_predict(
        self,
        cal_nc_scores,
        test_nc_scores,
        test_scores,
        pre_weights=None,
    ):
        """
        Same split-conformal calibration logic as split_cp.py / aps.py.
        """
        n_cal = cal_nc_scores.shape[0]
        n_test = test_scores.shape[0]
        n_cls = test_scores.shape[1]

        if pre_weights is None:
            pre_weights = np.ones_like(cal_nc_scores) / n_cal
        else:
            pre_weights = pre_weights / pre_weights.sum()

        ord_idxs = np.argsort(cal_nc_scores)
        sorted_nc_scores = cal_nc_scores[ord_idxs]

        quantile_level = (1 - self.alpha) * (1 + 1 / n_cal)
        cumulative_probs = np.cumsum(pre_weights[ord_idxs])
        quantile_idx = np.min(
            np.where(cumulative_probs >= quantile_level)
        )
        quantile_thresh = sorted_nc_scores[quantile_idx]

        if self.task == "reg":
            test_conf_pred = np.zeros((n_test, n_cls, 2))
            test_conf_pred[:, 0] = np.concatenate(
                (
                    test_scores - quantile_thresh.reshape(-1, 1),
                    test_scores + quantile_thresh.reshape(-1, 1),
                ),
                axis=1,
            )
        else:
            test_conf_pred = test_nc_scores <= quantile_thresh

        return {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }

    def daps_aps(
        self,
        all_outs,
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights=None,
        **kwargs,
    ):
        """
        APS + iterative SP/PPR-DAPS.

        APS scores are computed for ALL nodes/classes before propagation.
        """
        assert self.task == "cls", (
            "APS-based DAPS only supports classification."
        )

        cal_idxs_np = self._to_numpy_idxs(cal_idxs)
        test_idxs_np = self._to_numpy_idxs(test_idxs)

        # Reuse the project's existing APS implementation exactly.
        all_nc_scores = aps_transform(all_outs["scores"])

        # k=1 -> original DAPS.
        # k>1 -> score-propagation / PPR-DAPS.
        propagated_scores = self._propagate_scores(
            all_nc_scores,
            dataset,
        )

        cal_labels = np.asarray(
            all_outs["labels"][cal_idxs_np]
        ).astype(np.int64)

        cal_nc_scores = propagated_scores[
            cal_idxs_np,
            cal_labels,
        ]
        test_nc_scores = propagated_scores[test_idxs_np]
        test_scores = all_outs["scores"][test_idxs_np]

        return self._calibrate_and_predict(
            cal_nc_scores=cal_nc_scores,
            test_nc_scores=test_nc_scores,
            test_scores=test_scores,
            pre_weights=pre_weights,
        )

    def daps_split(
        self,
        all_outs,
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights=None,
        **kwargs,
    ):
        """
        Simple nonconformity score + iterative SP/PPR-DAPS.

        Classification:
            H[v,y] = 1 - p_v(y)

        Regression:
            H[v] = |y_v - f(x_v)|

        The regression path is a natural score-propagation extension;
        the published DAPS experiments primarily target classification.
        """
        cal_idxs_np = self._to_numpy_idxs(cal_idxs)
        test_idxs_np = self._to_numpy_idxs(test_idxs)

        if self.task == "reg":
            all_nc_scores = np.abs(
                all_outs["labels"] - all_outs["scores"]
            ).reshape(-1, 1)
        else:
            all_nc_scores = 1 - all_outs["scores"]

        propagated_scores = self._propagate_scores(
            all_nc_scores,
            dataset,
        )

        if self.task == "reg":
            cal_nc_scores = propagated_scores[
                cal_idxs_np
            ].reshape(-1)
            test_nc_scores = None
        else:
            cal_labels = np.asarray(
                all_outs["labels"][cal_idxs_np]
            ).astype(np.int64)
            cal_nc_scores = propagated_scores[
                cal_idxs_np,
                cal_labels,
            ]
            test_nc_scores = propagated_scores[test_idxs_np]

        test_scores = all_outs["scores"][test_idxs_np]

        return self._calibrate_and_predict(
            cal_nc_scores=cal_nc_scores,
            test_nc_scores=test_nc_scores,
            test_scores=test_scores,
            pre_weights=pre_weights,
        )
