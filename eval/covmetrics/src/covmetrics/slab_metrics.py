import torch 
import numpy as np
from sklearn.model_selection import train_test_split

from covmetrics.check import *
from covmetrics.utils import seed_everything


# Original code from https://github.com/Shai128/oqr


def sample_sphere_numpy(n_samples, dim, rng):
    """Sample n_samples points uniformly on a sphere in R^dim (NumPy)."""
    vectors = rng.standard_normal((n_samples, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def sample_sphere_torch(n_samples, dim):
    """Sample n_samples points uniformly on a sphere in R^dim (PyTorch)."""
    vectors = torch.randn((n_samples, dim))
    vectors /= vectors.norm(dim=1, keepdim=True)
    return vectors

def wsc_all_numpy(x, coverage, delta, V):
    """Compute worst slab coverage (NumPy backend)."""
    n, p = x.shape
    min_points_required = int(np.ceil(delta * n))
    n_directions = V.shape[0]

    # projections: shape (n_directions, n)
    projection = (x @ V.T).T
    order = np.argsort(projection, axis=1)
    projection_sorted = np.take_along_axis(projection, order, axis=1)
    coverage_sorted = np.take_along_axis(
        np.broadcast_to(coverage, (n_directions, n)), order, axis=1
    )

    max_start_index = n - min_points_required + 1
    arange_cache = np.arange(1, n + 1)

    best_coverages = np.ones(n_directions)
    best_start = np.zeros(n_directions, dtype=int)
    best_end = np.full(n_directions, n, dtype=int)

    for start_index in range(max_start_index):
        min_end_index = min(min_points_required, n) - 1
        cumulative_sums = np.cumsum(coverage_sorted[:, start_index:], axis=1)
        denom = arange_cache[: n - start_index][None, :]
        coverage_vals = cumulative_sums / denom
        coverage_vals[:, :min_end_index] = 1.0

        relative_end_index = np.argmin(coverage_vals, axis=1)
        current_cover = coverage_vals[np.arange(n_directions), relative_end_index]

        update_mask = current_cover < best_coverages
        best_coverages[update_mask] = current_cover[update_mask]
        best_start[update_mask] = start_index
        best_end[update_mask] = start_index + relative_end_index[update_mask]

    a_vals = projection_sorted[np.arange(n_directions), best_start]
    b_vals = projection_sorted[np.arange(n_directions), best_end - 1]

    return best_coverages, a_vals, b_vals

def wsc_all_torch(x, coverage, delta, V):
    """Compute worst slab coverage (Torch backend)."""
    n, p = x.shape
    min_points_required = int(np.ceil(delta * n))
    n_directions = V.shape[0]

    # projections: shape (n_directions, n)
    projection = (x @ V.T).T
    projection_sorted, order = torch.sort(projection, dim=1)
    coverage_sorted = torch.gather(coverage.expand(n_directions, -1), 1, order)

    max_start_index = n - min_points_required + 1
    arange_cache = torch.arange(1, n + 1)

    best_coverages = torch.ones(n_directions)
    best_start = torch.zeros(n_directions, dtype=torch.long)
    best_end = torch.full((n_directions,), n, dtype=torch.long)

    for start_index in range(max_start_index):
        min_end_index = min(min_points_required, n) - 1
        cumulative_sums = torch.cumsum(coverage_sorted[:, start_index:], dim=1)
        denom = arange_cache[: n - start_index].unsqueeze(0)
        coverage_vals = cumulative_sums / denom
        coverage_vals[:, :min_end_index] = 1.0

        relative_end_index = torch.argmin(coverage_vals, dim=1)
        current_cover = coverage_vals[torch.arange(n_directions), relative_end_index]

        update_mask = current_cover < best_coverages
        best_coverages[update_mask] = current_cover[update_mask]
        best_start[update_mask] = start_index
        best_end[update_mask] = start_index + relative_end_index[update_mask]

    a_vals = projection_sorted[torch.arange(n_directions), best_start]
    b_vals = projection_sorted[torch.arange(n_directions), best_end - 1]

    return best_coverages, a_vals, b_vals



class WSC:
    def __init__(self, delta=0.1, seed=1234):
        """
        :param delta: Tolerance parameter. Minimum fraction of data required in a slab.
        """
        self.delta = delta
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _compute_masked_mean(values, mask, use_torch):
        """
        Helper to compute mean of values[mask].
        Designed to be usable by ANY metric class (ERT, FSC, WSC) to calculate sizes.
        
        :param values: Array/Tensor of values (e.g., prediction set sizes)
        :param mask: Boolean Array/Tensor indicating the subset (e.g., worst slice)
        :param use_torch: Boolean indicating backend
        :return: Float mean value
        """
        if use_torch:
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values, device=mask.device, dtype=torch.float32)
            
            subset = values[mask]
            if subset.numel() == 0:
                return float('nan')
            return subset.float().mean().item()
        else:
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)
            
            subset = values[mask]
            if subset.size == 0:
                return float('nan')
            return float(np.mean(subset))

    def _find_worst_slice_params(self, x, coverage, delta, M):
        """
        Finds the parameters (v, a, b) that define the worst slice on the given data.
        Returns the coverage found during search, the params, and a backend flag.
        """
        use_torch = isinstance(x, torch.Tensor) or isinstance(coverage, torch.Tensor)

        if use_torch:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=coverage.dtype)
            if not isinstance(coverage, torch.Tensor):
                coverage = torch.tensor(coverage, dtype=x.dtype)
            
            # --- Vectorized Search (PyTorch) ---
            V = sample_sphere_torch(M, x.shape[1])
            if x.is_cuda: V = V.to(x.device)
                
            coverage_list, a_list, b_list = wsc_all_torch(x, coverage, delta, V)
            idx_star = torch.argmin(coverage_list).item()
            
            return (
                float(coverage_list[idx_star].item()),
                V[idx_star],
                float(a_list[idx_star].item()),
                float(b_list[idx_star].item()),
                use_torch
            )
        else:
            if isinstance(x, list):
                x = np.asarray(x, dtype=float)
            if isinstance(coverage, list):
                coverage = np.asarray(coverage, dtype=float)

            # --- Vectorized Search (NumPy) ---
            V = sample_sphere_numpy(M, x.shape[1], self.rng)
            coverage_list, a_list, b_list = wsc_all_numpy(x, coverage, delta, V)
            idx_star = int(np.argmin(coverage_list))
            
            return (
                float(coverage_list[idx_star]),
                V[idx_star],
                float(a_list[idx_star]),
                float(b_list[idx_star]),
                use_torch
            )

    def _apply_slice_and_measure(self, x, v, a, b):
        """
        Applies the slice defined by (v, a, b) to data `x`.
        Calculates coverage and (optionally) size within that slice.
        """
        z = np.dot(x, v)
        n = 6 # truncate a and b due to numerical instability to make the results stable
        a = np.trunc(a * 10**n) / 10**n
        b = np.trunc(b * 10**n) / 10**n
        mask = (z >= a) & (z <= b)

        return mask

    def evaluate(self, x, cover, sizes=None, delta=None, M=1000, seed=42, unbiased=False, split_size=0.5):
        """
        Compute Worst Slab Coverage (WSC) with optional Size calculation.
        Unifies approaches from Romano et al. (2020) and Braun et al. (2025).

        Parameters
            x: Input samples (n, d).
            cover: Coverage indicator vector (n,).
            sizes: (Optional) Prediction set sizes (n,). If provided, returns size of worst slice.
            delta: Minimum fraction of data for a slice.
            M: Number of directions to search.
            seed: Random seed.
            unbiased: (Boolean)
                - False (Default): Direct Optimization (Braun et al.). Finds worst slice on `x`.
                  Returns the exact worst coverage found. Prone to pessimism.
                - True: Split Optimization (Romano et al.). Splits `x` into Search/Eval.
                  Finds worst slice on Search, measures it on Eval. Statistically unbiased.
            split_size: Fraction of data used for Search if unbiased=True.

        Returns
            If sizes is None: Float (WSC value)
            If sizes is provided: Dict {'coverage': float, 'size': float}
        """
        # seed_everything(seed)
        if delta is None:
            delta = self.delta
        
        check_delta(delta)
        check_cover(cover)
        check_tabular_strict(x)
        check_consistency(cover, x)
        if sizes is not None:
            check_consistency(cover, sizes)

        # -----------------------------------------------------------
        # LOGIC BRANCH: Unbiased (Split) vs Direct (Pessimistic)
        # -----------------------------------------------------------
        
        if unbiased:
            # --- Method 1: Romano et al. (Split/Unbiased) ---
            # 1. Split data. 
            # Note: We must split sizes too if they exist.
            if sizes is not None:
                x_search, x_eval, cov_search, cov_eval, _, sizes_eval = train_test_split(
                    x, cover, sizes, train_size=split_size, random_state=seed
                )
            else:
                x_search, x_eval, cov_search, cov_eval = train_test_split(
                    x, cover, train_size=split_size, random_state=seed
                )
                sizes_eval = None
            
            # 2. Find parameters (v, a, b) on Search Set
            # We discard the coverage returned here because it's the "search" coverage.
            _, v_star, a_star, b_star, use_torch = self._find_worst_slice_params(
                x_search, cov_search, delta, M
            )
            
            # 3. Measure on Eval Set using found parameters
            mask = self._apply_slice_and_measure(
                x_eval, v_star, a_star, b_star
            )

        else:
            # --- Method 2: Braun et al. (Direct) ---
            # 1. Find parameters on full dataset
            # Note: _find_worst_slice_params returns the coverage, but doesn't calculate size.
            # So we re-calculate using _apply_slice_and_measure to get size if needed.
            
            # First, get parameters
            _, v_star, a_star, b_star, use_torch = self._find_worst_slice_params(
                x, cover, delta, M
            )
            
            # Then, measure (this is redundant for coverage but necessary for size)
            mask = self._apply_slice_and_measure(
                x, v_star, a_star, b_star
            )

        return mask



class OrgWSC:
    def __init__(self, delta=None):

        self.delta = delta

    def wsc(self, x, coverage, delta, M=1000):
        """
        Compute the worst slab coverage value for the input data.

        Parameters
            x: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            coverage: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as x. 
            delta: Tolerance parameter that controls the size of the slab. 
            M: Number of random directions drawn on the unit sphere for the search procedure.

        Returns
            A tuple with:
                coverage_value: Worst coverage value found.
                direction: The direction vector on the sphere that produces this value.
                a: Auxiliary scalar returned by the backend solver.
                b: Auxiliary scalar returned by the backend solver.

        Notes
            It draws M random unit vectors and evaluates the coverage for each. The minimum value is returned.
        """
        use_torch = isinstance(x, torch.Tensor) or isinstance(coverage, torch.Tensor)

        if use_torch:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=coverage.dtype)
            if not isinstance(coverage, torch.Tensor):
                coverage = torch.tensor(coverage, dtype=x.dtype)
        else:
            if isinstance(x, list):
                x = np.asarray(x, dtype=float)
            if isinstance(coverage, list):
                coverage = np.asarray(coverage, dtype=float)

        # main loop
        if use_torch:    
            V = sample_sphere_torch(M, x.shape[1])
            coverage_list, a_list, b_list = wsc_all_torch(x, coverage, delta, V)
            idx_star = torch.argmin(coverage_list).item()
            return (
                float(coverage_list[idx_star].item()),
                V[idx_star].detach().cpu().numpy(),
                float(a_list[idx_star].item()),
                float(b_list[idx_star].item()),
            )
        else:
            V = sample_sphere_numpy(M, x.shape[1])
            coverage_list, a_list, b_list = wsc_all_numpy(x, coverage, delta, V)
            idx_star = int(np.argmin(coverage_list))
            return (
                float(coverage_list[idx_star]),
                V[idx_star],
                float(a_list[idx_star]),
                float(b_list[idx_star]),
            )

    def evaluate(self, x, cover, delta=None, M=1000, seed=42):
        """
        Compute the worst slab coverage value for the input data.

        Parameters
            x: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as x. 
            delta: Tolerance parameter that controls the size of the slab. 
            M: Number of random directions drawn on the unit sphere for the search procedure.

        Returns
            Float: WSC estimated

        Notes
            The function detects whether numpy or torch is in use and dispatches to the matching backend.
            It draws M random unit vectors and evaluates the coverage for each. The minimum value is returned.
        """
        seed_everything(seed)
        if delta is None:
            delta = self.delta
        check_delta(delta)
        check_cover(cover)
        check_tabular_strict(x)
        check_consistency(cover, x)

        wsc_value, _, _, _ = self.wsc(
            x, cover, delta=delta, M=M
        )
        return float(wsc_value)

