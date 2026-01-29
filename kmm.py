import torch
import logging
import numpy as np
import cvxpy as cp
from joblib import Parallel, delayed
import ipdb

logger = logging.getLogger("global_logger")


def estimate_distance_quantile(dists_sq, q=0.5, max_pairs=100000, seed=None):
    """
    Estimate the q-th quantile of pairwise squared distances between X1 and X2.
    Uses random subsampling of up to max_pairs pairs.
    """
    rng = np.random.default_rng(seed)
    m, n = dists_sq.shape

    total_pairs = m * n
    sample_size = min(max_pairs, total_pairs)

    flat_indices = rng.choice(total_pairs, size=sample_size, replace=False)
    idx1, idx2 = np.divmod(flat_indices, n)

    diffs = dists_sq[idx1, idx2].copy()
    selec_q = np.quantile(diffs, q)

    return selec_q



def rbf_kernel(X1, X2, kernel_width_q=0.5, max_pairs=10000, seed=None):
    m, n = X1.shape[0], X2.shape[0]
    dists_sq = torch.cdist(X1, X2, p=2)**2

    sigma_sq = estimate_distance_quantile(
        dists_sq.cpu().numpy(), q=kernel_width_q,
        max_pairs=max_pairs, seed=seed
    )
    if sigma_sq <= 0:
        sigma_sq = 1.0

    return torch.exp(-dists_sq / (2.0 * sigma_sq))



def _solve_kmm_for_batch(X, Z, kernel, B, epsilon, max_iters, kernel_width_q=0.5, reg_fac=1e-3, seed=None):
    device = X.device
    X_unique, unique_indices = torch.unique(X, dim=0, return_inverse=True)

    status_msg = ""

    removed_duplicates = False
    if X_unique.shape[0] < X.shape[0]:
        status_msg += f"Warning: Removed {X.shape[0] - X_unique.shape[0]} duplicate samples from the training data.\n"
        removed_duplicates = True
        X = X_unique.clone()

    m = X.shape[0]
    n = Z.shape[0]

    if epsilon is None:
        epsilon = (np.sqrt(m) - 1.0) / np.sqrt(m)

    if kernel == 'rbf':
        K = rbf_kernel(X, X, kernel_width_q, seed=seed)
        kappa = (m / n) * rbf_kernel(X, Z, kernel_width_q, seed=seed).sum(dim=1)
    else:
        K = X @ X.T
        kappa = (m / n) * (X @ Z.T).sum(dim=1)

    K = K.cpu().numpy().astype(np.float64)
    kappa = kappa.cpu().numpy().astype(np.float64)

    beta = cp.Variable(m, name="beta")

    solved = False
    curr_iter = 0
    beta_np = None

    preferred_solvers = [cp.OSQP, cp.ECOS, cp.SCS]

    while (not solved) and (curr_iter < max_iters):
        reg_K = K + np.eye(m, dtype=K.dtype) * reg_fac

        objective = cp.Minimize(0.5 * cp.quad_form(beta, reg_K) - kappa @ beta)
        constraints = [
            beta >= 0,
            beta <= B,
            cp.abs(cp.sum(beta) - m) <= m * epsilon
        ]
        problem = cp.Problem(objective, constraints)

        for solver in preferred_solvers:
            try:
                problem.solve(solver=solver, verbose=False)
                break
            except Exception:
                continue

        if problem.status in ("optimal", "optimal_inaccurate") and (beta.value is not None):
            status_msg += f"Optimization solved with status: {problem.status}\n"
            beta = np.array(beta.value, dtype=np.float64).reshape(-1)
            solved = True
        else:
            reg_fac *= 10
            status_msg += f"Non-PSD K matrix observed! Current regularization: {reg_fac}\n"
            curr_iter += 1

    if not solved:
        status_msg += "KMM solver failed to converge; returning uniform weights for this batch.\n"
        beta = np.ones((m,), dtype=np.float64)

    if removed_duplicates:
        inv = unique_indices.cpu().numpy().astype(np.int64)
        beta = beta[inv]

    beta = torch.from_numpy(beta.astype(np.float64)).to(X.device)
    return beta, status_msg



def kernel_mean_matching_parallel(
    X, Z, kernel='rbf', B=1000, epsilon=None,
    batch_size=1024, n_jobs=-1, max_iters=5,
    kernel_width_q=0.5, base_seed=None, reg_fac=1e-3, **kwargs
):
    """
    Kernel Mean Matching implementation.
    X: torch tensor (m x d) training data
    Z: torch tensor (n x d) test/unlabeled data
    Returns: torch tensor of length m with weights beta
    """
    m_original = X.shape[0]
    X = X.cpu()
    Z = Z.cpu()

    rng = np.random.default_rng(base_seed)

    if batch_size is None or m_original <= batch_size or n_jobs == 1:
        logger.info("Data size is smaller than batch size. Solving in a single run...")
        run_seed = rng.integers(0, 2**32 - 1)
        beta, status_msg = _solve_kmm_for_batch(
            X, Z, kernel, B, epsilon, max_iters, kernel_width_q,
            seed=run_seed
        )
        logger.info(status_msg)
        return beta

    logger.info(f"Parallelizing solver across {m_original} samples in batches of {batch_size}...")

    indices_np = rng.permutation(m_original)
    shuffled_indices = torch.from_numpy(indices_np)

    batches_of_indices = [
        shuffled_indices[i:i + batch_size]
        for i in range(0, m_original, batch_size)
    ]
    
    n_batches = len(batches_of_indices)
    worker_seeds = rng.integers(0, 2**32 - 1, size=n_batches)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_solve_kmm_for_batch)(
            X[indices], Z, kernel, B, epsilon, max_iters, kernel_width_q, reg_fac,
            seed=worker_seeds[i]
        )
        for i, indices in enumerate(batches_of_indices)
    )

    beta_full = torch.zeros(m_original, device=X.device, dtype=torch.float64)
    for i, indices in enumerate(batches_of_indices):
        beta_batch, status_msg = results[i]
        beta_full[indices] = beta_batch
        
        if status_msg:
            logger.info(f"[Batch {i+1}/{n_batches}]: {status_msg}")

    torch.clamp(beta_full, min=0.0, max=B)
    return beta_full