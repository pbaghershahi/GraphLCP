import numpy as np



def entropy(weights, **kwargs):
    """
    Calculates the Shannon Entropy of a probability distribution.
    Math: H(W) = -sum(w_i * log2(w_i))
    Interpretation:
        - Higher Value: The distribution is more uniform and less "sharp." Weights are spread out.
        - Lower Value: The distribution is more concentrated on a few values (more "sharp").
    """
    # Add a small epsilon to avoid log(0) for zero weights
    epsilon = 1e-12
    weights = np.clip(weights, epsilon, 1.0)
    return -(weights * np.log2(weights)).sum(axis=1)



def center_of_mass(weights, **kwargs):
    """
    Calculates the center of mass (or weighted average index) of the distribution.
    Math: C(W) = sum(i * w_i) for i in 0..n-1
    Interpretation (assuming columns are sorted from low to high non-conformity):
        - Lower Value: The weight is concentrated on the early indices (low non-conformity samples).
        - Higher Value: The weight is concentrated on the later indices (high non-conformity samples).
    """
    indices = np.arange(weights.shape[1])[None, :]
    return (weights * indices).sum(axis=1)



def herfindahl_hirschman_index(weights, **kwargs):
    """
    Calculates the Herfindahl-Hirschman Index (HHI), a measure of concentration.
    Math: HHI(W) = sum(w_i^2)
    Interpretation:
        - Lower Value (closer to 0): The distribution is less concentrated (less sharp).
        - Higher Value (closer to 1): The distribution is highly concentrated on a few values.
    """
    return (weights**2).sum(axis=1)



def effective_number_of_samples(weights, **kwargs):
    """
    Calculates the effective number of samples, derived from the HHI.
    Math: N_eff(W) = 1 / sum(w_i^2)
    Interpretation:
        - Higher Value: The distribution is spread out over more samples (less sharp).
        - Lower Value (closer to 1): The distribution is concentrated on fewer samples.
    """
    hhi = herfindahl_hirschman_index(weights)
    # Safely handle division by zero
    return np.divide(1.0, hhi, out=np.zeros_like(hhi, dtype=float), where=hhi!=0)



def cumulative_weight_at_quantile(weights, quantile=0.1, **kwargs):
    """
    Calculates the sum of weights up to a specified quantile of the samples.
    Math: CDF_q(W) = sum(w_i) for i in 0..floor(q*n)
    Interpretation (assuming columns are sorted from low to high non-conformity):
        - Higher Value: More weight is allocated to the lowest non-conformity samples.
        - Lower Value: Less weight is allocated to the lowest non-conformity samples.
    """
    if not (0 < quantile <= 1):
        raise ValueError("Quantile must be between 0 and 1.")
    cutoff_index = int(np.floor(quantile * weights.shape[1]))
    return weights[:, :cutoff_index].sum(axis=1)



def median_index(weights, **kwargs):
    """
    Finds the median index of the distribution.
    Math: k such that sum(w_i) for i<k is < 0.5 and sum(w_i) for i<=k is >= 0.5
    Interpretation (assuming columns are sorted from low to high non-conformity):
        - Lower Value: The distribution's mass is concentrated on earlier indices.
        - Higher Value: The distribution's mass is concentrated on later indices.
    """
    cumulative_weights = weights.cumsum(axis=1)
    # Find the first index where the cumulative sum is >= 0.5 for each row
    return (cumulative_weights >= 0.5).argmax(axis=1)



def analyze_distribution(weights, metric_name='entropy', **kwargs):
    """
    General function to compute a specific metric for a weight distribution.

    Args:
        weights (np.ndarray): A 1D NumPy array representing a probability distribution.
        metric_name (str): The name of the metric to compute.
                           Options: 'entropy', 'center_of_mass', 'hhi',
                                    'effective_samples', 'cumulative_weight', 'median_index'.
        **kwargs: Additional arguments for specific metrics (e.g., 'quantile' for cumulative_weight).

    Returns:
        float: The calculated metric value.
    """
    metrics = {
        'entropy': entropy,
        'center_of_mass': center_of_mass,
        'hhi': herfindahl_hirschman_index,
        'effective_samples': effective_number_of_samples,
        'cumulative_weight': cumulative_weight_at_quantile,
        'median_index': median_index
    }

    metric_func = metrics[metric_name]
    return metric_func(weights, **kwargs)
