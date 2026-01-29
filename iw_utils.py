import os, logging, sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[0])

from kmm import kernel_mean_matching_parallel
from model import NonParamatericEncoder
import torch
import numpy as np
from munch import Munch
import ipdb

logger = logging.getLogger("global_logger")



def get_weights(w_path, dataset, iw_config, setup, encoder_config=None, latent_outs=None):

    if w_path.is_file():
        pre_weights = torch.load(w_path, weights_only=False)
        logger.info(f"Reading weights from: {w_path}")
    else:
        if setup["space"] == "input":
            logger.info(f"Creating and saving input embedding weights to: {w_path}")
            pre_weights = weights_from_input_embeddings(dataset, encoder_config, iw_config, setup["device"])
        elif setup["space"] == "latent":
            logger.info(f"Creating and saving latent embedding weights to: {w_path}")
            pre_weights = weights_from_latent_embeddings(dataset, latent_outs, iw_config)

        torch.save(pre_weights, w_path)
    
    return pre_weights.cpu().numpy()



def weights_from_input_embeddings(dataset, encoder_config, iw_config, device):

    simple_encoder = NonParamatericEncoder(**encoder_config).to(device)

    def encode(t_data_loader):
        batch_x = []
        for i, (batch, idxs) in enumerate(t_data_loader):
            batch_x += [simple_encoder(batch.to(device))]
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x

    if dataset.level == "node":
        all_x = encode(dataset.test_loader)
        cal_x = all_x[dataset.cal_idxs]
        test_x = all_x[dataset.test_idxs]
    else:
        cal_x = encode(dataset.cal_loader)
        test_x = encode(dataset.test_loader)

    pre_weights = kernel_mean_matching_parallel(
        torch.as_tensor(cal_x), 
        torch.as_tensor(test_x),
        **iw_config
    )

    return pre_weights



def weights_from_latent_embeddings(dataset, latent_outs, iw_config):

    cal_x = latent_outs["cal_outs"]["x"]
    test_x = latent_outs["test_outs"]["x"]

    pre_weights = kernel_mean_matching_parallel(
        torch.as_tensor(cal_x), 
        torch.as_tensor(test_x),
        **iw_config
    )

    return pre_weights



def generic_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma_sq: float = 1.0, max_pairs=1000000) -> torch.Tensor:

    dist_sq = torch.cdist(X, Y, p=2) ** 2
    m, n = dist_sq.shape
    total_pairs = m * n

    if total_pairs > max_pairs:
        sigma_sq = 1.0
    else:
        sigma_sq = max(dist_sq.median() + 1e-9, sigma_sq)

    kernel_matrix = torch.exp(-dist_sq / (2 * sigma_sq))
    return kernel_matrix



def mmd(set1: torch.Tensor, set2: torch.Tensor, sigma_sq=1.0, iw=None, max_size=10000):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

    MMD is a distance metric between distributions. An MMD of zero means the
    distributions are identical. This implementation uses the biased estimator
    of MMD^2.

    Args:
        set1 (torch.Tensor): Samples from the source distribution, tensor of shape (N, D).
        set2 (torch.Tensor): Samples from the target distribution, tensor of shape (M, D).
        sigma_sq (float): The RBF kernel bandwidth.
        iw (torch.Tensor): importance weights.

    Returns:
        torch.Tensor: A scalar tensor representing the MMD value.
    """

    # Ensure the tensors have the same number of feature dimensions
    assert set1.shape[1] == set2.shape[1], \
        f"The feature dimensions of the two sets must be the same. " \
        f"Got {set1.shape[1]} and {set2.shape[1]}."

    rng = np.random.default_rng()
    sets = [set1, set2]
    for i, x in enumerate(sets):
        if x.shape[0] > max_size:
            rand_idxs = rng.choice(x.shape[0], size=max_size, replace=False)
            sets[i] = x[rand_idxs]
    set1, set2 = sets

    # Compute the kernel matrices for within and between sets
    k_xx = generic_rbf_kernel(set1, set1, sigma_sq)
    k_yy = generic_rbf_kernel(set2, set2, sigma_sq)
    k_xy = generic_rbf_kernel(set1, set2, sigma_sq)

    if iw is None:
        # Calculate the MMD^2 statistic
        # MMD^2 = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
        mmd_sq = k_xx.mean() - 2 * k_xy.mean() + k_yy.mean()
    else:
        # Calculate the MMD^2 statistic
        # MMD^2 = E[β_iβ_jk(x_i, x_j)] - 2*E[β_ik(x_i, y_j)] + E[k(y, y')]
        weighted_k_xx = iw[None, :] * iw[:, None] * k_xx
        weighted_k_xy = iw[:, None] * k_xy
        mmd_sq = weighted_k_xx.mean() - 2 * weighted_k_xy.mean() + k_yy.mean()

    # The MMD value can be negative due to estimation noise with the biased
    # estimator, so we clamp it at 0 before taking the square root.
    return torch.sqrt(torch.clamp(mmd_sq, min=0))