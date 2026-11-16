import torch, random, os, ipdb, logging
import torch.nn as nn
from copy import deepcopy
from .base_cp import BaseCP
from .aps import aps_transform
from model import GNN
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch_geometric
from torch_sparse import SparseTensor
from sklearn.metrics.pairwise import cosine_similarity
import ipdb

logger = logging.getLogger("global_logger")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape,
        dtype=values.dtype,
    )



def compute_adj_knn(features, k=20):

    features = np.copy(features.cpu())
    features[features != 0] = 1
    sims = cosine_similarity(features)
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
    
    for i in range(len(sims)):
        indices_argsort = np.argsort(sims[i])
        sims[i, indices_argsort[:-k]] = 0
    
    A_feat = sp.coo_matrix(sims)
    row_sum = np.array(A_feat.sum(1))
    d_inv = np.power(row_sum + 1e-10, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    A_feat = d_mat_inv.dot(A_feat).tocoo()
    
    adj_knn_st = sparse_mx_to_torch_sparse_tensor(A_feat).float()
    
    return adj_knn_st



def snaps_transform(scores, edge_index, n_vertices, adj_knn, **kwargs):
    scores = scores.to(device)
    edge_index = edge_index.to(device)

    A = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1], device=device, dtype=scores.dtype),
        size=(n_vertices, n_vertices),
        dtype=scores.dtype,
        device=device,
    )

    degs = torch.matmul(A, torch.ones(A.shape[0], device=device, dtype=scores.dtype))
    aggregated_scores = torch.matmul(A, scores) * (1 / (degs + 1e-10))[:, None]

    knn_scores = torch.matmul(adj_knn.to(device), scores)

    res = (scores + aggregated_scores + knn_scores) / 3
    return res



class SNAPS(BaseCP):
    def __init__(
        self,
        base_method = "split",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_method = base_method
        if base_method == "aps":
            assert self.task == "cls", "APS-based weighting only supports classification tasks."
            self.cp_method = self.snaps
        elif base_method == "split":
            self.cp_method = self.snl1
        else:
            raise ValueError(f"Unknown base_method: {base_method}")



    def snaps(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights = None,
        **kwargs
    ):
        # With APS scoring. 
        assert self.task == "cls", "SNAPS is only implemented for classification tasks."
        num_nodes = dataset.num_nsamples

        edge_index = dataset._data.edge_index.clone()
        adj_knn = compute_adj_knn(dataset.x, k=20)

        aps_scores = aps_transform(all_outs["scores"])

        snaps_scores = snaps_transform(
            scores=torch.from_numpy(aps_scores).to(torch.float32),
            edge_index=edge_index,
            n_vertices=num_nodes, 
            adj_knn=adj_knn
        ).cpu().numpy()

        test_nc_scores = snaps_scores[test_idxs]
        cal_labels = all_outs["labels"][cal_idxs.numpy()]
        cal_nc_scores = snaps_scores[cal_idxs, cal_labels]

        n_test = test_nc_scores.shape[0]
        n_cal = cal_nc_scores.shape[0]
        n_cls = all_outs["scores"].shape[1]

        ord_idxs = np.argsort(cal_nc_scores)
        sorted_nc_scores = cal_nc_scores[ord_idxs]
        
        quantile_level = (1 - self.alpha) * (1 + 1/n_cal)

        if pre_weights is None:
            pre_weights = np.ones_like(cal_nc_scores) / n_cal
        else:
            pre_weights = pre_weights / pre_weights.sum()

        cumulative_probs = np.cumsum(pre_weights[ord_idxs])
        quantile_idx = np.min(np.where(cumulative_probs >= quantile_level))
        
        quantile_thresh = sorted_nc_scores[quantile_idx]
        test_conf_pred = test_nc_scores <= quantile_thresh

        outputs = {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
        
        return outputs


    def snl1(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights = None,
        **kwargs
    ):
        # With L1 scoring. 
        num_nodes = dataset.num_nsamples
        edge_index = dataset._data.edge_index.clone()
        adj_knn = compute_adj_knn(dataset.x, k=20)

        if self.task == "reg":
            nc_scores = np.abs(all_outs["labels"] - all_outs["scores"]).reshape(-1)
        else:
            nc_scores = 1 - all_outs["scores"]

        snaps_scores = snaps_transform(
            scores=torch.from_numpy(nc_scores).to(torch.float32),
            edge_index=edge_index,
            n_vertices=num_nodes, 
            adj_knn=adj_knn
        )

        if self.task == "reg":
            cal_nc_scores = snaps_scores[cal_idxs].cpu().numpy().reshape(-1)
        else:
            cal_labels = all_outs["labels"][cal_idxs.numpy()]
            cal_nc_scores = snaps_scores[cal_idxs, cal_labels].cpu().numpy()

        test_scores = all_outs["scores"][test_idxs]
        n_test = test_scores.shape[0]
        n_cal = cal_nc_scores.shape[0]
        n_cls = all_outs["scores"].shape[1]

        ord_idxs = np.argsort(cal_nc_scores)
        sorted_nc_scores = cal_nc_scores[ord_idxs]
        
        quantile_level = (1 - self.alpha) * (1 + 1/n_cal)

        if pre_weights is None:
            pre_weights = np.ones_like(cal_nc_scores) / n_cal
        else:
            pre_weights = pre_weights / pre_weights.sum()

        cumulative_probs = np.cumsum(pre_weights[ord_idxs])
        quantile_idx = np.min(np.where(cumulative_probs >= quantile_level))
        quantile_thresh = sorted_nc_scores[quantile_idx]

        if self.task == "reg":
            test_conf_pred = np.zeros((n_test, n_cls, 2))
            test_conf_pred[:, 0] = np.concatenate(
                (test_scores - quantile_thresh.reshape(-1, 1), test_scores + quantile_thresh.reshape(-1, 1)),
                axis=1
            )
        else:
            test_nc_scores = snaps_scores[test_idxs].cpu().numpy()
            test_conf_pred = test_nc_scores <= quantile_thresh

        outputs = {
            "test_predictions": test_conf_pred,
            "quantile_thresholds": quantile_thresh,
        }
        
        return outputs