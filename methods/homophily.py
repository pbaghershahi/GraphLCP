"""Graph homophily helpers used by TestCP."""

import numpy as np
import torch
from torch_geometric.utils import is_undirected, remove_self_loops, to_undirected, to_torch_coo_tensor
from torch_scatter import scatter

def get_k_hop_edge_index(edge_index, num_nodes, k=1):

    if k == 1:
        return edge_index

    adj = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes)).coalesce()
    accum_adj = adj.clone()
    pow_adj = adj.clone()

    for _ in range(k - 1):
        pow_adj = torch.sparse.mm(pow_adj, adj).coalesce()
        accum_adj = accum_adj + pow_adj
        
    accum_adj = accum_adj.coalesce()
    indices = accum_adj.indices()
    
    indices, _ = remove_self_loops(indices)
    
    return indices

def compute_test_node_homophily(data, test_indices, task='cls', k=1, edge_index=None):
    num_nodes = data.num_nodes
    y = data.y
    
    if edge_index is None:
        edge_index = data.edge_index

    if is_undirected(edge_index):
        base_edge_index = edge_index
    else:
        base_edge_index = to_undirected(edge_index)
        
    base_edge_index, _ = remove_self_loops(base_edge_index)
    
    # Get k-hop connectivity
    edge_index = get_k_hop_edge_index(base_edge_index, num_nodes, k)
    row, col = edge_index

    if task == 'cls':
        # Classification: 1 if same, 0 if different
        is_same = (y[row] == y[col]).float()
        node_metric = scatter(is_same, row, dim=0, dim_size=num_nodes, reduce='mean')

    elif task == 'reg':
        # Regression: Normalized Distance
        y = y.float()
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze()
        
        # 1. Calculate absolute difference for ALL edges
        diff = torch.abs(y[row] - y[col])
        
        # 2. Find the maximum difference observed in the graph structure
        # (This avoids squashing scores if the global y_range is huge but edges are short)
        max_diff = diff.max()
        
        if max_diff == 0:
            # If max difference is 0, it means every single neighbor has the exact same value.
            # Homophily is perfect (1.0) everywhere.
            node_metric = torch.ones(num_nodes, device=y.device)
        else:
            # 3. Normalize difference relative to the worst edge in the graph
            # normalized_diff is 0 (identical) to 1 (worst observed edge)
            normalized_diff = diff / max_diff
            
            # 4. Aggregate neighbors
            mean_norm_diff = scatter(normalized_diff, row, dim=0, dim_size=num_nodes, reduce='mean')
            
            # 5. Invert: 1.0 = Perfect Homophily, 0.0 = Worst observed local heterogeneity
            node_metric = 1.0 - mean_norm_diff

    else:
        raise ValueError("task must be 'cls' or 'reg'")

    metric_np = node_metric.cpu().numpy().flatten()
    metric_np = np.nan_to_num(metric_np, nan=0.0)
    
    if isinstance(test_indices, torch.Tensor):
        test_indices = test_indices.cpu().numpy()
    
    return metric_np[test_indices]
