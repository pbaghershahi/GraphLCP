import torch, random, os, ipdb, logging
import numpy as np
from scipy.spatial.distance import cdist
from torch_cluster import random_walk
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_edge_index, add_self_loops, remove_self_loops, coalesce, is_undirected, to_undirected
from torch_sparse import SparseTensor



def remove_duplicated_edges(edge_index, edge_weight, num_nodes=None):

    edges = edge_index.T
    weights = edge_weight

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    if is_undirected(edge_index):
        edges = edges.sort(dim=1)[0]
            
    keys = edges[:, 0] * num_nodes + edges[:, 1]
    order = keys.argsort()

    edges = edges[order]
    weights = weights[order]

    mask = torch.ones(edges.size(0), dtype=torch.bool)
    mask[1:] = (edges[1:] != edges[:-1]).any(dim=1)

    edges = edges[mask]
    weights = weights[mask]

    edge_index = edges.T
    edge_weight = weights
    
    return edge_index, edge_weight



def compute_ppr_weights(
    edge_index, 
    edge_weights, 
    num_nodes, 
    anchor_nodes, 
    position_mat = None,
    alpha = 0.15, 
    k = 10
):
    """
    Computes PPR weights for all nodes relative to the 'anchor_nodes' using APPNP propagation.
    
    Args:
        edge_index: (2, E) Graph connectivity
        num_nodes: Total nodes in graph
        anchor_nodes: The 'random centers' (N_test,)
        alpha: Teleport/Restart probability
        k: Number of propagation steps
    
    Returns:
        weights: (N_test, N_total) Matrix where w[i, j] is the influence of node j on anchor i.
                 Note: PPR is usually defined as influence of i -> j. Here we need the weight 
                 of calibration nodes (sources) 'hitting' the anchor (target).
                 For undirected graphs, this is symmetric. For directed, we might need A^T.
                 Assuming undirected for standard GNN tasks.
    """

    device = edge_index.device
    num_anchors = anchor_nodes.size(0)
    
    row, col = edge_index

    if isinstance(edge_weights, torch.Tensor):
        # Compute Transition Probabilities: w_ij / sum(w_i)
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_reduce_(0, row, edge_weights.to(deg.dtype), reduce='sum', include_self=False)
    else:
        # Compute Transition Probabilities: 1 / deg(source)
        deg = degree(row, num_nodes, dtype=torch.float)
        edge_weights = 1.0
    
    # Inverse degree (1/d) for Random Walk Normalization
    deg_inv = deg.pow(-1.0)
    deg_inv[deg_inv == float('inf')] = 0
    
    # Note: In SparseTensor(row=target, col=source), 'col' attribute holds sources.
    # But here 'row' variable holds edge_index[0] (sources).
    norm = edge_weights * deg_inv[row]
    
    # Create SparseTensor for efficient SpMM
    adj_t = SparseTensor(row=col, col=row, value=norm, sparse_sizes=(num_nodes, num_nodes))

    # 2. Initialize Signal H (One-Hot at Anchor Nodes)
    # We want to compute PPR vectors FOR the anchors. 
    # Shape: (Num_Nodes, Num_Anchors)
    # Note: If Num_Anchors is very large (e.g. 10k), this matrix might be too big.
    # In that case, we should batch this computation. 
    # But rand_lcp.py already batches by 'batch_size' (default 1024), so this is safe.
    
    if position_mat is None:
        H = torch.zeros(num_nodes, num_anchors, device=device)
        # Set 1.0 at the anchor indices for each corresponding column
        H[anchor_nodes, torch.arange(num_anchors, device=device)] = 1.0
    else:
        H = position_mat.to(device)
    
    Z = H.clone()

    # 3. Power Iteration (APPNP Propagation)
    # Z^(k+1) = (1-alpha) * (A @ Z^(k)) + alpha * H
    for _ in range(k):
        Z = (1 - alpha) * (adj_t @ Z) + alpha * H
    
    # Z[j, i] is the probability of walk starting at Anchor i ending at Node j?
    # No, APPNP computes: "If I start at Anchor i, where do I end up?"
    # So Z[j, i] is Prob(Anchor -> Node j).
    
    # For RLCP, we want weights for calibration nodes w.r.t the anchor.
    # "How likely is it that cal_node falls into the anchor's PPR distribution?"
    # So w_j = PPR_anchor(j).
    # This matches Z[j, i].
    
    # Transpose to match RLCP shape expectation: (N_test, N_total)
    weights = Z.T
    
    return weights



def sample_ppr_anchor(
    edge_index, 
    start_nodes, 
    alpha=0.15, 
    k=10, 
    num_nodes=None
):
    """
    Samples random anchors using a "Geometric Random Walk".
    Matches the PPR kernel distribution: P(L=t) = alpha * (1-alpha)^t
    
    Strategy: "Batch by Length"
    1. Pre-sample walk lengths for all nodes.
    2. Group nodes by walk length.
    3. Execute fixed-length walks for each group efficiently.
    """

    device = edge_index.device
    num_test = start_nodes.size(0)
    start_nodes = start_nodes.to(device)
    
    if start_nodes.ndim == 2:
        start_nodes = torch.multinomial(start_nodes, 1).view(-1)

    # 1. Sample Walk Lengths (Geometric Distribution)
    # We sample L ~ Geometric(alpha), clamped to K.
    # Formula for sampling Geometric: floor( log(U) / log(1-p) )
    rand_vals = torch.rand(num_test, device=device)
    # Add epsilon to log to avoid log(0)
    walk_lengths = (torch.log(rand_vals + 1e-10) / torch.log(torch.tensor(1 - alpha))).floor().long()
    walk_lengths = torch.clamp(walk_lengths, min=0, max=k)
    
    final_anchors = torch.zeros_like(start_nodes)
    
    # 2. Batch by Length
    unique_lengths = torch.unique(walk_lengths)
    
    for length in unique_lengths:
        mask = (walk_lengths == length)

        nodes_to_walk = start_nodes[mask]
        
        if length == 0:
            # Walk length 0: Anchor is the start node itself
            final_anchors[mask] = nodes_to_walk
        else:
            # Perform fixed length walk
            # random_walk returns (row, col). We just want the final destination.
            # Note: random_walk requires row, col vectors, not edge_index
            row, col = edge_index
            # Returns: [start_node, step1, step2, ..., step_L] (size: N x (L+1))
            walk_res = random_walk(row, col, nodes_to_walk, walk_length=length.item())
            
            # Take the last column as the destination
            destinations = walk_res[:, -1]
            final_anchors[mask] = destinations
            
    return final_anchors



def compute_guassian_weights(
    center_x, cal_x, test_x=None, bandwidth=1.0, normalize=True, eigenvectors=None
):

    if isinstance(bandwidth, np.ndarray) and bandwidth.ndim == 1 and bandwidth.size > 1:
        
        Sigma_inv_z = np.diag(1 / bandwidth) # the ipput bandwidth is self.bandwidth**2 * (variances + 1e-8)
        
        if eigenvectors is not None:
            Sigma_inv_z = eigenvectors.T @ Sigma_inv_z @ eigenvectors
        
        dists = cdist(center_x, cal_x, 'mahalanobis', VI=Sigma_inv_z)**2

        if test_x is not None:
            delta = test_x - center_x
            dists_to_test = np.sum((delta @ Sigma_inv_z) * delta, axis=1, keepdims=True)
            dists = np.concatenate([dists, dists_to_test], axis=1)

        sim_vec = np.exp(-0.5 * dists)
    
    else:
        dists = cdist(center_x, cal_x, 'sqeuclidean')
        
        if test_x is not None:
            dist_to_test = np.sum((test_x - center_x)**2, axis=-1)[:, None]
            dists = np.concatenate((dists, dist_to_test), axis=1)

        sim_vec = np.exp(-dists / (2 * bandwidth ** 2))

    if normalize:
        sim_vec += 1e-9
        weights = sim_vec / (np.sum(sim_vec, axis=-1))[:, None]
    else:
        weights = sim_vec
        
    return weights



def sample_from_guassian_kernel(x, sampling_scale=1.0, eigenvectors=None):

    # Create a matrix of standard normal noise (mean=0, std=1)
    if isinstance(sampling_scale, np.ndarray) and sampling_scale.ndim == 1 and sampling_scale.size > 1:
        noise_d_feats = sampling_scale.shape[0]
    else:
        noise_d_feats = x.shape[1]

    noise = np.random.randn(x.shape[0], noise_d_feats)
    scaled_noise = noise * sampling_scale # sampling_scale is self.bandwidth * np.sqrt(variances) in case of sampling along pca

    if eigenvectors is None:
        samples = x + scaled_noise
    else:
        samples = x + scaled_noise @ eigenvectors

    return samples



def get_shortest_paths(edge_index, num_nodes, test_idxs, cal_idxs):
    """
    Computes the shortest path distances (hops) between specific test nodes and calibration nodes.
    
    Args:
        edge_index (LongTensor): Graph connectivity (2, E).
        num_nodes (int): Total number of nodes in the graph.
        test_idxs (LongTensor): Indices of the test nodes (Sources).
        cal_idxs (LongTensor): Indices of the calibration nodes (Targets).
        
    Returns:
        dists (Tensor): A FloatTensor of shape (num_test, num_cal) containing 
                        shortest path distances. Inf indicates no path.
    """
    # 1. Convert PyG edge_index to Scipy Sparse Matrix (CSR format)
    # This is required for fast C-based path computation
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # 2. Compute Shortest Paths
    # We use 'indices' to only compute paths starting from the test nodes.
    # directed=False treats the graph as undirected (standard for most GNN tasks).
    # unweighted=True uses BFS (faster than Dijkstra).
    
    # sources_to_all shape: (num_test, num_total_nodes)
    sources_to_all = shortest_path(
        csgraph=adj_matrix, 
        method='auto', 
        directed=False, 
        unweighted=True, 
        indices=test_idxs.cpu().numpy()
    )

    # 3. Extract the specific columns for calibration nodes
    # Result shape: (num_test, num_cal)
    dists_to_cal = sources_to_all[:, cal_idxs.cpu().numpy()]

    return torch.from_numpy(dists_to_cal).float()



def get_all_khop_neighbors(num_nodes, edge_index, k, drop_self_loops=False):
    
    # 1. Add self-loops so that A^k includes neighbors from A^{k-1}, A^{k-2}...
    #    (i.e., "within" k hops, not "exactly" k hops)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 2. Convert to SparseTensor for efficient matrix multiplication
    #    This is generally much faster than raw torch.sparse_coo_tensor
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       sparse_sizes=(num_nodes, num_nodes))

    # 3. Perform Matrix Power: A^k
    #    Note: We treat values as 1.0 to check reachability.
    #    If k is small (e.g., 2 or 3), this is very fast.
    adj_k = adj
    for _ in range(k - 1):
        adj_k = adj_k @ adj

    # 4. Extract the indices (row, col) where connection exists
    row, col, _ = adj_k.coo()
    
    khop_edge_index = torch.stack([row, col], dim=0)

    if drop_self_loops:
        khop_edge_index, _ = remove_self_loops(khop_edge_index)

    return khop_edge_index



def cosine_similarity(x, idxs1=None, idxs2=None):

    if idxs1 is None or idxs2 is None:
        norm_x = x.norm(dim=1)

        assert (norm_x > 0).all()
        
        x = x / norm_x[:, None]
        sim_mat = x @ x.T
    else:
        x1 = x[idxs1]
        x2 = x[idxs2]
    
        norm_x1 = x1.norm(dim=1)[:, None]
        norm_x2 = x2.norm(dim=1)[:, None]

        assert (norm_x1 > 0).all() and (norm_x2 > 0).all()

        x1 = x1 / norm_x1
        x2 = x2 / norm_x2

        sim_mat = x1 @ x2.T

    return sim_mat



def get_rbf_similarity(x, bandwidth=1.0):

    if isinstance(bandwidth, np.ndarray) and bandwidth.ndim == 1 and bandwidth.size > 1:
        
        Sigma_inv_z = np.diag(1 / bandwidth) # the ipput bandwidth is self.bandwidth**2 * (variances + 1e-8)
        dists = cdist(x, x, 'mahalanobis', VI=Sigma_inv_z)**2
        sim_mat = np.exp(-0.5 * dists)
    
    else:
        dists = cdist(x, x, 'sqeuclidean')
        sim_mat = np.exp(-dists / (2 * bandwidth**2))
    
    return sim_mat



def get_dense_graph(x, edge_index, method="rbf", bandwidth=1.0, threshold=0.1, num_nodes=None, sim_mat=None):

    if sim_mat is None:
        if method == "cosine":
            sim_mat = cosine_similarity(torch.from_numpy(x))
        else:
            sim_mat = get_rbf_similarity(x, bandwidth)
            sim_mat = torch.from_numpy(sim_mat)

        if num_nodes is None:
            num_nodes = x.shape[0]

        sim_mat.fill_diagonal_(0.0)
        sim_mat[edge_index[0, :], edge_index[1, :]] = 1.0
        
    mask = sim_mat > threshold

    edge_index = mask.nonzero().T
    edge_weight = sim_mat[edge_index[0, :], edge_index[1, :]]
    # row, col = mask.nonzero(as_tuple=True)
    # rbf_weights = sim_mat[row, col]
    
    # orig_row, orig_col = edge_index
    # orig_weights = torch.ones(orig_row.shape[0])
    
    # combined_edges = torch.stack([
    #     torch.cat([orig_row, row]),
    #     torch.cat([orig_col, col])
    # ], dim=0)
    # combined_weights = torch.cat([orig_weights, rbf_weights])

    # edge_index, edge_weight = remove_duplicated_edges(combined_edges, combined_weights, num_nodes)
    
    return edge_index, edge_weight, sim_mat



def graph_homophily(edge_index, y, task='cls'):
    """
    Computes a single scalar homophily value for the entire graph [0, 1].
    """
    
    # 1. Prepare Edge Index (Undirected, No Self-Loops)
    # We want to count every connection once.
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
        
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    
    # Avoid division by zero if graph has no edges
    if row.shape[0] == 0:
        return 0.0

    # 2. Compute Metric
    if task == 'cls':
        # Ratio of edges connecting same-class nodes
        # sum(y_u == y_v) / total_edges
        is_same = (y[row] == y[col]).float()
        graph_homophily = is_same.mean().item()
        
    elif task == 'reg':
        y = y.float()
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze()
            
        # Calculate diffs for all edges
        diffs = torch.abs(y[row] - y[col])
        
        # Normalize by the MAX difference found in the graph structure
        # (Consistent with your node-level logic)
        max_diff = diffs.max()
        
        if max_diff == 0:
            # All connected nodes have identical values
            graph_homophily = 1.0
        else:
            # Average normalized difference
            mean_norm_diff = (diffs / max_diff).mean().item()
            graph_homophily = 1.0 - mean_norm_diff
            
    else:
        raise ValueError("task must be 'cls' or 'reg'")
        
    return graph_homophily