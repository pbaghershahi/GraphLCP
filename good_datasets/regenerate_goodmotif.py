import os
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT_PATH)

import torch
from good_datasets import GOODMotif
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from typing import List, Optional
import ipdb


class AddStructuralFeatures(BaseTransform):
    """
    A torch_geometric transform that calculates and adds structural node features.

    This transform is configurable via flags to add any combination of:
    1. Node Degree
    2. Local Clustering Coefficient
    3. PageRank Score
    4. Betweenness Centrality
    5. Closeness Centrality
    6. Eigenvector Centrality
    7. Katz Centrality
    
    The existing node features (`data.x`) will be replaced by these.
    """
    def __init__(self, degree: bool = True, clustering: bool = True, pagerank: bool = True,
                 betweenness: bool = True, closeness: bool = True, 
                 eigenvector: bool = True, katz: bool = True):
        self.degree = degree
        self.clustering = clustering
        self.pagerank = pagerank
        self.betweenness = betweenness
        self.closeness = closeness
        self.eigenvector = eigenvector
        self.katz = katz

    def __call__(self, data: Data) -> Data:
        """Applies the transformation to a single Data object."""
        
        nx_graph = to_networkx(data, to_undirected=not data.is_directed())

        if nx_graph.number_of_nodes() < data.num_nodes:
            nx_graph.add_nodes_from(range(data.num_nodes))

        feature_tensors = []
        feature_names = []

        if self.degree:
            degrees = [d for n, d in nx_graph.degree()]
            feature_tensors.append(torch.tensor(degrees, dtype=torch.float).view(-1, 1))
            feature_names.append("Degree")

        if self.clustering:
            coeffs = nx.clustering(nx_graph)
            coeffs_list = [coeffs.get(i, 0.0) for i in range(data.num_nodes)]
            feature_tensors.append(torch.tensor(coeffs_list, dtype=torch.float).view(-1, 1))
            feature_names.append("Clustering")

        if self.pagerank:
            try:
                pr = nx.pagerank(nx_graph, alpha=0.85)
                pr_list = [pr.get(i, 0.0) for i in range(data.num_nodes)]
                feature_tensors.append(torch.tensor(pr_list, dtype=torch.float).view(-1, 1))
            except nx.PowerIterationFailedConvergence:
                print("Warning: PageRank did not converge. Using zeros.")
                feature_tensors.append(torch.zeros((data.num_nodes, 1), dtype=torch.float))
            feature_names.append("PageRank")

        if self.betweenness:
            bc = nx.betweenness_centrality(nx_graph)
            bc_list = [bc.get(i, 0.0) for i in range(data.num_nodes)]
            feature_tensors.append(torch.tensor(bc_list, dtype=torch.float).view(-1, 1))
            feature_names.append("Betweenness")
        
        if self.closeness:
            cc = nx.closeness_centrality(nx_graph)
            cc_list = [cc.get(i, 0.0) for i in range(data.num_nodes)]
            feature_tensors.append(torch.tensor(cc_list, dtype=torch.float).view(-1, 1))
            feature_names.append("Closeness")

        if self.eigenvector:
            try:
                ec = nx.eigenvector_centrality(nx_graph, max_iter=1000, tol=1e-06)
                ec_list = [ec.get(i, 0.0) for i in range(data.num_nodes)]
                feature_tensors.append(torch.tensor(ec_list, dtype=torch.float).view(-1, 1))
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                print("Warning: Eigenvector Centrality did not converge. Using zeros.")
                feature_tensors.append(torch.zeros((data.num_nodes, 1), dtype=torch.float))
            feature_names.append("Eigenvector")
            
        if self.katz:
            try:
                kc = nx.katz_centrality(nx_graph, alpha=0.1, beta=1.0)
                kc_list = [kc.get(i, 0.0) for i in range(data.num_nodes)]
                feature_tensors.append(torch.tensor(kc_list, dtype=torch.float).view(-1, 1))
            except nx.PowerIterationFailedConvergence:
                print("Warning: Katz Centrality did not converge. Using zeros.")
                feature_tensors.append(torch.zeros((data.num_nodes, 1), dtype=torch.float))
            feature_names.append("Katz")
        
        if feature_tensors:
            data.x = torch.cat(feature_tensors, dim=1)
        else:
            print("Warning: No features selected. Node features remain unchanged.")
        
        return data


transform_custom = AddStructuralFeatures(
    degree=True, 
    clustering=False, 
    pagerank=True,
    betweenness=True,
    closeness=True,
    eigenvector=True,
    katz=False
)

dataset = GOODMotif.load(
    os.path.join(ROOT_PATH, "data"), 
    domain = "basis", 
    shift = "covariate", 
    generate = True,
    pre_transform = transform_custom,
    # force_reload = True
)