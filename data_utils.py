import torch, random, os, logging, ipdb, gc
import numpy as np
from torch_geometric.datasets import QM9, TUDataset, Flickr, CitationFull, Amazon, Coauthor
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler, NeighborLoader
from utils import *
from model import *
import pandas as pd
from torch_geometric.loader import DataLoader as PyG_Dataloader
from torch_geometric.data import Data, Batch, Dataset as PyG_Dataset
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.utils import degree, is_undirected, to_undirected, remove_self_loops, to_dense_adj, to_torch_coo_tensor, to_networkx
import networkx as nx
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from sklearn.decomposition import PCA
from good_datasets import GOODHIV, GOODTwitch, GOODMotif, GOODCMNIST, GOODWebKB, GOODCBAS, GOODZINC
from torch_scatter import scatter
from sklearn.cluster import KMeans
from torch_geometric.data import InMemoryDataset
from munch import Munch
from iw_utils import mmd
from model import NonParamatericEncoder


logger = logging.getLogger("global_logger")



def dataset_embeds_and_dists(dataset, encoder_config, device="cuda"):

    simple_encoder = NonParamatericEncoder(**encoder_config).to(device)

    def encode(t_data_loader):
        batch_x = []
        for i, (batch, idxs) in enumerate(t_data_loader):
            batch_x += [simple_encoder(batch.to(device))]
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x

    if dataset.level == "node":
        all_x = encode(dataset.test_loader)
        train_x = all_x[dataset.train_idxs]
        cal_x = all_x[dataset.cal_idxs]
        test_x = all_x[dataset.test_idxs]
    else:
        train_x = encode(dataset.train_loader)
        cal_x = encode(dataset.cal_loader)
        test_x = encode(dataset.test_loader)

    cal_train_mmd = mmd(cal_x, train_x, sigma_sq=1.0).cpu().numpy()
    test_train_mmd = mmd(test_x, train_x, sigma_sq=1.0).cpu().numpy()
    cal_test_mmd = mmd(cal_x, test_x, sigma_sq=1.0).cpu().numpy()

    logger.info(
        f"Cal-Train MMD distance: {cal_train_mmd.item():.3f} -- "
        f"Test-Train MMD distanceth: {test_train_mmd.item():.3f} -- "
        f"Cal-Test MMD distance: {cal_test_mmd.item():.3f}"
    )

    dists = dict(
        train_x = train_x.cpu(),
        cal_x = cal_x.cpu(),
        test_x = test_x.cpu(),
        org_cal_train_mmd = cal_train_mmd.item(),
        org_test_train_mmd = test_train_mmd.item(),
        org_cal_test_mmd = cal_test_mmd.item(),
    )
    return dists



class SimpleDataset(Dataset):
    def __init__(self,
                 graph_list: List,
                 **kwargs) -> None:
        super(SimpleDataset, self).__init__()
        self._data = graph_list

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], idx



class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class SubsetRandomSampler(SubsetSampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))



class CustomNeighborLoader(NeighborLoader):
    def __init__(self, data, **kwargs):
        kwargs = {k:v for k, v in kwargs.items() if k in ["batch_size", "num_neighbors", "input_nodes"]}
        super().__init__(data, **kwargs)

    def collate_fn(self, data_list):
        batch = super().collate_fn(data_list)
        return batch, batch.input_id



class CustomGraphSAINTLoader(GraphSAINTRandomWalkSampler):
    def __init__(self, data, batch_size: int, walk_length: int, **kwargs):
        kwargs = {k:v for k, v in kwargs.items() if k in ["num_steps", "sample_coverage", "save_dir"]}
        super().__init__(data, batch_size, walk_length, **kwargs)

    def _collate(self, data_list):
        batch = super()._collate(data_list)
        return batch, ""


        
def graph_collate(batch):
    g_list = []
    idxs = []
    if not isinstance(batch, list):
        for g in batch:
            g_list.extend(g[0])
            idxs.append(g[1])
    else:
        for d in batch:
            g_list.append(d[0])
            idxs.append(d[1])
    g_batch = Batch.from_data_list(g_list)
    idxs = torch.as_tensor(idxs)
    return g_batch, idxs



class GDataset():
    def __init__(self,):
        pass


    def init_loaders_(self,):
        "Implement this is children classes"
        pass


    def normalize_feats_(self,):
        "Implement this is children classes"
        return "x"


    def init_ds_idxs_(self, train_idxs=None, cal_idxs=None, test_idxs=None, train_test_split=None, shuffle=True, seed=1234, **kwargs):
        if (train_idxs is not None) and (cal_idxs is not None) and (test_idxs is not None):
            self.n_train = train_idxs.size(0)
            self.n_cal = cal_idxs.size(0)
            self.n_test = test_idxs.size(0)
            self.train_idxs = train_idxs
            self.cal_idxs = cal_idxs
            self.test_idxs = test_idxs
        else:
            all_idxs = torch.arange(self.n_samples)
            if shuffle: 
                fix_seed(seed)
                perm = torch.randperm(self.n_samples)
                all_idxs = all_idxs[perm]
            if train_test_split[0] + train_test_split[1] != 1.0:
                cal_per = 1 - (train_test_split[0] + train_test_split[1])
            else:
                cal_per = 0.0
            self.all_idxs = all_idxs
            self.n_train = int(self.n_samples * train_test_split[0])
            self.n_cal = int(self.n_samples * cal_per)
            self.n_test = self.n_samples - (self.n_train + self.n_cal)
            self.train_idxs = all_idxs[:int(self.n_train)]
            self.cal_idxs = all_idxs[self.n_train:self.n_train + self.n_cal]
            self.test_idxs = all_idxs[self.n_train + self.n_cal:self.n_train + self.n_cal + self.n_test]


    def initialize(self,):
        "Implement this is children classes"
        return "x"



class NDataset(GDataset):
    def __init__(self, task, **kwargs) -> None:
        super().__init__()
        self.level = "node"
        self.task = task
        self.partition_methods = ["node_degree", "homophily", "clustering_coefficient", "feature_clustering"]


    @property
    def x(self,):
        return self._data.x


    def partition_by_node_degree(self, data, n_partitions=2, **kwargs):

        num_nodes = data.x.shape[0]

        if data.is_undirected():
            edge_index = data.edge_index
        else:
            edge_index = to_undirected(data.edge_index)

        node_degrees = degree(edge_index[0], num_nodes=num_nodes)
        groups, bins = pd.qcut(node_degrees.cpu().numpy(), q=n_partitions, labels=False, retbins=True, duplicates='drop')
        n_group_samples = np.bincount(groups)

        # logger.info(f"Paritioned that data to {n_partitions} resulting in {n_group_samples} samples in the paritions for bins {bins}")

        partitions = torch.zeros((num_nodes, ), dtype=torch.long)
        for i in range(n_partitions):
            partitions[groups == i] = i

        return partitions


    def _get_k_hop_edge_index(self, edge_index, num_nodes, k):

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


    def partition_by_homophily(self, data, n_partitions=2, k=1, **kwargs):

        num_nodes = data.x.shape[0]
        y = data.y

        if data.is_undirected():
            base_edge_index = data.edge_index
        else:
            base_edge_index = to_undirected(data.edge_index)
            
        base_edge_index, _ = remove_self_loops(base_edge_index)
        edge_index = self._get_k_hop_edge_index(base_edge_index, num_nodes, k)    
        row, col = edge_index

        if self.task == 'cls':
            is_same = (y[row] == y[col]).float()
            node_metric = scatter(is_same, row, dim=0, dim_size=num_nodes, reduce='mean')

        elif self.task == 'reg':
            y = y.float()    
            diff = torch.abs(y[row] - y[col])
            mean_diff = scatter(diff, row, dim=0, dim_size=num_nodes, reduce='mean')
            node_metric = -1.0 * mean_diff

        else:
            raise ValueError("task must be 'classification' or 'regression'")

        metric_np = node_metric.cpu().numpy().flatten()
        metric_np = np.nan_to_num(metric_np, nan=0.0)

        try:
            groups, bins = pd.qcut(metric_np, q=n_partitions, labels=False, retbins=True, duplicates='drop')
        except ValueError as e:
            # Fallback if data is too skewed for qcut (e.g., all 0s)
            logger.warning(f"qcut failed, will use cut: {e}")
            groups, bins = pd.cut(metric_np, bins=n_partitions, labels=False, retbins=True)

        n_group_samples = np.bincount(groups)
        # logger.info(f"Partitioned data by {self.task} homophily (k={k}) to {n_partitions} bins: {bins}")

        partitions = torch.zeros((num_nodes, ), dtype=torch.long)
        for i in range(n_partitions):
            partitions[groups == i] = i

        return partitions


    def partition_by_clustering_coefficient(self, data, n_partitions=2, **kwargs):

        num_nodes = data.x.shape[0]
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)

        clustering_coeffs_dict = nx.clustering(G, **kwargs)
        coeffs_list = [clustering_coeffs_dict.get(i, 0.0) for i in range(num_nodes)]
        coeffs_np = np.array(coeffs_list)

        try:
            groups, bins = pd.qcut(coeffs_np, q=n_partitions, labels=False, retbins=True, duplicates='drop')
        except ValueError as e:
            logger.warning(f"qcut failed (likely identical values), falling back to cut. Error: {e}")
            groups, bins = pd.cut(coeffs_np, bins=n_partitions, labels=False, retbins=True)


        n_group_samples = np.bincount(groups)
        # logger.info(f"Partitioned data by Clustering Coefficient (NetworkX) to {len(n_group_samples)} bins: {bins}")

        partitions = torch.zeros((num_nodes, ), dtype=torch.long)
        unique_groups = np.unique(groups)
        for i in unique_groups:
            partitions[groups == i] = int(i)

        return partitions


    def partition_by_feature_clustering(self, data, n_partitions=2, **kwargs):

        features = data.x.detach().cpu().numpy()
        random_state = kwargs.get('random_state', 42)
        kmeans = KMeans(n_clusters=n_partitions, random_state=random_state, n_init=10)

        labels = kmeans.fit_predict(features)
        partitions = torch.from_numpy(labels).long()

        n_group_samples = np.bincount(labels)
        # logger.info(f"Partitioned data by Feature K-Means to {n_partitions} clusters. Sample counts: {n_group_samples}")

        return partitions


    def partitioning_func(
        self, 
        data, 
        n_partitions=2, 
        **kwargs
    ):

        if n_partitions < 1:
            return None

        partitions = []
        for m in self.partition_methods:
            partitions.append(
                getattr(self, f"partition_by_{m}")(data=data, n_partitions=n_partitions, **kwargs).reshape(-1, 1).numpy()
            )

        partitions = torch.from_numpy(np.concatenate(partitions, axis=1))

        return partitions


    def transform(self, transformation_type = "pca", transformation_config = dict(n_components = 1)):
        pass



class FromPyGNode(NDataset):
    def __init__(
        self, 
        main_dataset,
        init_config,
        name,
        task="cls",
        **kwargs
    ) -> None:
        super().__init__(task = task)
        graph = deepcopy(main_dataset[0])
        self.n_samples = graph.x.shape[0]
        self.n_feats = graph.x.shape[1]
        self.n_classes = graph.y.unique().shape[0]
        self.name = name
        self.num_nsamples = self.n_samples
        self.processed_dir = main_dataset.processed_dir

        self.init_ds_idxs_(**init_config)

        graph.train_mask = torch.as_tensor([False] * self.n_samples)
        graph.train_mask[self.train_idxs] = True

        graph.cal_mask = torch.as_tensor([False] * self.n_samples)
        graph.cal_mask[self.cal_idxs] = True

        graph.test_mask = torch.as_tensor([False] * self.n_samples)
        graph.test_mask[self.test_idxs] = True
        
        self._data = graph
        self.initialize_(**init_config)


    def collate_fn_(self, g_list):
        return Batch.from_data_list(g_list), 0

        
    def initialize_(
        self,
        cal_domain: str = "id",
        test_domain: str = "id", 
        normalize_mode = "none",
        loader_config = dict(train_loader_type="simple"),
        shuffle = True,
        n_partitions = 0, # options: [0: <no partitioning>, 2: <2 partitions>, -1: <all possible partitions>]
        **kwargs
    ) -> None:
        if shuffle: 
            fix_seed(kwargs["seed"] if "seed" in kwargs else 2411)

        self.train_mask_name = "train_mask"
        self.cal_mask_name = "cal_mask"
        self.test_mask_name = "test_mask"

        self.n_partitions = n_partitions
        self._data.partitions = self.partitioning_func(data=self._data, n_partitions=n_partitions)
        
        self._data.y = self._data.y.view(-1).float() if self.n_classes <= 2 else self._data.y.view(-1).long()
        if normalize_mode != "none":
            self._data.x, normal_params = normalize_(self._data.x, dim=0, mode=normalize_mode)

        if "transformation_type" in kwargs:
            self.transform(
                transformation_type = kwargs["transformation_type"],
                transformation_config = kwargs["transformation_config"]
            )

        num_worker = 1 if self.n_feats < 1000 else 0
        self.list_mappings = {}
        for k, v in self._data.items():
            if isinstance(v, list):
                unique_attrs = list(set(v))
                attr_mapping = dict(zip(unique_attrs, range(len(unique_attrs))))
                self.list_mappings[k] = attr_mapping
                mapped_attr = torch.as_tensor(list(map(lambda item: attr_mapping[item], v)), dtype=torch.int64)
                self._data[k] = mapped_attr

        if (loader_config["batch_size"] > self._data.x.shape[0]) or (loader_config["train_loader_type"] not in ["graphsaint", "neighbor"]):
            self.train_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=num_worker)
        else:
            if loader_config["train_loader_type"] == "graphsaint":
                pattern = "graphsaintrandomwalksampler*"
                files_to_delete = glob(os.path.join(self.processed_dir, pattern))

                for file_path in files_to_delete:
                    os.remove(file_path)
                    print(f"Deleted current processed_graphsaintrandomwalksampler: {file_path}")

                loader_config.update(save_dir = self.processed_dir)
                self.train_loader = CustomGraphSAINTLoader(
                    self._data, 
                    **loader_config
                )
            elif loader_config["train_loader_type"] == "neighbor":
                loader_config.update(input_nodes = self._data.train_mask)
                self.train_loader = CustomNeighborLoader(
                    self._data,
                    **loader_config
                )

        self.cal_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=num_worker)
        self.test_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=num_worker)



class FromPyGGraph(GDataset):
    def __init__(self, task, **kwargs) -> None:
        super(FromPyGGraph, self).__init__()
        self.level = "graph"
        self.task = task


    def gen_graph_ds(self, dataset):
        if len(dataset) == 0:
            return []
        x_idxs = dataset.slices["x"][:len(dataset)+1]
        all_graphs = []

        if self.task == "reg":
            y_dtype = torch.float
        elif self.n_classes <= 2:
            y_dtype = torch.float
        else:
            y_dtype = torch.long
            
        for i in range(x_idxs.size(0)-1):
            g = dataset[i]
            temp_g = Data(
                x = dataset._data.x[x_idxs[i]:x_idxs[i+1], :], 
                edge_index = g.edge_index, 
                y = g.y.view(-1).to(y_dtype)
            )
            if self.ne_feats > 0:
                temp_g.edge_attr = g.edge_attr
            all_graphs.append(temp_g)
        return all_graphs


    def transform(self, transformation_type = "pca", transformation_config = dict(n_components = 1)):

        if transformation_type == "pca":
            pca = PCA(**transformation_config)
            self.train_ds._data.x = torch.from_numpy(pca.fit_transform(self.train_ds._data.x.numpy()))
            self.cal_ds._data.x = torch.from_numpy(pca.transform(self.cal_ds._data.x.numpy()))
            self.test_ds._data.x = torch.from_numpy(pca.transform(self.test_ds._data.x.numpy()))
            self.n_feats = transformation_config["n_components"]
        else:
            pass


    def normalize_feats_(self, normalize_mode, **kwargs):
        self.train_ds._data.x, train_normal_params = normalize_(self.train_ds._data.x, dim=0, mode=normalize_mode)
        if self.n_cal > 0:
            self.cal_ds._data.x, _ = normalize_(self.cal_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds._data.x, _ = normalize_(self.test_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        

    def init_loaders_(self, batch_size, loader_collate = graph_collate):            
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.cal_loader = DataLoader(self.cal_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)

    
    def partition(self, dataset, **kwargs):
        pass



class FromTUDsGraph(FromPyGGraph):
    def __init__(
        self,
        main_dataset,
        init_config,
        name,
        task="cls",
        **kwargs
    ) -> None:
        super().__init__(task, **kwargs)
        self._data = deepcopy(main_dataset)
        self.n_feats = self._data.x.size(1)
        self.ne_feats = 0
        self.num_nsamples = self._data.x.size(0)
        self.num_nclass = self._data.num_node_labels
        self.n_classes = self._data.num_classes
        self.num_gsamples = len(self._data)
        self.n_samples = self.num_gsamples
        self.num_node_labels = main_dataset.num_node_labels
        self.name = name
        if self.name in ["ENZYMES", "PROTEINS", "DHFR"]:
            self.partitioning_func = self.partition_by_node_labels
            self.type = "mol"
        else:
            raise Exception("The dataset is not supported!")
        self.initialize_(**init_config)


    def partition_by_node_labels(self, partitioning_ds="test", n_partitions=2, granularity_level=0.2, **kwargs):

        if partitioning_ds == "test":
            dataset = self.test_ds
        elif partitioning_ds == "train":
            dataset = self.train_ds
        elif partitioning_ds == "cal":
            dataset = self.cal_ds
        else:
            raise Exception

        n_samples = len(dataset)
        graph_node_mask = torch.zeros((len(dataset), self.num_node_labels), dtype=torch.bool)

        for i, graph in enumerate(dataset):
            for j in range(self.num_node_labels):
                if (graph.x[:, -self.num_node_labels+j] == 1).any():
                    graph_node_mask[i, j] = True

        cls_sum = graph_node_mask.sum(dim=0)
        cls_freq = torch.stack((cls_sum, (cls_sum - n_samples).abs())).T
        min_cls, ord_idxs = cls_freq.min(dim=1).values.sort(descending=True)

        graph_node_mask = graph_node_mask[:, ord_idxs]

        min_ratio = min_cls / n_samples
        condition_idx = torch.where(min_ratio <= max(min_ratio.min(), granularity_level))[0][0]
        logger.info(
            f"Partition granularity level: {granularity_level}. Chosen level: {min_ratio[condition_idx]:.4f}. "
            f"Possible partition frequencies >> max: {min_ratio.max():.4f}, median: {min_ratio.median():.4f} and min: {min_ratio.min():.4f}"
        )

        if n_partitions == -1:
            partitions = [graph_node_mask[:, i] == 1 for i in range(self.num_node_labels)]
        else:
            partitions = [graph_node_mask[:, condition_idx], ~graph_node_mask[:, condition_idx]]

        return partitions

    
    def initialize_(
        self,
        train_idxs: torch.Tensor = None,
        cal_idxs: torch.Tensor = None,
        test_idxs: torch.Tensor = None,
        train_test_split = [0.85, 0.15],
        loader_collate = graph_collate,
        batch_size = 32, 
        normalize_mode = "none",
        shuffle = False, 
        n_partitions = 0,
        **kwargs
    ) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, cal_idxs = cal_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        self.train_ds = self._data.copy(self.train_idxs)
        self.cal_ds = self._data.copy(self.cal_idxs) if self.n_cal > 0 else []
        self.test_ds = self._data.copy(self.test_idxs) if self.n_test > 0 else []

        if (n_partitions > 0) or (n_partitions == -1):
            self.partitions = self.partitioning_func(n_partitions=2, **kwargs)
        else:
            self.partitions = None

        if normalize_mode != "none":
            self.normalize_feats_(normalize_mode)
        self.train_ds = SimpleDataset(self.gen_graph_ds(self.train_ds))
        self.cal_ds = SimpleDataset(self.gen_graph_ds(self.cal_ds))
        self.test_ds = SimpleDataset(self.gen_graph_ds(self.test_ds))
        self.init_loaders_(batch_size, loader_collate)
        


class FromGOODNode(NDataset):
    def __init__(
        self,
        main_dataset,
        init_config,
        name,
        **kwargs
    ) -> None:
        task = "reg" if main_dataset[0].task == "Regression" else "cls"
        super().__init__(task, **kwargs)
        self._data = main_dataset[0][0]
        meta_info = main_dataset[1]
        assert self._data.x.shape[1] == meta_info.dim_node
        self.n_feats = meta_info.dim_node
        self.ne_feats = meta_info.dim_edge
        self.n_envs = meta_info.num_envs
        self.processed_dir = main_dataset[0].processed_dir

        if self.task == "reg":
            self.n_classes = meta_info.num_classes
        else:
            self.num_nsamples = self._data.x.shape[0]
            self.n_classes = main_dataset[0].num_classes

        self.name = name
        self.type = meta_info.dataset_type

        self.initialize_(**init_config)


    def collate_fn_(self, g_list):
        return Batch.from_data_list(g_list), 0


    def initialize_(
        self,
        cal_domain: str = "id",
        test_domain: str = "id", 
        normalize_mode = "none",
        loader_config = dict(train_loader_type="simple"),
        shuffle = True,
        n_partitions = 0, # options: [0: <no partitioning>, 2: <2 partitions>, -1: <all possible partitions>]
        **kwargs
    ) -> None:
        if shuffle: 
            fix_seed(kwargs["seed"] if "seed" in kwargs else 2411)

        self.train_mask_name = "train_mask"
        self.cal_mask_name = "val_mask" if cal_domain == "ood" else "id_val_mask"
        self.test_mask_name = "test_mask" if test_domain == "ood" else "id_test_mask"

        self.init_ds_idxs_(
            train_idxs = getattr(self._data, self.train_mask_name).nonzero().T.view(-1),
            cal_idxs = getattr(self._data, self.cal_mask_name).nonzero().T.view(-1),
            test_idxs = getattr(self._data, self.test_mask_name).nonzero().T.view(-1),
        )

        self.n_partitions = n_partitions
        self._data.partitions = self.partitioning_func(data=self._data, n_partitions=n_partitions)
        self._data.y = self._data.y.view(-1).float() if self.n_classes <= 2 else self._data.y.view(-1).long()
        if normalize_mode != "none":
            self._data.x, normal_params = normalize_(self._data.x, dim=0, mode=normalize_mode)

        if "transformation_type" in kwargs:
            self.transform(
                transformation_type = kwargs["transformation_type"],
                transformation_config = kwargs["transformation_config"]
            )

        self.list_mappings = {}
        for k, v in self._data.items():
            if isinstance(v, list):
                unique_attrs = list(set(v))
                attr_mapping = dict(zip(unique_attrs, range(len(unique_attrs))))
                self.list_mappings[k] = attr_mapping
                mapped_attr = torch.as_tensor(list(map(lambda item: attr_mapping[item], v)), dtype=torch.int64)
                self._data[k] = mapped_attr

        if (loader_config["batch_size"] > self._data.x.shape[0]) or (loader_config["train_loader_type"] not in ["graphsaint", "neighbor"]):
            self.train_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=1)
        else:
            if loader_config["train_loader_type"] == "graphsaint":
                pattern = "graphsaintrandomwalksampler*"
                files_to_delete = glob(os.path.join(self.processed_dir, pattern))

                for file_path in files_to_delete:
                    os.remove(file_path)
                    print(f"Deleted current processed_graphsaintrandomwalksampler: {file_path}")

                loader_config.update(save_dir = self.processed_dir)
                self.train_loader = CustomGraphSAINTLoader(
                    self._data, 
                    **loader_config
                )
            elif loader_config["train_loader_type"] == "neighbor":
                loader_config.update(input_nodes = self._data.train_mask)
                self.train_loader = CustomNeighborLoader(
                    self._data,
                    **loader_config
                )

        self.cal_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=1)
        self.test_loader = DataLoader([self._data], batch_size=1, collate_fn=self.collate_fn_, num_workers=1)



class SingleGraphDataset(InMemoryDataset):
    def __init__(self, graph, seed=1234, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        
        data, slices = self.collate([graph])
        
        self._data = data
        self.slices = slices

    def __len__(self):
        return 1

    def get(self, idx):
        return self._data



class FromCFGNNNode(FromPyGNode):
    def __init__(
        self,
        main_dir,
        init_config,
        name,
        **kwargs
    ) -> None:
        
        edges = pd.read_csv(os.path.join(main_dir, name + '_edge_list.txt'), sep = '\t', header = None) -1
        feats = pd.read_csv(os.path.join(main_dir, name + '_features.txt'), sep = '\t', header = None)
        labels = pd.read_csv(os.path.join(main_dir, name + '_labels.txt'), sep = '\t', header = None)

        edge_index = torch.tensor(edges[[0, 1]].values.T, dtype=torch.long)
        x = torch.tensor(feats.values, dtype=torch.float)
        y = torch.tensor(labels.values, dtype=torch.float)

        self.level = "node"
        self.task = "reg"
        self.partition_methods = ["node_degree", "homophily", "clustering_coefficient", "feature_clustering"]
        self.num_nsamples, self.n_feats = x.shape
        self.n_samples = self.num_nsamples
        self.name = name
        self.n_classes = 1
        self.processed_dir = os.path.join(main_dir, name)

        graph = Data(x=x, edge_index=edge_index, y=y)
        self.init_ds_idxs_(train_test_split = init_config["train_test_split"], seed = init_config["seed"])

        graph.train_mask = torch.as_tensor([False] * self.n_samples)
        graph.train_mask[self.train_idxs] = True

        graph.cal_mask = torch.as_tensor([False] * self.n_samples)
        graph.cal_mask[self.cal_idxs] = True

        graph.test_mask = torch.as_tensor([False] * self.n_samples)
        graph.test_mask[self.test_idxs] = True
        
        self._data = graph

        self.initialize_(**init_config)



class FromGOODGraph(FromPyGGraph):
    def __init__(
        self,
        main_dataset,
        init_config,
        name,
        **kwargs
    ) -> None:
        task = "reg" if main_dataset[0]["task"] == "Regression" else "cls"
        super().__init__(task, **kwargs)
        self._data = main_dataset[0]
        meta_info = main_dataset[1]
        assert self._data["train"].x.shape[1] == meta_info.dim_node
        self.n_feats = meta_info.dim_node
        self.ne_feats = meta_info.dim_edge
        self.n_envs = meta_info.num_envs
        
        if self.task == "reg":
            self.n_classes = meta_info.num_classes
        else:
            self.num_nsamples = self._data["train"].x.shape[0]
            self.n_classes = self._data["train"].num_classes

        self.name = name
        self.type = meta_info.dataset_type
        if self.name in ["GOODHIV", "GOODZINC"]:
            self.partitioning_func = self.partition_by_atoms
        elif self.name in ["GOODCMNIST"]:
            self.partitioning_func = self.partition_by_node_degree
        elif self.name in ["GOODMotif"]:
            self.partitioning_func = self.partition_by_edge_density
        else:
            raise Exception("The dataset is not supported!")
        self.initialize_(**init_config)


    def partition_by_atoms(self, partitioning_ds="test", n_partitions=2, granularity_level=0.2, **kwargs):

        if partitioning_ds == "test":
            dataset = self.test_ds
        elif partitioning_ds == "train":
            dataset = self.train_ds
        elif partitioning_ds == "cal":
            dataset = self.cal_ds
        else:
            raise Exception

        n_samples = len(dataset)
        atoms = dataset.x.tolist()

        unique_atoms = []
        for row in atoms:
            if row in unique_atoms:
                continue
            else:
                unique_atoms.append(row)

        graph_atom_mask = []
        for g in dataset:
            graph_atom_mask.append([])
            graph_atoms = g.x.tolist()
            for atom in unique_atoms:
                if atom in graph_atoms:
                    graph_atom_mask[-1].append(True)
                else:
                    graph_atom_mask[-1].append(False)

        graph_atom_mask = torch.as_tensor(graph_atom_mask)
        n_unique_atoms = graph_atom_mask.shape[1]
        cls_sum = graph_atom_mask.sum(dim=0)
        cls_freq = torch.stack((cls_sum, (cls_sum - n_samples).abs())).T
        min_cls, ord_idxs = cls_freq.min(dim=1).values.sort(descending=True)

        unique_atoms = [unique_atoms[i] for i in ord_idxs]
        graph_atom_mask = graph_atom_mask[:, ord_idxs]

        min_ratio = min_cls / n_samples
        condition_idx = torch.where(min_ratio <= max(min_ratio.min(), granularity_level))[0][0]
        logger.info(
            f"Partition granularity level: {granularity_level}. Chosen level: {min_ratio[condition_idx]:.4f}. "
            f"Possible partition frequencies >> max: {min_ratio.max():.4f}, median: {min_ratio.median():.4f} and min: {min_ratio.min():.4f}"
        )

        if n_partitions == -1:
            partitions = [graph_atom_mask[:, i] for i in range(n_unique_atoms)]
        elif n_partitions == 2:
            partitions = [graph_atom_mask[:, condition_idx], ~graph_atom_mask[:, condition_idx]]
        else:
            raise Exception(f"Number of partitions: {n_partitions}. Partitioning to combination of multiple atoms is not supported!")

        return partitions


    def partition_by_edge_density(self, partitioning_ds="test", n_partitions=2, **kwargs):

        if partitioning_ds == "test":
            dataset = self.test_ds
        elif partitioning_ds == "train":
            dataset = self.train_ds
        elif partitioning_ds == "cal":
            dataset = self.cal_ds
        else:
            raise Exception

        densities = dataset.slices["edge_index"].diff()
        sorted_densities, sorted_idxs = densities.sort()
        density_cdf = (torch.ones_like(sorted_densities) / sorted_densities.shape[0]).cumsum(dim=0)

        quantile_threshs = torch.linspace(0, 1, n_partitions+1)[1:-1]
        quantiles = densities[sorted_idxs[
                [(density_cdf >= q).nonzero().T[0, 0].item() for q in quantile_threshs]
            ]]
            
        if n_partitions == -1:
            quantiles = torch.cat((quantiles, torch.tensor(torch.inf).view(-1)))
            partitions = []
            min_v = densities.min() - 1
            for i in range(n_partitions):
                partitions += [(min_v < densities) & (densities <= quantiles[i])]
                min_v = quantiles[i]
        elif n_partitions == 2:
            partitions = [densities <= quantiles[0], densities > quantiles[0]]
        else:
            raise Exception(f"Number of partitions: {n_partitions}. Partitioning to combination of multiple atoms is not supported!")

        return partitions


    def partition_by_node_degree(self, partitioning_ds="test", n_partitions=2, **kwargs):

        if partitioning_ds == "test":
            dataset = self.test_ds
        elif partitioning_ds == "train":
            dataset = self.train_ds
        elif partitioning_ds == "cal":
            dataset = self.cal_ds
        else:
            raise Exception

        edge_counter = torch.zeros((dataset.x.shape[0],))
        edge_counter.scatter_add_(0, dataset.edge_index[0, :], torch.ones((dataset.edge_index.shape[1],)))

        quantile_threshs = torch.linspace(0, 1, n_partitions+1)[1:-1]
        graph_densities = edge_counter.view(-1, dataset[0].x.shape[0]).mean(dim=1)
        quantiles = graph_densities.quantile(quantile_threshs)

        if n_partitions == -1:
            quantiles = torch.cat((quantiles, torch.tensor(torch.inf).view(-1)))
            partitions = []
            min_v = graph_densities.min() - 1
            for i in range(n_partitions):
                partitions += [(min_v < graph_densities) & (graph_densities <= quantiles[i])]
                min_v = quantiles[i]
        elif n_partitions == 2:
            partitions = [graph_densities <= quantiles[0], graph_densities > quantiles[0]]
        else:
            raise Exception(f"Number of partitions: {n_partitions}. Partitioning to combination of multiple atoms is not supported!")

        return partitions


    def initialize_(
        self,
        cal_domain: str = "id",
        test_domain: str = "id",
        loader_collate = graph_collate,
        batch_size = 32, 
        normalize_mode = "none",
        shuffle = True,
        n_partitions = 0, # options: [0: <no partitioning>, 2: <2 partitions>, -1: <all possible partitions>]
        **kwargs
    ) -> None:
        if shuffle: 
            fix_seed(kwargs["seed"] if "seed" in kwargs else 2411)

        self.train_mask_name = "train_mask"
        self.cal_mask_name = "val_mask"
        self.test_mask_name = "test_mask"

        self.train_ds = self._data["train"]
        self.cal_ds = self._data["val"] if cal_domain == "ood" else self._data["id_val"]
        self.test_ds = self._data["test"] if test_domain == "ood" else self._data["id_test"]

        # self.train_ds = self.train_ds[:1000]
        # self.cal_ds = self.cal_ds[:1000]
        # self.test_ds = self.test_ds[:1000]

        self.n_train = len(self.train_ds)
        self.n_cal = len(self.cal_ds)
        self.n_test = len(self.test_ds)
        self.train_idxs = torch.arange(self.n_train)
        self.cal_idxs = torch.arange(self.n_cal)
        self.test_idxs = torch.arange(self.n_test)

        if (n_partitions > 0) or (n_partitions == -1):
            self.partitions = self.partitioning_func(n_partitions = n_partitions, **kwargs)
        else:
            self.partitions = None

        if normalize_mode != "none":
            self.normalize_feats_(normalize_mode)

        if "transformation_type" in kwargs:
            self.transform(
                transformation_type = kwargs["transformation_type"],
                transformation_config = kwargs["transformation_config"]
            )

        self.train_ds = SimpleDataset(self.gen_graph_ds(self.train_ds))
        self.cal_ds = SimpleDataset(self.gen_graph_ds(self.cal_ds))
        self.test_ds = SimpleDataset(self.gen_graph_ds(self.test_ds))
        self.init_loaders_(batch_size, loader_collate)



class GenDataset(object):
    def __init__(self,) -> None:
        pass

    def get_graph_dataset(
        self,
        dataset_name,
        store_to_path = "./data",
        train_test_split = [0.8, 0.2],
        batch_size = 32,
        normal_mode = "normal",
        data_config = None,
        n_partitions = 2, 
        seed = 2411,
        transformation_type = "nothing",
        transformation_config = {},
        report_ds_stats = True,
        encoder_config = None,
        **kwargs
    ):

        if dataset_name in ["ENZYMES", "PROTEINS", "DHFR"]:

            dataset = FromTUDsGraph(
                main_dataset = TUDataset(
                    root = store_to_path,
                    name = dataset_name,
                    use_node_attr = data_config["node_attributes"]
                ),
                init_config = dict(
                    train_test_split = train_test_split,
                    batch_size = batch_size,
                    normalize_mode = normal_mode,
                    shuffle = True,
                    n_partitions = n_partitions,
                    seed = seed,
                    **kwargs
                ),
                name = dataset_name
            )

        elif dataset_name in ["Cora_ML", "CiteSeer", "DBLP", "PubMed", "Cora", "Computers", "Photo", "Physics", "CS"]:

            if dataset_name in ["Cora_ML", "CiteSeer", "DBLP", "PubMed", "Cora"]:
                ds_class = CitationFull
            elif dataset_name in ["Computers", "Photo"]:
                ds_class = Amazon
            elif dataset_name in ["Physics", "CS"]:
                ds_class = Coauthor
            else:
                raise Exception("Dataset nor supported")

            org_dataset = ds_class(
                root = store_to_path,
                name = dataset_name,
            )

            dataset = FromPyGNode(
                main_dataset = org_dataset,
                init_config = dict(
                    cal_domain = data_config["cal_domain"],
                    test_domain = data_config["test_domain"],
                    train_test_split = train_test_split,
                    batch_size = batch_size,
                    normalize_mode = normal_mode,
                    shuffle = True,
                    n_partitions = n_partitions,
                    seed = seed,
                    transformation_type = transformation_type,
                    transformation_config = transformation_config,
                    **kwargs
                ),
                name = dataset_name
            )

        elif dataset_name in ["GOODHIV", "GOODCMNIST", "GOODMotif", "GOODZINC"]:
            
            if dataset_name == "GOODHIV":
                ds_class = GOODHIV
            elif dataset_name == "GOODCMNIST":
                ds_class = GOODCMNIST
            elif dataset_name == "GOODMotif":
                ds_class = GOODMotif
            elif dataset_name == "GOODZINC":
                ds_class = GOODZINC
            
            org_dataset = ds_class.load(
                store_to_path, 
                domain = data_config["domain"], 
                shift = data_config["shift"], 
                generate = data_config["generate"]
            )

            dataset = FromGOODGraph(
                main_dataset = org_dataset,
                init_config = dict(
                    cal_domain = data_config["cal_domain"],
                    test_domain = data_config["test_domain"],
                    batch_size = batch_size,
                    normalize_mode = normal_mode,
                    shuffle = True,
                    n_partitions = n_partitions,
                    seed = seed,
                    transformation_type = transformation_type,
                    transformation_config = transformation_config,
                    **kwargs
                ),
                name = dataset_name
            )
            
        elif dataset_name in ["GOODTwitch", "GOODWebKB", "GOODCBAS"]:
            
            if dataset_name == "GOODTwitch":
                ds_class = GOODTwitch
            elif dataset_name == "GOODWebKB":
                ds_class = GOODWebKB
            elif dataset_name == "GOODCBAS":
                ds_class = GOODCBAS

            org_dataset = ds_class.load(
                store_to_path, 
                domain = data_config["domain"], 
                shift = data_config["shift"], 
                generate = data_config["generate"]
            )

            dataset = FromGOODNode(
                main_dataset = org_dataset,
                init_config = dict(
                    cal_domain = data_config["cal_domain"],
                    test_domain = data_config["test_domain"],
                    batch_size = batch_size,
                    normalize_mode = normal_mode,
                    shuffle = True,
                    n_partitions = n_partitions,
                    seed = seed,
                    transformation_type = transformation_type,
                    transformation_config = transformation_config,
                    **kwargs
                ),
                name = dataset_name
            )
        elif dataset_name in ["Anaheim", "ChicagoSketch", "twitch_PTBR", "county_education_2012", "county_election_2016", "county_income_2012", "county_unemployment_2012"]:

            dataset = FromCFGNNNode(
                main_dir = store_to_path, 
                init_config = dict(
                    cal_domain = data_config["cal_domain"],
                    test_domain = data_config["test_domain"],
                    train_test_split = train_test_split,
                    batch_size = batch_size,
                    normalize_mode = normal_mode,
                    shuffle = True,
                    n_partitions = n_partitions,
                    seed = seed,
                    transformation_type = transformation_type,
                    transformation_config = transformation_config,
                    **kwargs
                ),
                name = dataset_name, 
            )

        embeds_dists = None
        # if report_ds_stats and (encoder_config is not None):
        #     embeds_dists = dataset_embeds_and_dists(dataset, encoder_config | {"level": dataset.level})

        return dataset, embeds_dists

