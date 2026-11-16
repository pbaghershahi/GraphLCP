import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, GCN, WLConvContinuous, global_mean_pool, global_add_pool, global_max_pool
from typing import List
from good_models import CustomGCN, CustomvGIN
from utils import *
import ipdb


class GNN(nn.Module):
    def __init__(
            self, 
            gnn_type,
            in_channels, 
            hidden_channels, 
            out_channels, 
            gnn_num_hid_layers = 2,
            device = "cuda",
            pooling = None,
            decoder = "gnn",
            with_nonlinearity = True,
            with_bn = False,
            in_dropout = 0.4,
            hid_dropout = 0.4,
            out_dropout = 0.4,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.gnn_type = gnn_type
        self.num_gnn_layers = gnn_num_hid_layers
        self.device = device
        self.with_bn = with_bn
        self.with_nonlinearity = with_nonlinearity
        self.decoder_type = decoder
        self.in_dropout = in_dropout
        self.hid_dropout = hid_dropout
        self.out_dropout = out_dropout

        self.gnn_layers = nn.ModuleList([self.get_gnn_layer(gnn_type, in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])

        if decoder == "gnn":
            gnn_num_hid_layers -= 1

        for i in range(1, gnn_num_hid_layers):
            self.gnn_layers.append(self.get_gnn_layer(gnn_type, hidden_channels, hidden_channels))
            if self.with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        if decoder == "gnn":
            if self.with_bn:
                self.bns.append(nn.BatchNorm1d(out_channels))
            self.decoder = self.get_gnn_layer(gnn_type, hidden_channels, out_channels)
        else:
            self.decoder = nn.Linear(hidden_channels, out_channels)

        self.level = "node" if pooling is None else "graph"

        if self.level == "graph":
            self.pooling_func = getattr(torch_geometric.nn, f"global_{pooling}_pool")
        else:
            self.pooling_func = lambda x, batch: x


    def get_gnn_layer(self, gnn_type, in_channels, out_channels):

        match gnn_type:
            case "gcn":
                return GCNConv(in_channels, out_channels)
            case "gat":
                return GATConv(in_channels, out_channels)
            case "gin":
                return GINConv(nn.Sequential(nn.Linear(in_channels, out_channels)))
            case "sage":
                return SAGEConv(in_channels, out_channels)
            case _:
                raise Exception("The model is not implemented!")
    

    def forward(self, data=None, x=None, edge_index=None, batch=None, decoder = True):

        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.clone()
        if self.in_dropout > 0:
            x = F.dropout(x, p=self.in_dropout, training=self.training)
        
        n_gnn_layers = len(self.gnn_layers)
        
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)

            if self.with_bn and i < n_gnn_layers - 1:
                x = self.bns[i](x)
            
            if self.with_nonlinearity:
                x = F.relu(x)

            if i < n_gnn_layers -1 and self.hid_dropout > 0:
                x = F.dropout(x, p=self.hid_dropout, training=self.training)

        if not decoder:
            return None, x

        x = self.pooling_func(x, batch)
        if self.out_dropout > 0:
            scores = F.dropout(x, p=self.out_dropout, training=self.training)
        else:
            scores = x

        if self.decoder_type == "gnn":
            scores = self.decoder(scores, edge_index)
        else:
            scores = self.decoder(scores)
        
        return scores, x



class NonParamatericEncoder(nn.Module):
    def __init__(
            self,
            num_itrs = 2,
            pooling = "mean",
            level = "graph",
            device = "cuda",
            *args, 
            **kwargs
        ) -> None:
        super(NonParamatericEncoder, self).__init__(*args, **kwargs)
        self.num_itrs = num_itrs
        self.device = device

        self.enc_layers = nn.ModuleList([WLConvContinuous() for _ in range(self.num_itrs)])
        match pooling:
            case "max":
                self.pooling = global_max_pool
            case "add":
                self.pooling = global_add_pool
            case _:
                self.pooling = global_mean_pool

        self.level = level
    
    
    def forward(self, graph_batch=None, x=None, edge_index=None, batch=None):

        if graph_batch is not None:
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        x_concat = [x]
        for i, layer in enumerate(self.enc_layers):
            x = layer(x, edge_index)
            x_concat.append(x)
        x_concat = torch.cat(x_concat, dim=1)

        if self.level == "node":
            return x_concat
        
        x = self.pooling(x_concat, batch)
        
        return x



def get_model(config, dataset):

    if dataset.name.lower().startswith("good"):
        if dataset.level == "node":
            model = CustomGCN(config)
        else:
            model = CustomvGIN(config)
    else:
        model = GNN(**config)
    
    return model