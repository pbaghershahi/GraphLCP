"""
Base classes for Graph Neural Networks
"""
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch import Tensor
from munch import Munch
from torch.nn import Identity

from typing import Callable, Optional

import torch_geometric.nn as gnn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

import ipdb



x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding



class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding



class GNNBasic(torch.nn.Module):
    r"""
    Base class for graph neural networks

    Args:
        *args (list): argument list for the use of :func:`~arguments_read`
        **kwargs (dict): key word arguments for the use of :func:`~arguments_read`

    """
    def __init__(self, config, *args, **kwargs):
        super(GNNBasic, self).__init__()
        self.config = config

    def arguments_read(self, *args, **kwargs):
        r"""
        It is an argument reading function for diverse model input formats.
        Support formats are:
        ``model(x, edge_index)``
        ``model(x, edge_index, batch)``
        ``model(data=data)``.

        Notes:
            edge_weight is optional for node prediction tasks.

        Args:
            *args: [x, edge_index, [batch]]
            **kwargs: data, [edge_weight]

        Returns:
            Unpacked node features, sparse adjacency matrices, batch indicators, and optional edge weights.
        """

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.config.dataset_level != 'node':
            # --- Maybe batch size --- Reason: some method may filter graphs leading inconsistent of batch size
            batch_size: int = kwargs.get('batch_size') or (batch[-1].item() + 1)

        if self.config.dataset_level == 'node':
            edge_weight = kwargs.get('edge_weight')
            return x, edge_index, edge_weight, batch
        elif self.config.dim_edge:
            edge_attr = data.edge_attr
            return x, edge_index, edge_attr, batch, batch_size

        return x, edge_index, batch, batch_size

    def probs(self, *args, **kwargs):
        # nodes x classes
        return self(*args, **kwargs).softmax(dim=1)



class BasicEncoder(torch.nn.Module):
    r"""
        Base GNN feature encoder.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dataset_level`, :obj:`config.global_pool`, :obj:`config.dropout_rate`)

        .. code-block:: python

            config = munchify({model: {dim_hidden: int(300),
                               model_layer: int(5),
                               dataset_level: str('node'),
                               global_pool: str('mean'),
                               dropout_rate: float(0.5),}
                               })


    """

    def __init__(self, config, **kwargs):
        if type(self).mro()[type(self).mro().index(__class__) + 1] is torch.nn.Module:
            super(BasicEncoder, self).__init__()
        else:
            super(BasicEncoder, self).__init__(config)
        num_layer = config.model_layer

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if kwargs.get('no_bn'):
            self.batch_norm1 = Identity()
            self.batch_norms = [
                Identity()
                for _ in range(num_layer - 1)
            ]
        else:
            self.batch_norm1 = nn.BatchNorm1d(config.dim_hidden)
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(config.dim_hidden)
                for _ in range(num_layer - 1)
            ])
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.dropout_rate)
            for _ in range(num_layer - 1)
        ])
        if config.dataset_level == 'node':
            self.readout = lambda x, batch: x
        elif config.global_pool == 'mean':
            self.readout = gnn.global_mean_pool
        else:
            self.readout = gnn.global_max_pool

        self.device = config.device



class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.num_classes`)
    """
    def __init__(self, config):

        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.dim_hidden, 2 * config.dim_ffn), nn.BatchNorm1d(2 * config.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.dim_ffn, config.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.dim_hidden, config.num_classes)]
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)



class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.num_classes`, :obj:`config.dataset_type`)
    """

    def __init__(self, config):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None
        self.device = config.device

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out



class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.dataset_type`)
    """
    def __init__(self, config, **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        num_layer = config.model_layer
        if config.dataset_type == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.edge_feat = False
        self.device = config.device

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout



class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`)
    """

    def __init__(self, config, *args, **kwargs):

        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.dim_hidden)

        if kwargs.get('without_embed'):
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.dim_hidden, 2 * config.dim_hidden),
                                               nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.dim_hidden, config.dim_hidden)))
        else:
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.dim_node, 2 * config.dim_hidden),
                                               nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.dim_hidden, config.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                gnn.GINConv(nn.Sequential(nn.Linear(config.dim_hidden, 2 * config.dim_hidden),
                                      nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.dim_hidden, config.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )
        self.device = config.device

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout



class GINMolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`)
    """

    def __init__(self, config, **kwargs):
        super(GINMolEncoder, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        num_layer = config.model_layer
        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.dim_hidden)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(config.dim_hidden, 2 * config.dim_hidden),
                                            nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU(),
                                            nn.Linear(2 * config.dim_hidden, config.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(config.dim_hidden, 2 * config.dim_hidden),
                                       nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU(),
                                       nn.Linear(2 * config.dim_hidden, config.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout



class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.bone_encoder = BondEncoder(in_channels)
        # if edge_dim is not None:
        #     self.lin = Linear(edge_dim, in_channels)
        #     # self.lin = Linear(edge_dim, config.dim_hidden)
        # else:
        #     self.lin = None
        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if self.bone_encoder:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class vGIN(GNNBasic):
    r"""
        The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.num_classes`, :obj:`config.dataset_type`, :obj:`config.dropout_rate`)
    """

    def __init__(self, config):
        super(vGIN, self).__init__(config)
        self.feat_encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The vGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class vGINFeatExtractor(GINFeatExtractor):
    r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.dataset_type`, :obj:`config.dropout_rate`)
            **kwargs: `without_readout` will output node features instead of graph features.
    """
    def __init__(self, config, **kwargs):
        super(vGINFeatExtractor, self).__init__(config)
        num_layer = config.model_layer
        if config.dataset_type == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False


class VirtualNodeEncoder(torch.nn.Module):
    r"""
        The virtual node feature encoder for vGIN.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.dropout_rate`)
    """
    def __init__(self, config, *args, **kwargs):
        super(VirtualNodeEncoder, self).__init__()
        self.virtual_node_embedding = nn.Embedding(1, config.dim_hidden)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(config.dim_hidden, 2 * config.dim_hidden),
                 nn.BatchNorm1d(2 * config.dim_hidden), nn.ReLU()] +
                [nn.Linear(2 * config.dim_hidden, config.dim_hidden),
                 nn.BatchNorm1d(config.dim_hidden), nn.ReLU(),
                 nn.Dropout(config.dropout_rate)]
        ))
        self.virtual_pool = gnn.global_add_pool


class vGINEncoder(GINEncoder, VirtualNodeEncoder):
    r"""
    The vGIN encoder for non-molecule data, using the :class:`~vGINConv` operator for message passing.

    Args:
        config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.dropout_rate`)
    """

    def __init__(self, config, **kwargs):
        super(vGINEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The vGIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout



class vGINMolEncoder(GINMolEncoder, VirtualNodeEncoder):
    r"""The vGIN encoder for molecule data, using the :class:`~vGINEConv` operator for message passing.

        Args:
            config (config): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dropout_rate`)
    """

    def __init__(self, config, **kwargs):
        super(vGINMolEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The vGIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))

        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout



class GCN(GNNBasic):
    r"""
    The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`, :obj:`config.num_classes`)
    """

    def __init__(self, config):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out



class GCNFeatExtractor(GNNBasic):
    r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`)
    """
    def __init__(self, config):
        super(GCNFeatExtractor, self).__init__(config)
        self.encoder = GCNEncoder(config)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        out_readout = self.encoder(x, edge_index, edge_weight, batch)
        return out_readout



class GCNEncoder(BasicEncoder):
    r"""
    The GCN encoder using the :class:`~GCNConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.dim_hidden`, :obj:`config.model_layer`, :obj:`config.dim_node`)
    """

    def __init__(self, config):
        super(GCNEncoder, self).__init__(config)
        num_layer = config.model_layer

        self.conv1 = GCNConv(config.dim_node, config.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(config.dim_hidden, config.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_weight, batch):
        r"""
        The GCN encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout



class GCNConv(gnn.GCNConv):
    r"""The graph convolutional operator from the `"Semi-supervised
        Classification with Graph Convolutional Networks"
        <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        *args (list): argument list for the use of arguments_read.
        **kwargs (dict): Additional key word arguments for the use of arguments_read.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.__explain__ = False
        self.__edge_mask__ = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        r"""
        The GCN graph convolutional operator.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights

        Returns (Tensor):
            node feature representations

        """

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self._collect(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:

                edge_mask = self.layer_edge_mask.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            return self.update(out, **update_kwargs)



class CustomGCN(GCN):
    def __init__(self, config):
        super().__init__(config)
        self.device = config["device"]
    

    def forward(self, data, decoder = True):

        x = self.feat_encoder(data=data)

        if not decoder:
            return "scores", x
        scores = self.classifier(x)
        return scores, x



class CustomvGIN(vGIN):
    def __init__(self, config):
        super().__init__(config)
        self.device = config["device"]


    def forward(self, data, decoder = True):

        x = self.feat_encoder(data=data)

        if not decoder:
            return "scores", x
        scores = self.classifier(x)
        return scores, x