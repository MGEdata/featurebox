"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch.nn import Module, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import PairTensor, OptTensor, Adj, Size

from featurebox.models_geo.basemodel import BaseCrystalModel
from featurebox.models_geo.general import collect_edge_attr_jump, lift_jump_index_select

class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F_{t})` if bipartite
    """
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out = out + x[1]
        return out

    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, node_hidden_channels=64, num_node_interaction_channels=64, num_edge_features=3,
                 n_conv=1, **kwargs):
        super(_Interactions, self).__init__()

        self.lin0 = Linear(node_hidden_channels, num_node_interaction_channels)
        self.conv = ModuleList()

        for _ in range(n_conv):
            cg = CGConv(channels=num_node_interaction_channels, dim=num_edge_features,
                           aggr='add', batch_norm=True,
                           bias=True, )
            self.conv.append(cg)

        self.n_conv = n_conv

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.short.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.softplus(self.lin0(h))

        for convi in self.conv:
            out = out + convi(x=out, edge_index=edge_index, edge_attr=edge_attr)

        return out


class CrystalGraphConvNet(BaseCrystalModel):
    """
    CrystalGraph.
    """

    def __init__(self, *args, num_edge_features=3, num_node_interaction_channels=16,
                 num_node_hidden_channels=8,
                 **kwargs):
        super(CrystalGraphConvNet, self).__init__(*args,
                                                  num_edge_features=num_edge_features,
                                                  num_node_interaction_channels=num_node_interaction_channels,
                                                  num_node_hidden_channels=num_node_hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.num_node_hidden_channels,
                                          self.num_node_interaction_channels,
                                          num_edge_features=self.num_edge_features,
                                          n_conv=self.num_interactions,
                                          kwargs=self.interaction_kwargs)
