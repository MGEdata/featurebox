"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Linear, Parameter
from torch.nn import Module, ModuleList
from torch_geometric.nn import CGConv, MessagePassing
from torch_geometric.typing import PairTensor, OptTensor, Adj, Size
from torch_sparse import SparseTensor

from featurebox.models_geo.basemodel import BaseCrystalModel
from featurebox.models_geo.general import collect_edge_attr_jump, lift_jump_index_select


class SLayer(nn.Module):
    def __init__(self, r_index_num=0, r_cs=0.3, r_c=6.0):
        super().__init__()
        self.r_index_num = r_index_num
        self.r_cs = r_cs
        self.r_c = r_c

    def forward(self, x):
        r_mark = x[:, self.r_index_num]
        r_m1 = r_mark < self.r_cs
        r_m3 = r_mark > self.r_c
        r_m2 = ~(r_m3 & r_m1)
        x[r_m1] = 1 / x[r_m1]
        x[r_m2] = 1 / x[r_m2] * (0.5 * torch.cos(torch.pi * (x[r_m2] - self.r_cs) / (self.r_c - self.r_cs)) + 0.5)
        x[r_m3] = 0
        return x


# nn.Embedding
class GLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ):
        super().__init__()
        device = None
        dtype = None
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        self.gl = nn.Sequential(Linear(num_embeddings, num_embeddings),
                                nn.Softplus(),
                                Linear(num_embeddings, num_embeddings))

    def forward(self, x_j):
        # emb = F.embedding(edge_index_i, self.weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        g = self.gl(x_j)
        return g


class RotNet(MessagePassing):

    def __init__(self, node_hidden_channels, num_node_interaction_channels, **kwargs):
        super(RotNet, self).__init__(aggr="mean",
                                     flow="source_to_target", node_dim=-2,
                                     decomposed_layers=1)
        self.sl = SLayer(r_index_num=0, r_cs=0.3, r_c=6.0)
        self.gl = GLayer(num_embeddings=16, embedding_dim=100)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_weight=None,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        # if isinstance(x, Tensor):
        #     x: PairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, size=size, )
        return out

    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)

    def message_and_aggregate(self, adj_t, x_j) -> Tensor:
        x_j = self.sl(x_j)
        G = self.gl(x_j)
        # 分开做下面的部分。
        t = torch.matmul(G.T, x_j)
        t = torch.matmul(t, x_j.T)
        t = torch.matmul(t, G)
        return t


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, node_hidden_channels=64, num_edge_gaussians=None, num_node_interaction_channels=64,
                 n_conv=1, **kwargs):
        super(_Interactions, self).__init__()

        short_len = 1  # not gauss
        assert short_len == 1, "keep 1 for sparse tensor"

        self.lin0 = Linear(node_hidden_channels, num_node_interaction_channels)
        self.short = Linear(num_edge_gaussians, short_len)
        self.conv = RotNet(100, 100)

        self.n_conv = n_conv

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.short.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.softplus(self.lin0(h))
        edge_attr = F.softplus(self.short(edge_attr))

        out = self.conv(x=out, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

        return out


class RotateNet(BaseCrystalModel):
    """
    CrystalGraph.
    """

    def __init__(self, *args, num_edge_features=3,num_node_interaction_channels=16, num_node_hidden_channels=8,
                 **kwargs):
        super(RotateNet, self).__init__(*args,
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
