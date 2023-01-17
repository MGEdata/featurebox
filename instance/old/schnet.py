"""
Note:
    This is a reconstitution version from  pytorch_geometric/ torch_geometric / nn / models_geo / schnet.py.
"""

from math import pi as PI

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor

from featurebox.models_geo.basemodel import BaseCrystalModel, ShiftedSoftplus
from featurebox.models_geo.general import collect_edge_attr_jump, lift_jump_index_select


class SchNet(BaseCrystalModel):
    """
    SchNet.
    """

    def __init__(self, *args, **kwargs):
        super(SchNet, self).__init__(*args, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _InteractionBlockLoop(self.num_node_hidden_channels, self.num_edge_gaussians,
                                                  self.num_node_interaction_channels,
                                                  cutoff=self.cutoff,
                                                  n_conv=self.num_interactions,
                                                  kwargs=self.interaction_kwargs)


class _InteractionBlockLoop(torch.nn.Module):
    def __init__(self, num_node_hidden_channels, num_edge_gaussians, num_node_interaction_channels, cutoff, n_conv=2,
                 **kwargs):
        super(_InteractionBlockLoop, self).__init__()
        self.interactions = ModuleList()
        self.n_conv = n_conv
        self.out = Linear(num_node_hidden_channels, num_node_interaction_channels)

        for _ in range(self.n_conv):
            block = SchNet_InteractionBlock(num_node_hidden_channels, num_edge_gaussians, num_node_hidden_channels,
                                            cutoff)
            self.interactions.append(block)

    def forward(self, h, edge_index, edge_weight, edge_attr, data=None):

        for interaction in self.interactions:
            h = h + F.softplus(interaction(h, edge_index, edge_weight, edge_attr, data=data))

        h = self.out(h)

        return h


class SchNet_InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, jump=True):
        super(SchNet_InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.conv = _CFConv(hidden_channels, hidden_channels, num_filters,
                            self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        # self.act = ReLU()

        self.reset_parameters()
        self.jump = jump

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, **kwargs):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        return x


class _CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(_CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        if isinstance(edge_index, SparseTensor):
            edge_weight = edge_index.storage.value()
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)  # W 传递给message, 这里不是必须的
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor):  # [num_edge,]
        return x_j * W
