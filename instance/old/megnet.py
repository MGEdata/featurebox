"""
Note:
    This is a simple version for mgenet, where the bond update iteration is omitted.
"""
import warnings
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.nn import Sequential
from torch_geometric.nn import MessagePassing
from torch_scatter import segment_csr
from torch_sparse import SparseTensor

from featurebox.models_geo.basemodel import BaseCrystalModel, ShiftedSoftplus
from featurebox.models_geo.general import get_ptr, collect_edge_attr_jump, lift_jump_index_select


class MEGNet(BaseCrystalModel):
    """
    MEGNet
    """

    def __init__(self, *args, **kwargs):
        super(MEGNet, self).__init__(*args, **kwargs)
        if self.num_state_features == 0:
            warnings.warn(
                "you use no state_attr !!!, please make sure the ``num_state_features`` compat with your data")

    def get_interactions_layer(self):
        self.interactions = _Interaction(self.num_node_hidden_channels, self.num_edge_gaussians,
                                         self.num_node_interaction_channels,
                                         cutoff=self.cutoff,
                                         n_conv=self.num_interactions, num_state_features=self.num_state_features,
                                         kwargs=self.interaction_kwargs
                                         )


class _Interaction(torch.nn.Module):
    def __init__(self, num_node_hidden_channels, num_edge_gaussians, num_node_interaction_channels, cutoff, n_conv=2,
                 num_state_features=0, **kwargs):
        super(_Interaction, self).__init__()
        self.interactions = ModuleList()
        self.lin_list1 = ModuleList()
        self.n_conv = n_conv
        self.out = Linear(num_node_hidden_channels, num_node_interaction_channels)

        for _ in range(self.n_conv):
            self.lin_list1.append(Linear(num_node_hidden_channels + num_state_features, num_node_hidden_channels))
            block = Meg_InteractionBlock(num_node_hidden_channels, num_edge_gaussians, num_node_interaction_channels,
                                         cutoff)
            self.interactions.append(block)
        self.num_state = num_state_features

    def forward(self, h, edge_index, edge_weight, edge_attr, data=None):
        state_attr = data.state_attr

        for lin1, interaction in zip(self.lin_list1, self.interactions):
            state_attr = state_attr[data.batch]
            hs = torch.cat((state_attr, h), dim=1)
            h = lin1(hs)
            h = h + interaction(h, edge_index, edge_weight, edge_attr, data=data)

            r = segment_csr(h, get_ptr(data.batch), reduce="mean")

            state_attr = torch.sum(r, dim=1, keepdim=True)

            state_attr = state_attr.expand(
                data.state_attr.shape)

        h = self.out(h)
        return h


class Meg_InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(Meg_InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.conv = _CFConv(hidden_channels, hidden_channels, num_filters,
                            self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        # self.lin = Linear(num_node_hidden_channels, num_node_hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, data=None):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        # x = self.lin(x)
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

    def message(self, x_j, W):  # [num_edge,]
        return x_j * W
