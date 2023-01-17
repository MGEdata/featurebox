"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.nn import ReLU, Sequential, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import softmax
from torch_scatter import segment_csr

from featurebox.models_geo.basemodel import BaseCrystalModel
from featurebox.models_geo.general import collect_edge_attr_jump, get_ptr, lift_jump_index_select


# class NNConvJump(NNConv):
#     """# torch.geometric scatter is unstable especially for small data in cuda device.!!!"""
#
#     @property
#     def device(self):
#         return check_device(self)
#
#     @temp_jump_cpu()
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         return super().propagate(edge_index, size=size, **kwargs)
#
#     @temp_jump()
#     def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
#         return super().message(x_j, edge_attr)


class Set2SetNew(Set2Set):

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)

            r = segment_csr((a * x), get_ptr(batch))
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NNConvNew(NNConv):
    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, um_node_hidden_channels=64, num_edge_gaussians=None, num_node_interaction_channels=64, n_conv=2,
                 **kwargs
                 ):
        super(_Interactions, self).__init__()
        nf = num_node_interaction_channels
        self.lin0 = Linear(um_node_hidden_channels, nf)

        self.short = Linear(num_edge_gaussians, nf)

        nn = Sequential(Linear(nf, nf // 4), ReLU(), Linear(nf // 4, nf * nf))

        self.conv = NNConvNew(nf, nf, nn, aggr='mean')

        self.n_conv = n_conv
        self.gru = GRU(nf, nf)

    def forward(self, h, edge_index, edge_weight, edge_attr, **kwargs):
        out = F.relu(self.lin0(h))

        edge_attr = F.relu(self.short(edge_attr))
        h = out.unsqueeze(0)

        for i in range(self.n_conv):
            m = F.relu(self.conv(out, edge_index, edge_attr))

            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        return out


class CGGRU_ReadOut(Module):
    def __init__(self, num_filters=128, n_set2set=2, out_size=1, **kwargs):
        super(CGGRU_ReadOut, self).__init__()
        nf = num_filters

        self.set2set = Set2SetNew(nf, processing_steps=n_set2set)  # very import
        self.lin1 = Linear(2 * nf, nf)
        self.lin2 = Linear(nf, out_size)

    def forward(self, out, batch):
        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class CGGRUNet(BaseCrystalModel):
    """
    CrystalGraph with CGGRUN.
    weight_decay=0.001, best.
    """

    def __init__(self, *args, num_edge_gaussians=None, num_node_interaction_channels=64, num_node_hidden_channels=64,
                 **kwargs):
        super(CGGRUNet, self).__init__(*args, num_node_interaction_channels=num_node_interaction_channels,
                                       num_edge_gaussians=num_edge_gaussians,
                                       num_node_hidden_channels=num_node_hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.num_node_hidden_channels, self.num_edge_gaussians,
                                          self.num_node_interaction_channels,
                                          n_conv=self.num_interactions, kwargs=self.interaction_kwargs)

    def get_readout_layer(self):
        self.readout_layer = CGGRU_ReadOut(self.num_node_interaction_channels,
                                           n_set2set=2, out_size=self.out_size, kwargs=self.readout_kwargs)
