"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Module, ModuleList
from torch_geometric.nn import GATConv

from featurebox.models_geo.basemodel import BaseCrystalModel
from featurebox.models_geo.general import collect_edge_attr_jump, lift_jump_index_select


class GATConvNew(GATConv):
    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, node_hidden_channels=64, num_edge_gaussians=None, num_node_interaction_channels=64, n_conv=2,
                 **kwargs,
                 ):
        super(_Interactions, self).__init__()

        _ = num_edge_gaussians

        self.lin0 = Linear(node_hidden_channels, num_node_interaction_channels)
        self.conv = ModuleList()

        for _ in range(n_conv):
            nn = GATConvNew(
                in_channels=num_node_interaction_channels, out_channels=num_node_interaction_channels,
                add_self_loops=False,

                bias=True, )
            self.conv.append(nn)

        self.n_conv = n_conv

    def forward(self, x, edge_index, edge_weight, edge_attr, **kwargs):

        out = F.softplus(self.lin0(x))
        for convi in self.conv:
            out = out + convi(x=out, edge_index=edge_index)
            # out = F.relu(convi(x=out, edge_index=edge_index))

        return out


class CrystalGraphGAT(BaseCrystalModel):
    """
    CrystalGraph with GAT.
    """

    def __init__(self, *args, num_edge_gaussians=None, num_node_interaction_channels=32,
                 num_node_hidden_channels=64, **kwargs):
        # kwargs["readout_kwargs_active_layer_type"]="ReLU"
        super(CrystalGraphGAT, self).__init__(*args, num_edge_gaussians=num_edge_gaussians,
                                              num_node_interaction_channels=num_node_interaction_channels,
                                              num_node_hidden_channels=num_node_hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.num_node_hidden_channels, self.num_edge_gaussians,
                                          self.num_node_interaction_channels,
                                          n_conv=self.num_interactions,
                                          kwargs=self.interaction_kwargs)
