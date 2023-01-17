"""This part contains base model for crystal problem and tools."""
import warnings
from abc import abstractmethod

import ase.data as ase_data
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, LayerNorm, ModuleList, Softplus, ReLU, Sequential, BatchNorm1d
from torch.nn import Module
from torch_scatter import segment_csr

from featurebox.models_geo.general import get_ptr


class BaseCrystalModel(Module):
    """
    Base model for crystal problem.
    """

    def __init__(self,
                 num_node_features=1,
                 num_edge_features=3,
                 num_state_features=0,

                 num_node_hidden_channels=128,
                 num_node_interaction_channels=128,
                 num_interactions=1,

                 num_node_embeddings=120,
                 cutoff=10.0,
                 out_size=1,
                 readout='add',
                 node_data="x",
                 interactions=None,
                 readout_layer=None,
                 add_node=None,
                 add_state=False,
                 norm=False,
                 mean=None,
                 std=None,
                 scale=None,
                 **kwargs
                 ):
        """
        Model for crystal problem.

        Args:
            num_node_features: (int) input number of node feature (atom feature).
            num_edge_features: (int) input number of bond feature. if ``num_edge_gaussians` offered,
            this parameter is neglect.
            num_state_features: (int) input number of state feature.
            num_node_embeddings: (int) number of embeddings, For generate the initial embedding matrix to on
            behalf of node feature.
            num_node_hidden_channels: (int) num_node_hidden_channels for node feature.
            num_node_interaction_channels: (int) channels for node feature.
            num_interactions: (int) conv number.
            cutoff: (float) cutoff for calculate neighbor bond.
            readout: (str) Merge node method. such as "add","mean","max","mean".
            mean: (float) mean for y.
            std: (float) std for y.
            norm:(bool) False or True norm for y.
            add_node: (torch.tensor shape (120,1)) properties for atom. such as target y is volumes of compound,
                add_node could be the atom volumes of all atom (H,H,He,Li,...). And you could copy the first term to
                make sure the `H` index start form 1.
            node_data: (bool,str) just used "z" or used "x" to calculate.
            interactions: (Callable) torch module for interactions dynamically: pass the torch module to
            interactions parameter.static: re-define the ``get_interactions_layer`` and keep this parameter is None.
                the forward input is (h, edge_index, edge_weight, edge_attr, data=data)
            readout_layer: (Callable) torch module for interactions  dynamically: pass the torch module to
            interactions parameter. static: re-define the ``get_interactions_layer`` and keep this parameter is None.
            The forward input is (out,)
            add_state: (bool) add state attribute before output.
            out_size:(int) number of out size. for regression,is 1 and for classification should be defined.
        """
        super(BaseCrystalModel, self).__init__()

        self.interaction_kwargs = {}
        for k, v in kwargs.items():
            if "interaction_kwargs_" in k:
                self.interaction_kwargs[k.replace("interaction_kwargs_", "")] = v

        self.readout_kwargs = {}
        for k, v in kwargs.items():
            if "readout_kwargs_" in k:
                self.readout_kwargs[k.replace("readout_kwargs_", "")] = v

        assert readout in ['add', 'sum', 'min', 'mean', "max"]

        self.num_node_hidden_channels = num_node_hidden_channels
        self.num_state_features = num_state_features
        self.num_node_interaction_channels = num_node_interaction_channels
        self.num_interactions = num_interactions
        self.num_edge_features = num_edge_features
        self.cutoff = cutoff
        self.readout = readout

        self.interactions = interactions
        self.readout_layer = readout_layer
        self.out_size = out_size
        self.norm = norm
        self.mean = mean
        self.std = std
        self.scale = scale
        self.add_state = add_state
        self.add_node = add_node
        self.node_data = node_data

        # 定义输入
        # 使用原子性质,或者使用Embedding 产生随机数据。
        # 使用键性质,或者使用Embedding 产生随机数据。
        if num_node_embeddings < 120:
            print("Default, num_node_embeddings>=120, if you want simple the net work and "
                  "This network does not apply to other elements, the num_node_embeddings could be less but large than "
                  "the element type number in your data.")

        # 原子个数，一般不用动，这是所有原子种类数，
        # 一般来说，采用embedding的网络，
        # 在向其他元素（训练集中没有的）数据推广的能力较差。

        if node_data == "z":
            if num_node_features != 0:
                print("node_data=='z' just accept num_node_features == 0, and don't use your self-defined 'x' data, "
                      "but element number Z.")
            self.embedding_e = Embedding(num_node_embeddings, num_node_hidden_channels)
        elif self.node_data  == "x":
            self.embedding_l = Linear(num_node_features, num_node_hidden_channels)
            self.embedding_l2 = Linear(num_node_hidden_channels, num_node_hidden_channels)
        elif self.node_data  == "xz":
            self.embedding_e = Embedding(num_node_embeddings, num_node_hidden_channels)
            self.embedding_l = Linear(num_node_features, num_node_hidden_channels)
            self.embedding_l2 = Linear(num_node_hidden_channels, num_node_hidden_channels)
        else:
            raise ValueError("The node_data just accept 'z', 'x' and 'xz'.")

        self.bn = BatchNorm1d(num_node_hidden_channels)

        # 交互层 需要自定义 get_interactions_layer
        if interactions is None:
            self.get_interactions_layer()
        elif isinstance(interactions, ModuleList):
            self.get_res_interactions_layer(interactions)
        elif isinstance(interactions, Module):
            self.interactions = interactions
        else:
            raise NotImplementedError("Please implement get_interactions_layer function, "
                                      "or pass interactions parameters.")
        # 合并层 需要自定义
        if readout_layer is None:
            self.get_readout_layer()
        elif isinstance(readout_layer, Module):
            self.readout_layer = readout_layer
        else:
            raise NotImplementedError("please implement get_readout_layer function, "
                                      "or pass readout_layer parameters.")
        # 原子性质嵌入
        if add_node is not None:
            self.add_node = Emb(add_node)

        if self.add_state:
            assert self.num_state_features > 0
            self.dp = LayerNorm(self.num_state_features)
            self.ads = Linear(self.num_state_features, 2 * self.num_state_features)
            self.ads2 = Linear(2 * self.num_state_features, 2 * self.num_state_features)
            self.ads3 = Linear(2 * self.num_state_features, self.out_size)

        self.reset_parameters()

    def forward_weight_attr(self, data):



        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise NotImplementedError("Must offer edge_weight or edge_attr.")
            else:
                if data.edge_attr.shape[1] == 1:
                    data.edge_weight = data.edge_attr.reshape(-1)
                else:
                    data.edge_weight = torch.norm(data.edge_attr, dim=1, keepdim=True)

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            if not hasattr(data, "edge_weight") or data.edge_weight is None:
                raise NotImplementedError("Must offer edge_weight or edge_attr.")
            else:
                data.edge_attr = data.edge_weight.reshape(-1, 1)

        return data

    def forward(self, data):

        if data.edge_index is None:
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        data = self.forward_weight_attr(data)

        # 使用embedding 作为假的原子特征输入，而没有用原子特征输入
        assert hasattr(data, "z")
        assert hasattr(data, "pos")
        assert hasattr(data, "batch")
        z = data.z
        batch = data.batch
        # pos = data.pos
        # 处理数据阶段
        if self.node_data == "z":
            # 处理数据阶段
            assert z.dim() == 1 and z.dtype == torch.long

            h = self.embedding_e(z)
            h = F.softplus(h)
        elif self.node_data == "x":
            assert hasattr(data, "x")
            x = data.x
            h = F.softplus(self.embedding_l(x))
            h = self.embedding_l2(h)
        else:
            assert hasattr(data, "x")
            x = data.x
            h1 = self.embedding_e(z)
            x = F.softplus(self.embedding_l(x))
            h2 = self.embedding_l2(x)
            h = h1 + h2

        data.x = h

        h = self.bn(h)

        h = self.interactions(h, edge_index, data.edge_weight, data.edge_attr,
                              data=data)

        if self.add_node is not None:
            h = h + self.add_node(z)

        out = self.readout_layer(h, batch)

        if self.add_state:
            assert hasattr(data, "state_attr"), "the ``add_state`` must accept ``state_attr`` in data."
            sta = self.dp(data.state_attr)
            sta = self.ads(sta)
            sta = F.relu(sta)
            sta = self.ads2(sta)
            sta = F.relu(sta)
            sta = self.ads3(sta)
            out = out + sta

        out = self.output_forward(out)

        return self.output_forward(out)

    def get_res_interactions_layer(self, interactions):
        self.interactions = GeoResNet(interactions)

    @abstractmethod
    def get_interactions_layer(self):
        """This part shloud re-defined. And must add the ``interactions`` attribute.

        Examples::

            >>> ...
            >>> self.layer_interaction = YourNet()
        """

    def get_readout_layer(self):
        """This part shloud re-defined. And must add the ``readout_layer`` attribute.

        Examples::

            >>> self.layer_readout = torch.nn.Sequential(...)

        Examples::

            >>> ...
            >>> self.layer_readout = YourNet()
        """
        if "readout_kwargs_layers_size" in self.readout_kwargs:
            self.readout_layer = GeneralReadOutLayer(**self.readout_kwargs)
        else:
            self.readout_layer = GeneralReadOutLayer(
                [self.num_node_interaction_channels, self.readout, 200, self.out_size], **self.readout_kwargs)

    def output_forward(self, out):
        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean
        if self.norm is True:
            out = torch.norm(out, dim=-1, keepdim=True)
        if self.scale is not None:
            out = self.scale * out
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_node_hidden_channels={self.num_node_hidden_channels}, '
                f'num_node_interaction_channels={self.num_node_interaction_channels}, '
                f'num_interactions={self.num_interactions}, '
                f'cutoff={self.cutoff})')

    def reset_parameters(self):
        if hasattr(self, "embedding_e"):
            self.embedding_e.reset_parameters()
        if hasattr(self, "embedding_l"):
            self.embedding_l.reset_parameters()
        if hasattr(self, "embedding_l2"):
            self.embedding_l2.reset_parameters()
        self.bn.reset_parameters()


class ReadOutLayer(Module):
    """Merge node layer."""

    def __init__(self, channels, out_size=1, readout="add"):
        super(ReadOutLayer, self).__init__()
        self.readout = readout
        self.lin1 = Linear(channels, channels * 10)
        self.s1 = ShiftedSoftplus()
        self.lin2 = Linear(channels * 10, channels * 5)
        self.s2 = ShiftedSoftplus()
        self.lin3 = Linear(channels * 5, out_size)

    def forward(self, h, batch):
        h = self.lin1(h)
        h = self.s1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        h = self.lin2(h)
        h = self.s2(h)
        h = self.lin3(h)
        return h


class Emb(Module):

    def __init__(self, array="atomic_radii"):

        super().__init__()

        if array == "atomic_mass":
            array = torch.from_numpy(ase_data.atomic_masses)  # 嵌入原子质量
        elif array == "atomic_radii":
            array = torch.from_numpy(ase_data.covalent_radii)  # 嵌入共价半径
        elif isinstance(array, np.ndarray):
            assert array.shape[0] == 120
            array = torch.from_numpy(array)
        elif isinstance(array, Tensor):
            assert array.shape[0] == 120
        else:
            raise NotImplementedError("just accept str,np,ndarray or tensor with shape (120,)")
        # 嵌入原子属性，需要的时候运行本函数
        # (嵌入别太多，容易慢，大多数情况下用不到。)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。
        self.register_buffer('atomic_temp', array)

    def forward(self, z):
        return self.atomic_temp[z]


class GeneralReadOutLayer(Module):
    """General Merge node layer.
    """

    def __init__(self, layers_size=(128, "sum", 32, 1), last_layer=None, active_layer_type="ShiftedSoftplus"):
        super(GeneralReadOutLayer, self).__init__()
        l = len(layers_size)
        readout = [i for i in layers_size if isinstance(i, str)]
        if active_layer_type == "ShiftedSoftplus":
            active_layer = ShiftedSoftplus
        elif active_layer_type == "Softplus":
            active_layer = Softplus
        elif active_layer_type == "ReLU":
            active_layer = ReLU
        elif isinstance(active_layer_type, Module):
            active_layer = active_layer_type.__class__
        else:
            raise NotImplementedError("can't identify the type of layer.")

        assert len(readout) == 1, "The readout layer must be set one time, please there are {} layer: {}.".format(
            len(readout), readout)

        readout_site = [n for n, i in enumerate(layers_size) if isinstance(i, str)][0]
        readout = layers_size[readout_site]
        assert readout in ('sum', 'max', 'min', 'mean', 'add')
        self.readout = readout
        layers_size = list(layers_size)
        layers_size[readout_site] = layers_size[readout_site - 1]

        part_first = []
        part_second = []
        i = 0
        while i < l - 1:
            if i < readout_site - 1:
                part_first.append(Linear(layers_size[i], layers_size[i + 1]))
                part_first.append(active_layer())
            elif i < readout_site:
                pass
            else:
                part_second.append(Linear(layers_size[i], layers_size[i + 1]))
                part_second.append(active_layer())
            i += 1

        self.modules_list_1 = Sequential(*part_first)
        self.modules_list_2 = Sequential(*part_second)
        self.last_layer = last_layer

    def reset_parameters(self):
        self.modules_list_1.reset_parameters()
        self.modules_list_2.reset_parameters()
        self.last_layer.reset_parameters()

    def forward(self, h, batch):
        if len(self.modules_list_1) > 0:
            h = self.modules_list_1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        if len(self.modules_list_2) > 0:
            h = self.modules_list_2(h)
        if self.last_layer is not None:
            h = self.last_layer(h)
        return h


class GeoResNet(Module):
    """Simple ResNet"""

    def __init__(self, module_list: ModuleList):
        super().__init__()
        self.modules_list = module_list

    def forward(self, h, edge_index, edge_weight, edge_attr, data=None):
        for interaction in self.modules_list:
            h = h + interaction(h, edge_index, edge_weight, edge_attr, data=data)
        return h


class ShiftedSoftplus(Module):
    """Softplus with one log2 intercept."""

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
