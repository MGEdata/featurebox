# -*- coding: utf-8 -*-

# @Time    : 2021/8/1 18:07
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import copy
from typing import Union, List

import numpy as np
import pandas as pd
from torch.nn import Module
from torch_geometric.data import InMemoryDataset, DataLoader, Data

from featurebox.featurizers.generator_geo import SimpleDataset


def get_layers_with_weight(model: Module, add_name=""):
    name_weight_module = {}
    for name, layer in model._modules.items():
        name = "{}->{}".format(add_name, name)
        # if isinstance(layer,Module):
        if len(layer._modules) > 0:
            name_weight_module.update(get_layers_with_weight(layer, name))
        else:
            if hasattr(layer, "weight"):
                name_weight_module[name] = layer
            else:
                pass
    return name_weight_module


class LogModule:
    def __init__(self):
        self.name_weight_module_log = []
        self.name_weight_module = {}
        self.stats = {}
        self.stats_log = []
        self.result = {}

    def _get_weight(self, model: Module, add_name=""):
        name_weight_module = {}
        for name, layer in model._modules.items():
            name = "{}->{}".format(add_name, name)
            # if isinstance(layer,Module):
            if len(layer._modules) > 0:
                name_weight_module.update(self._get_weight(layer, name))
            else:
                if hasattr(layer, "weight"):
                    name_weight_module[name] = layer.weight.detach().cpu().numpy()
                else:
                    pass
        return name_weight_module

    def get_weight(self, model: Module, add_name=""):
        self.name_weight_module = self._get_weight(model, add_name=add_name)
        self.name_weight_module_log.append(self.name_weight_module)

    def stats_single(self):
        for k, v in self.name_weight_module.items():
            if isinstance(v, np.ndarray):
                self.stats[k] = np.array([np.max(v), np.min(v), np.mean(v), np.std(v)])
            else:
                self.stats[k] = None
        return self.stats

    def stats_loop(self, return_message="value"):

        if return_message == "rate":
            size = len(self.name_weight_module_log) - 1
        else:
            size = len(self.name_weight_module_log)

        for i in range(size):
            self.stats = {}
            for k in self.name_weight_module_log[i].keys():
                v = self.name_weight_module_log[i][k]
                if isinstance(v, np.ndarray):
                    if return_message == "rate":
                        v2 = self.name_weight_module_log[i + 1][k]
                        self.stats[k] = np.array([np.max(v2 - v), np.min(v2 - v), np.mean(v2 - v), np.std(v2 - v)])
                    else:
                        self.stats[k] = np.array([np.max(v), np.min(v), np.mean(v), np.std(v)])
                else:
                    self.stats[k] = None
            self.stats_log.append(self.stats)

        result = {}

        for i in self.stats_log[0].keys():
            nps = {}
            for n, v in enumerate(self.stats_log):
                if return_message == "rate":
                    nps["lp_{}-{}".format(n, n + 1)] = v[i]
                else:
                    nps["lp_{}".format(n)] = v[i]

            nps = pd.DataFrame(nps, index=["max", "min", "mean", "std"])

            result[i] = nps

        self.stats_log = []
        self.result = result

        return result


def print_log_dataset(dataset: Union[InMemoryDataset, SimpleDataset, List[Data], Data]):
    num_node_features = -1
    num_edge_features = -1
    num_state_features = -1

    if isinstance(dataset, List):
        dataset = dataset[0]
    if isinstance(dataset, Data):
        data = dataset
    else:
        data = dataset.data

    assert isinstance(data, Data)

    print("\n数据信息:\n###############")

    try:
        print("图（化合物）数={},".format(max(data.idx) + 1))
    except AttributeError:
        pass
    try:
        num_node_features = data.x.shape
        print("原子（节点）总个数: {}, 原子特征数: {}".format(num_node_features[0], num_node_features[1]))
    except AttributeError:
        pass

    try:
        num_edge_features = data.edge_attr.shape

        print("键（连接）连接个数: {}, 键特征数: {}".format(num_edge_features[0], num_edge_features[1]))
    except AttributeError:
        pass

    try:
        num_state_features = data.state_attr.shape

        print("状态数: {}, 状态特征数 {}".format(num_state_features[0], num_state_features[1]))
    except AttributeError:
        pass

    print("\n建议参数如下(若后处理，以处理后为准):")
    if num_node_features != -1:
        print("num_node_features={},".format(num_node_features[1]))
    if num_edge_features != -1:
        print("num_edge_features={},".format(num_edge_features[1]))
    if num_state_features != -1:
        print("num_state_features={},".format(num_state_features[1]))


def print_log_dataloader(dataloader: DataLoader, print_index_of_sample=False):
    dataloader0 = copy.copy(dataloader)
    names = ['x', 'edge_attr', 'y', 'pos', 'batch', 'ptr', 'z', 'idx', 'state_attr', 'adj_t', 'edge_weight',
             "num_graphs"]
    for data in dataloader0:
        shapes = []
        for i in names:
            shapes.append(getattr(data, i, None))

        print("\n每批信息（示例第0批）:\n###############")

        if shapes[-1] is not None:
            print("图（化合物）数={},".format(np.array(shapes[-1])))
        if shapes[0] is not None:
            print("节点（原子）数={}, 原子特征数={}".format(shapes[0].shape[0], shapes[0].shape[1]))
        if shapes[1] is not None:
            print("键（连接）数={}, 键特征数={}".format(shapes[1].shape[0], shapes[1].shape[1]))
        if shapes[8] is not None:
            print("状态数={}, 状态特征数={}".format(shapes[8].shape[0], shapes[8].shape[1]))

        if print_index_of_sample:
            if shapes[7] is not None:
                print("样本序号={},".format(np.array(shapes[7])))

        print("\n建议参数如下:")
        if shapes[0] is not None:
            print("num_node_features={},".format(shapes[0].shape[1]))
        if shapes[1] is not None:
            print("num_edge_features={},".format(shapes[1].shape[1]))
            print("num_edge_gaussians={},".format(shapes[1].shape[1]))
        if shapes[8] is not None:
            print("num_state_features={},".format(shapes[8].shape[1]))

        break


def make_dot_(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    # 画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    # 蓝色节点表示有梯度计算的变量Variables;
    # 橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.
    """
    Produces Graphviz representation of PyTorch autograd graph.

    First install graphviz:
        https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224  (windows)

        yum install graphviz (centos)

        apt-get install graphviz (centos)

    Second:
        pip install graphviz
        pip install torchviz

    use:
        >>> vis_graph = make_dot_(y_pred, params=dict(list(model.named_parameters())))
        >>> vis_graph.render(format="pdf")

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:

     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
            params=dict(model.named_parameters()
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    from torchviz import make_dot
    return make_dot(var, params=params, show_attrs=show_attrs, show_saved=show_saved, max_attr_chars=max_attr_chars)


class HookGradientLayer:
    def __init__(self, layer):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None

        self.forward_hook = layer.register_forward_hook(self.hook_fn_act)
        # self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)
        self.backward_hook = layer.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def stats_single(self):
        try:
            activations = self.activations
            gradients = self.gradients
            v = (gradients * activations)
            v = v.detach().cpu().numpy()
            return np.array([np.max(v), np.min(v), np.mean(v), np.std(v)])
        except BaseException:
            return None


class HookGradientModule():
    def __init__(self, target_layers=()):

        self.svls = []
        for i in target_layers:
            svl = HookGradientLayer(i)
            self.svls.append(svl)

    def apply(self, func):
        result = []
        for i in self.svls:
            ri = getattr(i, func)()
            result.append(ri)
        return result

    def stats_single(self):
        return self.apply("stats_single")
