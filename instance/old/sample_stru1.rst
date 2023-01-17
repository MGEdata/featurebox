Build model with QM9 data
==========================

>>> import os.path as osp
>>> import torch
>>> import torch_geometric.transforms as T
>>> from torch_geometric.data import DataLoader
>>> from torch_geometric.datasets import QM9
>>> from torch_geometric.utils import remove_self_loops


>>> from featurebox.models_geo import CrystalGraphConvNet
>>> from featurebox.models_geo.flow_geo import LearningFlow


>>> target = 0


>>> class MyTransform(object):
...    def __call__(self, data):
... # Specify target.
...        data.y = data.y[:, target]
...        return data
>>> class RemoveStr(object):
...    def __call__(self, data):
>>>        for key, item in data:
>>>            if isinstance(item, (str, float, int)):
>>>                data[key] = None
>>>             return data

>>> class Complete(object):
>>>     def __call__(self, data):
...         device = data.edge_index.device
...         row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
...         col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
...         row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
...         col = col.repeat(data.num_nodes)
...         edge_index = torch.stack([row, col], dim=0)
...         edge_attr = None
...         if data.edge_attr is not None:
...             idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
...             size = list(data.edge_attr.size())
...             size[0] = data.num_nodes * data.num_nodes
...             edge_attr = data.edge_attr.new_zeros(size)
...             edge_attr[idx] = data.edge_attr
...         edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
...         data.edge_attr = edge_attr
...         data.edge_index = edge_index
...         data.state_attr = torch.zeros((1, 2))
...         data.edge_weight = torch.norm(data.edge_attr[:, :3], dim=1, keepdim=True)
...         return data


>>> path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
>>> transform = T.Compose([MyTransform(), Complete(),  RemoveStr(), T.ToSparseTensor()])
>>> dataset = QM9(path, transform=transform).shuffle()

>>> # Normalize targets to mean = 0 and std = 1.
>>> mean = dataset.data.y.mean(dim=0, keepdim=True)
>>> std = dataset.data.y.std(dim=0, keepdim=True)
>>> dataset.data.y = (dataset.data.y - mean) / std
>>> mean, std = mean[:, target].item(), std[:, target].item()

>>> # Split datasets.
>>> test_dataset = dataset[:1000]
>>> val_dataset = dataset[1000:2000]
>>> train_dataset = dataset[2000:3000]
>>> test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
>>> val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
>>> train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

>>> device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
>>> # model = CGGRUNet(11, 5)
>>> model = CrystalGraphConvNet(11, 5)
>>> # model = CrystalGraphGCN(11, 5)
>>> # model = CrystalGraphGCN2(11, 5)
>>> # model = CrystalGraphGAT(11, 5).to(device)
>>> # model = SchNet(0,5,simple_z=True)
>>> # model = MEGNet(11, 5)
>>> # model = SchNet(11,5)
>>> # model = SchNet(0,5,simple_z=True)
>>> # model = SchNet(11,5)
>>> # model = CrystalGraphConvNet(0,5,simple_z=True)
>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
>>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
... factor=0.7, patience=2, min_lr=0.001)

>>> lf= LearningFlow(model, train_loader, validate_loader=val_loader, device= "cuda:1",
>>> optimizer=None, clf= False, loss_method=None, learning_rate = 1e-3, milestones=None,
... weight_decay= 0.01, checkpoint=True, scheduler=scheduler,
... loss_threshold= 0.1, print_freq= 0, print_what="all")

>>> # lf.run(50)