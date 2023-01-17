2.Graph Network
==============================


GaussianSmearing transform data
---------------------------------

>>> test_dataset = dataset[:1000]
>>> val_dataset = dataset[1000:2000]
>>> train_dataset = dataset[2000:3000]
>>> import torch_geometric.transforms as T
>>> from featurebox.models_geo.basemodel import GaussianSmearing
>>> trans = T.Compose([GaussianSmearing(num_gaussians=50),T.ToSparseTensor(),])

>>> from featurebox.featurizers.generator_geo import SimpleDataset
>>> train_dataset = SimpleDataset(data_train, pre_transform=trans)
>>> test_dataset = SimpleDataset(data_test,pre_transform=trans)
>>> val_dataset = SimpleDataset(val_data,pre_transform=trans)

>>> from torch_geometric.data.dataloader import DataLoader
>>> train_loader = DataLoader(
... dataset=train_dataset,
... batch_size=200,
... shuffle=False,
... num_workers=0)

>>> test_loader = ...
>>> val_loader = ...

Model
--------------

>>> from featurebox.models_geo.cgcnn import CrystalGraphConvNet
>>>  model = CrystalGraphConvNet(num_node_features=91,
...  num_edge_features=3,
...  num_state_features=29)
>>> # model = CrystalGraphGCN(...)
>>> # model = CrystalGraphGCN2(...)
>>> # model = CrystalGraphGAT(...)
>>> # model = SchNet(...)
>>> # model = MEGNet(...)
>>> # model = SchNet(...)

Training
--------------

>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
>>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2,...min_lr=0.001)
>>> device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

>>> from featurebox.models_geo.flow_geo import LearningFlow
>>> lf= LearningFlow(model, train_loader, validate_loader=val_loader, device= "cuda:1",
... optimizer=None, clf= False, loss_method=None, learning_rate = 1e-3, milestones=None,
... weight_decay= 0.01, checkpoint=True, scheduler=scheduler,
... loss_threshold= 0.1, print_freq= None, print_what="all")

>>> #lf.run(50)

More transforms can be shown in ``torch_geometric``.

To SparseTensor is more stable with random seed.

Note
----

    Each Graph data (for each structure):

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]
    
    ``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long
    
    ``edge_attr``: Edge feature matrix. np.ndarray, with shape [num_edges, num_edge_features]

    ``edge_weight``: Edge feature matrix. np.ndarray, with shape [num_edges, ]
    
    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]
    
    ``y``: target. np.ndarray, shape (1, num_target), default shape (1,)
    
    ``state_attr``: state feature. np.ndarray, shape (1, num_state_features)
    
    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]
    
    Where the state_attr is added newly.