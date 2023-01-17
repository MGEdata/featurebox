1.Structure Graph Processing
==============================

Get structure data
------------------

>>> data = pd.read_pickle("pymatgen_structures_list.pkl_pd")
>>> y = ...
>>> # using your self pymatgen.structure List

Check structure elements in scope
---------------------------------
>>> from featurebox.data.check_data import CheckElements
>>> ce = CheckElements.from_pymatgen_structures()
>>> checked_data = ce._check(data)
>>> y = np.array(y)[ce.passed_idx()]
>>> ce = CheckElements.from_pymatgen_structures()
>>> checked_data = ce.check(data)
>>> y = np.array(y)[ce.passed_idx()]

Transform parallel
------------------

>>> from featurebox.featurizers.base_graph_geo import StructureGraphGEO
>>> gt = StructureGraphGEO(n_jobs=2)
>>>
>>> """dict data for show."""
>>> in_data = gt.transform(checked_data,y=y)

>>> """1. torch_geometric.data.Data type data."""
>>> data = gt.transform_and_to_data(checked_data,y=y)

>>> """2. torch_geometric.data.Data type data and save to local path."""
>>> data = gt.transform_and_save(checked_data,y, root_dir = "path")

Using data
----------

>>> from torch_geometric.data import DataLoader
>>> import torch_geometric.transforms as T
>>> """1. Just use data (small data)."""
>>> sparse = T.ToSparseTensor()
>>> data = sparse(data)
>>> loader = DataLoader(
                dataset=data,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                )

>>> from torch_geometric.data import DataLoader
>>> import torch_geometric.transforms as T
>>> """2. Use local data (middle data)."""
>>> gen = InMemoryDatasetGeo(root="path", pre_transform=T.ToSparseTensor())
>>> loader = DataLoader(
                dataset=gen,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                )

>>> from Instances.old_net.featurebox import train_test_pack, GaussianSmearing
>>> import torch_geometric.transforms as T
>>> from torch_geometric.data import DataLoader
>>> """3. Use local data (large data)."""
>>> gen = DatasetGeo(root="path",pre_transform=T.Compose([GaussianSmearing(num_gaussians=50),T.ToSparseTensor(),]))
>>> loader = DataLoader(
                dataset=gen,
                batch_size=1,
                shuffle=True,
                num_workers=0,  )

More transforms can be shown in ``torch_geometric``.

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