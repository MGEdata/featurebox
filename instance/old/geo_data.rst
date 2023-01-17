Crystal Structure Data
=======================

Crystal structure data is not a 2D numpy data, but a cluster of tuple data.

Therefore, it suit for ``torch`` network rather than ``sklearn``.

For unification with ``torch_geometrics``, the Crystal Structure Data (Graph neural network data) use the following name.

.. image:: prop.png
    :scale: 80 %
    :align: center

Each Graph data (for each structure) contains:

``x``: Node feature matrix.  with shape [num_nodes, num_node_features]

alias atom features.

``edge_index``: Graph connectivity in COO format. with shape [2, num_edges] and type torch.long

``edge_attr``: Edge feature matrix. with shape [num_edges, num_edge_features]

alias bond features.

``edge_weight``: Edge feature matrix. with shape [num_edges, ]

alias bond length.

``pos``: Node position matrix. with shape [num_nodes, num_dimensions]

``y``: target. np.ndarray, shape [1, num_target], default shape [1,]

``state_attr``: state feature. shape [1, num_state_features]

alias state features.

``z``: atom numbers. np.ndarray, with shape [num_nodes,]

Where the state_attr is added newly.

Name alias::
    ``"node"`` <-> ``"atom"``,
    ``"edge"`` <-> ``"bond"``

Access
----------

Example:

>>> from featurebox.featurizers.base_graph_geo import StructureGraphGEO
>>> gt = StructureGraphGEO(n_jobs=2)
>>> graph_data_list = gt.transform(structures_list,y=y)

and the ``x``, ( ``atom features`` ) get by :mod:`featurebox.featurizers.atom.mapper` ,
and the ``edge_weight``,``edge_attr``(``bond features``) get by :mod:`featurebox.featurizers.envir.environment` .

.. seealso::
    :doc:`../Examples/sample_fea3`

The usage of Graph data could find in `torch geometrics <https://pytorch-geometric.readthedocs.io/en/latest/>`_
and examples: :doc:`../Examples/sample_stru2`, :doc:`../Examples/sample_stru3` .

