"""
Abstract classes for building graph representations consist with ``pytorch-geometric``.

All the Graph in this part should return data as following:

Each Graph data (for each structure):


``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

``edge_attr``: Edge feature matrix. np.ndarray,  with shape [num_edges, num_edge_features]

``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

``state_attr``: state feature. np.ndarray, shape (1, num_state_features)

``z``: atom numbers. np.ndarray, with shape [num_nodes,]

Where the state_attr is added newly.

"""
import os
import warnings
from collections import Iterable
from shutil import rmtree
from typing import Dict, List

import numpy as np
import torch
import torch_geometric
from mgetool.tool import parallelize, batch_parallelize
from pymatgen.core import Structure

from featurebox.featurizers.base_feature import DummyConverter, BaseFeature, ConverterCat
from featurebox.featurizers.envir.environment import GEONNGet


class _BaseStructureGraphGEO(BaseFeature):

    def __init__(self, collect=False, return_type="tensor", batch_calculate: bool = True,
                 add_label=False, check=True,
                 **kwargs):
        """

        Args:
            collect:(bool), Gather the batch data by different name.
            True Just for show!!
            return_type:(str), Return torch.tensor default, "numpy" is Just for show!!
            **kwargs:
        """
        super().__init__(batch_calculate=batch_calculate, **kwargs)
        self.graph_data_name = []

        assert return_type in ["tensor", "np", "numpy", "array", "ndarray"]

        if collect is True:
            warnings.warn("The collect=True just used for temporary show!", UserWarning)
        if return_type != "tensor":
            warnings.warn("The collect != tensor just used for temporary display the shape of ndarray!!",
                          UserWarning)
        self.return_type = return_type
        self.collect = collect
        self.convert_funcs = [i for i in dir(self) if "_convert_" in i]
        self.add_label = add_label
        self.check = check

    def __add__(self, other):
        raise TypeError("There is no add.")

    def __call__(self, structure: Structure, **kwargs) -> Dict:
        return self.convert(structure, **kwargs)

    @staticmethod
    def _get_dummy_converter() -> DummyConverter:
        return DummyConverter()

    def _transform(self, structures: List[Structure], **kwargs) -> List[Dict]:
        """

        Args:
            structures:(list) Preprocessing of samples need to transform to Graph.

            **kwargs:

        Returns:
            list of graphs:
                List of dict

        """
        assert isinstance(structures, Iterable)
        if hasattr(structures, "__len__"):
            assert len(structures) > 0, "Empty input data!"

        le = len(structures)

        for i in kwargs.keys():
            if kwargs[i] is None:
                kwargs[i] = [kwargs[i]] * len(structures)
            elif not isinstance(kwargs[i], Iterable):
                kwargs[i] = [kwargs[i]] * len(structures)

        kw = [{k: v[i] for k, v in kwargs.items()} for i in range(le)]

        iterables = zip(structures, kw)

        if not self.batch_calculate:
            rets = parallelize(self.n_jobs, self._wrapper, iterables, tq=True, respective=True, respective_kwargs=True)
            ret, self.support_ = zip(*rets)

        else:
            rets = batch_parallelize(self.n_jobs, self._wrapper, iterables, respective=True, respective_kwargs=True,
                                     tq=True, mode="j", batch_size=self.batch_size)

            ret, self.support_ = zip(*rets)

        if self.add_label and self.return_type == "tensor":
            [i.update({"label": torch.tensor([n, n])}) for n, i in enumerate(ret)]  # double for after.
        elif self.add_label:
            [i.update({"label": np.array([n, n])}) for n, i in enumerate(ret)]  # double for after.
        else:
            pass
        return ret

    def get_collect_data(self, graphs: List[Dict]):
        """
        Not used in default, just for shown.

        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.
        Aim to get input for GraphGenerator.

        Args:
            graphs: list of dict
                list of graph dictionary for each structure
        """
        if self.collect is False:
            return graphs

        output = {}  # Will be a list of arrays

        # Convert the graphs to matrices
        for n, fi in enumerate(self.graph_data_name):
            np_data = [np.array(gi[fi]) if isinstance(gi, dict) else np.array(gi[n]) for gi in graphs]
            if self.return_type == "tensor":
                output[n] = [torch.from_numpy(i) for i in np_data]
            else:
                output[n] = np_data
        return output

    def transform(self, structures: List[Structure], state_attributes=None, y=None, **kwargs) -> List[
        Dict]:
        """
        New type of transform structure.

        Args:
            structures: (list)
                preprocessing of samples need to transform to Graph.
            state_attributes: (list)
                preprocessing of samples need to add to Graph.
            y: (list)
                Target to train against (the same size with structure)
        Returns:
            data
        """
        data = self._transform(structures, state_attributes=state_attributes, y=y, **kwargs)
        if self.check:
            # data = np.array(data)[np.array(self.support_)].tolist()
            data = [di for di, si in zip(data, self.support_) if si]

        return data

    def transform_and_collect(self, structures: List[Structure], **kwargs) -> Dict:
        """
        New type of transform structure.

        Args:
            structures: (list)
                preprocessing of samples need to transform to Graph.
            state_attributes: (list)
                preprocessing of samples need to add to Graph.
            y: (list)
                Target to train against (the same size with structure)
        Returns:
            data
        """
        self.collect = True
        return self.get_collect_data(self._transform(structures, **kwargs))

    def save(self, obj, name, root_dir=".") -> None:
        """Save."""
        torch.save(obj, os.path.join(root_dir, "raw", '{}.pt'.format(name)))

    def check_dup(self, structures, file_names="composition_name"):
        """Check the names duplication"""
        names = [i.composition.reduced_formula for i in structures]
        if file_names == "composition_name" and len(names) != len(set(names)):
            raise KeyError("There are same composition name for different structure, "
                           "please use file_names='rank_number' "
                           "to definition or specific names list.")
        elif len(set(file_names)) == len(structures):
            return file_names
        elif file_names == "rank_number":
            return ["raw_data_{}".format(i) for i in range(len(structures))]
        else:
            return names

    def transform_and_save(self, *args, root_dir=".", file_names="composition_name", save_mode="o", **kwargs):
        r"""Save the data to 'root_dir/raw' if save_mode="i", else 'root_dir', conpact with InMemoryDatasetGeo"""
        raw_path = os.path.join(root_dir, "raw")
        if os.path.isdir(raw_path):
            rmtree(raw_path)
        os.makedirs(raw_path)

        result = self.transform(*args, **kwargs)
        print("Save raw files to {}.".format(raw_path))
        if save_mode in ["I", "R", "i", "r", "Respective", "respective"]:
            fns = self.check_dup(args[0], file_names=file_names)
            [self.save(i, j, root_dir) for i, j in zip(result, fns)]
        else:
            self.save(result, "raw_data", root_dir=root_dir)
        print("Done.")
        return result

    def transform_and_to_data(self, *args, **kwargs) -> List[torch_geometric.data.Data]:
        """Return list of torch_geometric.data.Data."""
        from torch_geometric.data import Data
        result = self.transform(*args, **kwargs)
        return [Data.from_dict(i) for i in result]

    def convert(self, structure: Structure, **kwargs) -> Dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.

        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: list
                state attributes
            structure: Structure
                pymatgen Structure
            y:list
                Target

        Returns:
            dict
        """

        def rs(res):
            dt = {}
            for key, value in res.items():
                if key in self.graph_data_name:
                    if isinstance(value, np.ndarray):
                        dt[key] = torch.from_numpy(value)
                    elif isinstance(value, (int, float)):
                        dt[key] = value
                    else:
                        dt[key] = value
            return dt

        convert_funcs = [i for i in dir(self) if "_convert_" in i] if not self.convert_funcs else self.convert_funcs
        if len(convert_funcs) <= 1:
            raise NotImplementedError("Please implement the ``_convert_*`` functions, "
                                      "such as ``_convert_edge_attr``.")
        result_old = [getattr(self, i)(structure, **kwargs) for i in convert_funcs]
        result = {}
        for dcti in result_old:
            for k in dcti:
                if k in result:
                    result[k] = np.concatenate((result[k], dcti[k]),
                                               axis=-1)  # please just for state-attr,edge_attr, and x
                else:
                    result[k] = dcti[k]
        result = {key: value for key, value in result.items() if key in self.graph_data_name}
        if self.check:
            for key, value in result.items():
                if key != "y":
                    if not np.all(np.isreal(value)):
                        raise ValueError(
                            "There is not real numerical data for {} of {}.  Not Acceptable: nan, infinite, complex.".format(
                                structure.composition, key))

        if self.return_type == "tensor":
            result = rs(result)

        return result

    def _convert_sample(self, *args, **kwargs):
        """
        Sample for convert.

        Returns:
            data:(dict)

        """

        return {}


class BaseStructureGraphGEO(_BaseStructureGraphGEO):
    """

    Returns

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

    ``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]

    Examples:
        >>> from torch_geometric.loader import DataLoader
        >>> sg1 = BaseStructureGraphGEO()
        >>> data_list = sg1.transform_and_to_data(structures_checked)
        >>> loader = DataLoader(data_list, batch_size=3)
        >>> for i in loader:
        ...     print(i)


    """

    def __init__(self,
                 atom_converter: BaseFeature = None,
                 state_converter: BaseFeature = None,
                 **kwargs):
        """

        Args:
            atom_converter: (BinaryMap) atom features converter. See Also:
                :class:`featurebox.test_featurizers.atom.mapper.AtomTableMap` , :class:`featurebox.test_featurizers.atom.mapper.AtomJsonMap` ,
                :class:`featurebox.test_featurizers.atom.mapper.AtomPymatgenPropMap`, :class:`featurebox.test_featurizers.atom.mapper.AtomTableMap`
            state_converter: (Converter) state features converter. See Also:
                :class:`featurebox.test_featurizers.state.state_mapper.StructurePymatgenPropMap`
                :mod:`featurebox.test_featurizers.state.statistics`
                :mod:`featurebox.test_featurizers.state.union`
            **kwargs:
        """
        super().__init__(**kwargs)

        self.atom_converter = atom_converter or self._get_dummy_converter()
        self.state_converter = state_converter or self._get_dummy_converter()
        self.graph_data_name = ["x", 'y', 'pos', "state_attr", 'z']

    def _convert_state_attr(self, structure, state_attributes=None, **kwargs):
        """return shape (1, num_state_features)"""
        _ = kwargs
        if state_attributes is not None:
            state_attributes = np.array(state_attributes)
        else:
            state_attributes = np.array([0.0, 0.0], dtype="float32")
        if isinstance(self.state_converter, DummyConverter):
            pass
        else:
            state_attributes = np.concatenate(
                (state_attributes, np.array(self.state_converter.convert(structure)).ravel()))

        state_attributes = state_attributes[np.newaxis, :]

        state_attributes = state_attributes.astype(dtype=np.float32)

        return {'state_attr': state_attributes}

    def _convert_y(self, structure, y=None, **kwargs):
        """return shape (1, num_state_features)"""
        _ = kwargs
        _ = structure
        if y is not None:
            if isinstance(y, np.ndarray):
                if y.dtype == np.float64:
                    y = y.astype(np.float32)
                y = y.ravel()[np.newaxis, :]
            elif isinstance(y, float):
                y = np.array(y, dtype=np.float32).ravel()[np.newaxis, :]
            elif isinstance(y, int):
                y = np.array(y, dtype=np.int64).ravel()[np.newaxis, :]
            else:
                y = np.array(y, dtype=np.float32).ravel()[np.newaxis, :]
        else:
            y = np.array(np.nan)

        return {'y': y}

    def _convert_x(self, structure, **kwargs):
        _ = kwargs
        z = np.array(structure.atomic_numbers).ravel()
        if isinstance(self.atom_converter, DummyConverter):
            atoms = self.atom_converter.convert(z)
        elif isinstance(self.atom_converter, ConverterCat):
            self.atom_converter.force_concatenate = True  # just accept the data could be concatenate as one array.
            atoms = self.atom_converter.convert(structure)
        else:
            atoms = self.atom_converter.convert(structure)

        atoms = atoms.astype(dtype=np.float32)
        if atoms.ndim <= 1:
            atoms = atoms.reshape(1, -1)
        z = z.astype(dtype=np.int64)
        return {'x': atoms, "z": z}

    def _convert_pos(self, structure, **kwargs):
        _ = kwargs
        pos = structure.cart_coords.astype(dtype=np.float32)
        if pos.ndim <= 1:
            pos = pos.reshape(1, -1)
        return {'pos': pos}


class StructureGraphGEO(BaseStructureGraphGEO):
    """
    Returns

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

    ``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

    ``edge_attr``: Edge feature matrix. np.ndarray,  with shape [num_edges, num_edge_features]

    ``edge_weight``: Edge feature matrix. np.ndarray,  with shape [num_edges, ]

    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

    ``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

    ``state_attr``: state feature. np.ndarray, shape (1, num_state_features)

    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]

    Where the state_attr is added newly.

    Examples:
    >>> from torch_geometric.loader import DataLoader
    >>> sg1 = BaseStructureGraphGEO()
    >>> data_list = sg1.transform_and_to_data(structures_checked)
    >>> loader = DataLoader(data_list, batch_size=3)
    >>> for i in loader:
    ...     print(i)

    """

    def __init__(self, nn_strategy="find_points_in_spheres",
                 bond_generator=None,
                 bond_converter: BaseFeature = None,
                 cutoff: float = 5.0,
                 pbc=True,
                 **kwargs):
        """
        Args:
            nn_strategy: (str) NearNeighbor strategy.
                ["find_points_in_spheres", "find_xyz_in_spheres",
                "BrunnerNN_reciprocal", "BrunnerNN_real", "BrunnerNN_relative",
                "EconNN", "CrystalNN", "MinimumDistanceNNAll", "find_points_in_spheres","UserVoronoiNN",
                "ACSF","BehlerParrinello","EAD","EAMD","SOAP","SO3","SO4_Bispectrum","wACSF",]
            atom_converter: (BinaryMap) atom features converter.
                See Also:
                :class:`featurebox.test_featurizers.atom.mapper.AtomTableMap` , :class:`featurebox.test_featurizers.atom.mapper.AtomJsonMap` ,
                :class:`featurebox.test_featurizers.atom.mapper.AtomPymatgenPropMap`, :class:`featurebox.test_featurizers.atom.mapper.AtomTableMap`
            bond_converter: (Converter)
                bond features converter, default=None.
            state_converter: (Converter)
                state features converter.
                See Also:
                :class:`featurebox.test_featurizers.state.state_mapper.StructurePymatgenPropMap`
                :mod:`featurebox.test_featurizers.state.statistics`
                :mod:`featurebox.test_featurizers.state.union`
            bond_generator: (GEONNGet, )
                bond features converter.
            cutoff: (float)
                Whether to use depends on the ``nn_strategy``.
        """

        super().__init__(**kwargs)
        self.cutoff = cutoff

        if bond_generator is None or bond_generator == "GEONNDict":  # default use GEONNDict
            self.bond_generator = GEONNGet(nn_strategy, numerical_tol=1e-8, pbc=pbc, cutoff=cutoff)
        else:
            self.bond_generator = bond_generator

        self.bond_converter = bond_converter or self._get_dummy_converter()

        self.graph_data_name = ["x", 'y', 'pos', "state_attr", 'z', 'edge_index', "edge_weight", "edge_attr", ]

    def _convert_edges(self, structure: Structure, **kwargs):
        """get edge data."""
        _ = kwargs

        center_indices, edge_index, edge_attr, edge_weight, center_prop = self.bond_generator.convert(structure)

        index = edge_index[0] < edge_index[1]

        edge_index = edge_index[:, index]
        edge_weight = edge_weight[index]
        edge_attr = edge_attr[index]

        if edge_weight.ndim == 2:
            if edge_weight.shape[1] == 3:
                edge_weight = np.sum(edge_weight ** 2, axis=1) ** 0.5
                edge_weight = edge_weight.reshape(-1, 1)
            elif edge_weight.shape[1] == 1:
                pass
            else:
                raise TypeError("edge_weight just accpet 'xyz' shape(n_node,3) or 'r' shape(n_node,)")
        elif edge_weight.ndim != 1:
            raise TypeError("edge_weight just accpet 'xyz' shape(n_node,3) or 'r' shape(n_node,1)")

        assert len(center_indices) == len(structure.atomic_numbers), \
            "center_indices less than atomic_numbers, this structure is illegal," \
            "which could due to there is discrete atom out of cutoff distance in structure."
        if not np.all(np.equal(np.sort(center_indices), np.array(center_indices))):
            raise UserWarning("center_indices rank is not compact with atomic_numbers,"
                              "which could due to the structure atoms rank are re-arranged in ``bond_generator``,"
                              "this could resulting in a correspondence error",
                              )

        edge_weight = self.bond_converter.convert(edge_weight).ravel()

        if edge_weight.ndim == 2 and edge_weight.shape[1] == 1:
            edge_weight = edge_weight.ravel()

        edge_index = edge_index.astype(dtype=np.int64)
        edge_weight = edge_weight.astype(dtype=np.float32)
        edge_attr = edge_attr.astype(dtype=np.float32)

        if edge_index.ndim <= 1:
            edge_index = edge_index.reshape(2, -1)

        if edge_weight.ndim <= 1:
            edge_weight = edge_weight.ravel()
            if edge_weight.shape[0] == 0:
                raise ValueError(
                    "Bad data The {} is with no edge_index in cutoff. May lead to later wrong.".format(
                        structure.composition), )

        if edge_attr.ndim <= 1:
            edge_attr = edge_attr.reshape(1, -1)

        if center_prop is not None and np.all(np.isreal(center_prop)) and center_prop.size != 1:
            center_prop = center_prop.astype(dtype=np.float32)

            return {'edge_index': edge_index, "edge_weight": edge_weight, "edge_attr": edge_attr, "x": center_prop}
        else:
            return {'edge_index': edge_index, "edge_weight": edge_weight, "edge_attr": edge_attr}
