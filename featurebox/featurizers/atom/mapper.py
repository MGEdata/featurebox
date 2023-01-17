"""Get pure atom properties.

Embedded data: "ele_table.csv", "ele_megnet.json", "ie.json", "oe.csv"

"""
import functools
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, Callable

import numpy as np
import pandas as pd
from monty.serialization import loadfn
from pymatgen.core import Element, Structure, Lattice

from featurebox.data.check_data import ALL_ELE_N_MAP, ALL_N_ELE_MAP
from featurebox.featurizers.base_feature import BaseFeature

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


def get_atom_fea_number(structure: Structure) -> List[int]:
    """
    Get atom features from structure, may be overwritten.

    Args:
        structure: (Pymatgen.Structure) pymatgen structure
    Returns:
        List of atomic numbers
    """
    return [i.specie.Z for i in structure]


def get_atom_fea_name(structure: Structure) -> List[dict]:
    """
    For a structure return the list of dictionary for the site occupancy
    for example, Fe0.5Ni0.5 site will be returned as {"Fe": 0.5, "Ni": 0.5}

    Args:
        structure (Structure): pymatgen Structure with potential site disorder

    Returns:
        a list of site fraction description
    """

    return [{str(i.symbol): 1} for i in structure.species]
    # return [{i.element.symbol: 1} for i in structure.species]
    # return [i.species.to_dict() for i in structure.sites]


def get_ion_fea_name(structure: Structure) -> List[dict]:
    """
    For a structure return the list of dictionary for the site occupancy
    for example, Fe0.5Ni0.5 site will be returned as {"Fe2+": 0.5, "Ni2+": 0.5}

    Args:
        structure (Structure): pymatgen Structure with potential site disorder

    Returns:
        a list of site fraction description
    """

    return [i.species.to_reduced_dict for i in structure.sites]


##############################################################

class AtomMap(BaseFeature):
    """
    Base class for atom converter. Map the element type and weight to element data.
    """

    def __init__(self, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'any'):
        super(AtomMap, self).__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    @staticmethod
    def get_json_embeddings(file_name: str = "ele_megnet.json") -> Dict:
        """get json preprocessing"""
        data = loadfn(MODULE_DIR / "data" / file_name)
        data = {i: np.array(j) for i, j in data.items()}
        return data

    @staticmethod
    def get_csv_embeddings(data_name: str) -> pd.DataFrame:
        """get csv preprocessing"""
        oedata = pd.read_csv(os.path.join(MODULE_DIR, "data", data_name), index_col=0)
        oedata = oedata.apply(pd.to_numeric, errors='ignore')
        oedata = oedata.fillna(0)
        return oedata


class BinaryMap(AtomMap):
    """Base converter with 2 different search_tp.

    """

    def __init__(self, search_tp: str = "number", weight: bool = False, **kwargs):
        """

        Args:
            search_tp: (str)
            weight: (bool) , For true,the same key data are summed together.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.search_tp = search_tp
        self.weight = weight
        self.ndim = 1

    def _convert(self, d: Any) -> Any:
        if self.search_tp in ["name", "name_dict"]:
            if isinstance(d, Structure):
                d = get_atom_fea_name(d)
            return self.convert_dict(d)
        elif self.search_tp == "ion_name":
            if isinstance(d, Structure):
                d = get_ion_fea_name(d)
            return self.convert_dict(d)
        else:
            if isinstance(d, Structure):
                d = get_atom_fea_number(d)
            return self.convert_number(d)

    @abstractmethod
    def convert_dict(self, d: List[Dict]):
        """convert list of dict to data"""

        """the subclass must del all the following code and write yourself"""

        # atoms_name = [ALL_ELE_N_MAP[list(i.keys())[0]] for i in d]
        # return self.convert_number(atoms_name)

    @abstractmethod
    def convert_number(self, d: List[int]):
        """convert list of number to data"""
        """The subclass must del all the following code and write yourself"""

        # atoms_name = [{ALL_N_ELE_MAP[i]: 1} for i in d]
        # return self.convert_dict(atoms_name)


# ##############Atom#######################


class AtomJsonMap(BinaryMap):
    """
    Fixed Atom json map.

    Examples
    ----------
    >>> tmps = AtomJsonMap(search_tp="number")
    >>> s = [1,76]                   #[i.specie.Z for i in structure]
    >>> a = tmps.convert(s)
    >>> tmps = AtomJsonMap(search_tp="name")
    >>> s = [{"H": 2, }, {"Al": 1}]  #[i.species.as_dict() for i in pymatgen_structure.sites]
    >>> a = tmps.convert(s)
    >>>
    >>> tmps = AtomJsonMap(search_tp="name")
    >>> s = [[{"H": 2, }, {"Ce": 1}],[{"H": 2, }, {"Al": 1}]]
    >>> a = tmps.transform(s)

    """

    def __init__(self, embedding_dict: Union[str, Dict] = None, search_tp: str = "name", **kwargs):
        """

        Args:
            embedding_dict: (str,dict)
                Name of file or dict,element to element vector dictionary

                Provides the pre-trained elemental embeddings using formation energies,
                which can be used to speed up the training. The embeddings
                are also extremely useful elemental descriptors that encode chemical
                similarity that may be used in other ways.
        """

        super(AtomJsonMap, self).__init__(**kwargs)
        if embedding_dict is None:
            embedding_dict = self.get_json_embeddings()
        elif isinstance(embedding_dict, str):
            embedding_dict = self.get_json_embeddings(embedding_dict)

        assert len(set([len(i) for i in embedding_dict.values()])) == 1, \
            "The element number should be same with `ele_megnet.json`, " \
            "which contains elemental features for 89 elements (up to Z=94, excluding Po, At, Rn, Fr, Ra) "

        self.embedding_dict = embedding_dict
        self.search_tp = search_tp
        self.ndim = 1

    def __add__(self, other):
        if isinstance(other, AtomJsonMap):
            for key in other.embedding_dict.keys():
                if key in self.embedding_dict:
                    if isinstance(self.embedding_dict[key], list):
                        self.embedding_dict[key].extend(other.embedding_dict[key])
                    else:
                        self.embedding_dict[key] = np.append(self.embedding_dict[key], other.embedding_dict[key])
                else:
                    pass
        else:
            raise TypeError("only same class can be added.")
        return self

    def convert_dict(self, atoms: List[dict]) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        if isinstance(atoms, dict):
            atoms = [{k: v} for k, v in atoms.items()]

        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                if self.weight:
                    emb += np.array(self.embedding_dict[k]).astype(np.float64) * v
                else:
                    emb += np.array(self.embedding_dict[k]).astype(np.float64)
            features.append(emb)
        if len(atoms) == 1:
            return np.array(features).ravel().astype(np.float64)
        else:
            return np.array(features).reshape((len(atoms), -1)).astype(np.float64)

    def convert_number(self, atoms: List[int]) -> np.ndarray:

        atoms_name = [ALL_N_ELE_MAP[i] for i in atoms]
        features = [self.embedding_dict[i] for i in atoms_name]

        if len(atoms) == 1:
            return np.array(features).astype(np.float64).ravel()
        else:
            return np.array(features).astype(np.float64).reshape((len(atoms), -1))


class AtomTableMap(BinaryMap):
    """
    Fixed Atom embedding map.
    Default table is oe.csv.
    you can change the table yourself for different preprocessing.
    The table contains elemental features for 92 U elements at least.
    Please check all your data is int or float!!!

    Such as:

    ===== ===== ===== =====
    Data    F0    F1    ...
    ----- ----- ----- -----
    H     V     V     ...
    He    V     V     ...
    Li    V     V     ...
    Be    V     V     ...
    ...   ...   ...   ...
    ===== ===== ===== =====

    Examples
    --------
    >>> tmps = AtomTableMap(search_tp="number")
    >>> s = [1,76]
    >>> tmps.convert(s)
    array([[2.245000e+01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00],
           [2.383710e+03, 3.937715e+04, 3.783280e+03, 9.866700e+02,
            8.349720e+03, 6.978800e+02, 1.861780e+03, 1.549970e+03,
            9.784700e+02, 2.231900e+02, 2.633000e+02, 1.689800e+02,
            2.854000e+01, 0.000000e+00, 1.841000e+01, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00]])

    >>> tmps = AtomTableMap(search_tp="name")
    >>> s = [{"H": 2, }, {"Po": 1}]  #[i.species.as_dict() for i in pymatgen.structure.sites]
    >>> a = tmps.convert(s)
    ...
    >>> tmps = AtomTableMap(search_tp="name",tablename="oe.csv")
    >>> s = [[{"H": 2, }, {"Po": 1}],[{"H": 2, }, {"Po": 1}]]
    >>> a = tmps.transform(s)
    ...

    >>> tmps = AtomTableMap(tablename=None)
    >>> tmps = AtomTableMap(tablename="ele_table.csv")
    >>> s = [{"H": 2, }, {"Pd": 1}]
    >>> b = tmps.convert(s)
    ...

    """

    def __init__(self, tablename: Union[str, np.ndarray, pd.DataFrame, None] = "oe.csv",
                 search_tp: str = "name", **kwargs):
        """

        Parameters
        ----------
        tablename: str,np.ndarray, pd.Dateframe
            1. Name of table in bgnet.preprocessing.resources. if tablename is None,
            use the embedding "ele_table.csv".\n
            2. np.ndarray, search_tp = "number".\n
            3. pd.dataframe, search_tp = "name"

        search_tp:str
            Name
        """

        super(AtomTableMap, self).__init__(search_tp=search_tp, **kwargs)

        if tablename is None:
            self.da = self.get_ele_embeddings("ele_table_norm.csv")
            self.dax = self.da.values
            self.da_columns = list(self.da.columns)

        elif isinstance(tablename, str):
            if tablename in ["ele_table.csv", "ele_table_norm.csv"]:
                self.da = self.get_ele_embeddings(tablename)
            else:
                self.da = self.get_csv_embeddings(tablename)
            self.dax = self.da.values
            self.da_columns = list(self.da.columns)

        elif isinstance(tablename, np.ndarray):
            self.da_columns = None
            self.dax = tablename.astype(np.float64)

        elif isinstance(tablename, pd.DataFrame):
            tablename = tablename.apply(pd.to_numeric)
            self.da = tablename
            self.dax = self.da.values
            self.da_columns = list(self.da.columns)
        else:
            raise TypeError("just accept np.array,pd.dataframe, or str name of csv")
        self.dax = np.concatenate((np.zeros((1, self.dax.shape[1])), self.dax), axis=0)  # for index from 1
        self.tablename = tablename

        if search_tp == "number":
            self.ndim = None
        else:
            self.ndim = 1

    @staticmethod
    def get_ele_embeddings(name="ele_table_norm.csv") -> pd.DataFrame:
        """get CSV preprocessing"""
        oedata = pd.read_csv(os.path.join(MODULE_DIR, "data", name), index_col=0, header=0, skiprows=0)
        oedata = oedata.drop(["abbrTex", "abbr"], axis=0)
        oedata = oedata.fillna(0)
        oedata = oedata.apply(pd.to_numeric, errors='ignore')
        return oedata

    def convert_dict(self, atoms: List[Dict]) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        if isinstance(atoms, dict):
            atoms = [{k: v} for k, v in atoms.items()]

        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                try:
                    if self.weight:
                        emb += self.da.loc[k, :].values * v
                    else:
                        emb += self.da.loc[k, :].values
                except TypeError as e:
                    emb = np.NaN
                    print("try add {} and {}".format(emb, self.da.loc[k, :].values),
                          "with dtype {} and {}".format(np.array(emb).dtype, np.array(self.da.loc[k, :].values).dtype),
                          "with size {} and {}".format(np.array(emb).shape, np.array(self.da.loc[k, :].values).shape),
                          "The preprocessing cannot be convert in to float, "
                          "or the preprocessing after func are with different size.",
                          "please check you func, which keep the number of results consistent.")
                    raise e

            features.append(emb)
        if len(atoms) == 1:
            return np.array(features).ravel()
        else:
            return np.array(features).reshape((len(atoms), -1))

    def convert_number(self, atoms: List[int]) -> np.ndarray:
        """
        Convert atom number list to numeric features
        """
        return self.dax[atoms, :]

    def __add__(self, other):
        if isinstance(other, AtomTableMap):
            assert other.search_tp == self.search_tp, "should be same"
            if self.search_tp == "number":
                self.dax = np.concatenate((self.dax, other.dax), axis=1)
                self.da = None
            else:
                self.da = pd.concat((self.da, other.da), axis=1)
                self.dax = None
        else:
            raise TypeError("only same class can be added.")
        return self

    @property
    def feature_labels(self):
        if self.da_columns is not None:
            return self.da_columns


def process_uni(i):
    """Post-processing for bool preprocessing General properties."""
    dataii = []

    if i is None:
        dataii.append(0.0)

    elif isinstance(i, (list, tuple)):
        if not i:
            dataii.append(0.0)
        else:
            [dataii.extend(process_uni(ii)) for ii in i]

    elif isinstance(i, (int, float)):
        dataii.append(i)
    elif isinstance(i, str):
        dataii.append(i)
    elif isinstance(i, np.ndarray):
        dataii.extend(i.tolist())
    elif isinstance(i, dict):
        if i == {}:
            dataii.append(0.0)
        else:
            dataii.extend(list(i.values()))
    elif isinstance(i, pd.DataFrame):
        dataii.extend(i.values.ravel())
    elif isinstance(i, np.ndarray):
        if i is np.NaN or np.isnan(i):
            dataii.append(0.0)
        else:
            raise TypeError("The {} is cant serialized.".format(i))
    else:
        raise TypeError("The {} is cant serialized.".format(i))

    return dataii


def process_atomic_orbitals(o):
    """Post-processing for dict preprocessing with "1s", "2p", ..."""
    if o is None or {}:
        return [0.0] * 18
    orb = ("1s", "2p", "2s", "3d", "3p", "3s", "4d", "4f", "4p", "4s", "5d", "5f", "5p", "5s", "6d", "6p", "6s", "7s")
    return [o[orbi] if orbi in o else 0.0 for orbi in orb]


def process_tuple_oxidation_states(ox, size=10):
    """Post-processing for tuple of float preprocessing."""
    if ox is None or ():
        return [0.0] * size
    ox = list(ox)
    ox.extend([0.0] * size)
    return ox[:size]


def process_tuple_full_electronic_structure(full_e):
    """Post-processing for electronic_structure preprocessing ( (1,"s",2),...   )"""
    if full_e is None or ():
        return [0.0] * 18
    dic_all = {"{}{}".format(i, j): k for i, j, k in full_e}
    return process_atomic_orbitals(dic_all)


def process_bool_transition_metal(tm):
    """Post-processing for bool preprocessing"""
    if tm is None:
        return [-1, ]
    elif tm is True:
        return [1, ]
    else:
        return [0, ]


after_treatment_func_map_ele = {
    "atomic_orbitals": process_atomic_orbitals,
    "atomic_radius": process_uni,
    "atomic_mass": process_uni,
    "Z": process_uni,
    "X": process_uni,
    "number": process_uni,
    "max_oxidation_state": process_uni,
    "min_oxidation_state": process_uni,
    "oxidation_states": functools.partial(process_tuple_oxidation_states, size=10),
    "common_oxidation_states": functools.partial(process_tuple_oxidation_states, size=10),
    "full_electronic_structure": process_tuple_full_electronic_structure,
    "row": process_uni,
    "group": process_uni,
    "is_transition_metal": process_bool_transition_metal,
    "is_post_transition_metal": process_bool_transition_metal,
    "is_metalloid": process_bool_transition_metal,
    "is_alkali": process_bool_transition_metal,
    "is_alkaline": process_bool_transition_metal,
    "is_halogen": process_bool_transition_metal,
    "is_lanthanoid": process_bool_transition_metal,
    "is_actinoid": process_bool_transition_metal,
    "atomic_radius_calculated": process_uni,
    "van_der_waals_radius": process_uni,
    "mendeleev_no": process_uni,
    "molar_volume": process_uni,
    "thermal_conductivity": process_uni,
    "boiling_point": process_uni,
    "melting_point": process_uni,
    "bulk_modulus": process_uni,
    "youngs_modulus": process_uni,
    "brinell_hardness": process_uni,
    "rigidity_modulus": process_uni,
    "critical_temperature": process_uni,
    "density_of_solid": process_uni,
    "coefficient_of_linear_thermal_expansion": process_uni,
    "average_ionic_radius": process_uni,
    "average_cationic_radius": process_uni,
    "average_anionic_radius": process_uni,
    "ionic_radii": functools.partial(process_tuple_oxidation_states, size=6),
}


class AtomPymatgenPropMap(BinaryMap):
    """
    Get pymatgen element preprocessing.
    prop_name = [
    "atomic_radius",
    "atomic_mass",
    "number",
    "max_oxidation_state",
    "min_oxidation_state",
    "row",
    "group",
    "atomic_radius_calculated",
    "mendeleev_no",
    "critical_temperature",
    "density_of_solid",
    "average_ionic_radius",
    "average_cationic_radius",
    "average_anionic_radius",]

    Examples
    --------
    >>> tmps = AtomPymatgenPropMap(search_tp="number",prop_name=["X"])
    >>> s = [1,76]
    >>> a = tmps.convert(s)
    >>> tmps = AtomPymatgenPropMap(search_tp="name",prop_name=["X"])
    >>> s = [{"H": 2, }, {"Po": 1}]  #[i.species.as_dict() for i in pymatgen.structure.sites]
    >>> a = tmps.convert(s)
    >>> tmps = AtomPymatgenPropMap(search_tp="name",prop_name=["X"])
    >>> s = [[{"H": 2, }, {"Po": 1}],[{"H": 2, }, {"Po": 1}]]
    >>> a = tmps.transform(s)

    """

    def __init__(self, prop_name: Union[str, List[str]], func: Callable = None, search_tp: str = "name", **kwargs):
        """

        Args:
            prop_name: (str,list of str)
                prop name or list of prop name
            func: (callable or list of callable)
                please make sure the size of it is the same with prop_name.
            search_tp: (str)
                location method.
                "name" for dict
                "number" for int.
        """
        super(AtomPymatgenPropMap, self).__init__(search_tp=search_tp, **kwargs)

        if isinstance(prop_name, (list, tuple)):
            self.prop_name = list(prop_name)
        else:
            self.prop_name = [prop_name, ]

        if func is None:
            func = len(self.prop_name) * [process_uni]
        if isinstance(func, (list, tuple)):
            self.func = list(func)
        else:
            self.func = [func, ]

        if len(self.func) == 1 and len(self.prop_name) > 1:
            self.func *= len(self.prop_name)

        assert len(self.prop_name) == len(self.func), "The size of prop and func should be same."
        self.func = [process_uni if i is None else i for i in self.func]

        for i, j in enumerate(self.prop_name):
            if j in after_treatment_func_map_ele:
                self.func[i] = after_treatment_func_map_ele[j]
        self.da = [Element.from_Z(i) for i in range(1, 119)]
        self.da.insert(0, np.nan)  # for start from 1
        self.ele_map = []

    def convert_dict(self, atoms: List[Dict]) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        if isinstance(atoms, dict):
            atoms = [{k: v} for k, v in atoms.items()]

        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                k = ALL_ELE_N_MAP[k]
                data_all = []
                datai = [getattr(self.da[k], pi) for pi in self.prop_name]
                datai = [self.func[i](j) for i, j in enumerate(datai)]
                [data_all.extend(i) for i in datai]
                try:
                    if self.weight:
                        emb += np.array(data_all) * v
                    else:
                        emb += np.array(data_all)
                except BaseException:
                    raise TypeError("try add {} and {}".format(emb, datai),
                                    "with dtype {} and {}".format(np.array(emb).dtype, np.array(datai).dtype),
                                    "with size {} and {}".format(np.array(emb).shape, np.array(datai).shape),
                                    "The preprocessing cannot be convert in to float, "
                                    "or the preprocessing after func are with different size.",
                                    "please check you func, which keep the number of results consistent.")

            features.append(emb)
        if len(set([i.shape for i in features])) != 1:
            print("The preprocessing after func are with different size.",
                  "please check you func, which keep the number of results consistent.")
        if len(atoms) == 1:
            return np.array(features).ravel()
        else:
            return np.array(features).reshape((len(atoms), -1))

    def convert_number(self, atoms: List[int]) -> np.ndarray:
        """
        Convert int list to numeric features
        """
        features = []
        for atom in atoms:
            data_all = []
            datai = [getattr(self.da[int(round(atom))], pi) for pi in self.prop_name]
            datai = [self.func[i](j) for i, j in enumerate(datai)]
            [data_all.extend(i) for i in datai]
            emb = np.array(data_all)
            features.append(emb)
        if len(set([i.shape for i in features])) != 1:
            print("The preprocessing after func are with different size.",
                  "please check you func, which keep the number of results consistent.")
        if len(atoms) == 1:
            return np.array(features).ravel()
        else:
            return np.array(features).reshape((len(atoms), -1))

    def __add__(self, other):
        if isinstance(other, AtomPymatgenPropMap):
            assert other.search_tp == self.search_tp, "should be same"
            self.prop_name.extend(other.prop_name)
            self.func.extend(other.func)
        else:
            raise TypeError("only same class can be added.")
        return self

    @property
    def feature_labels(self):
        return self.prop_name


after_treatment_func_map_structure = {
    "atomic_orbitals": process_atomic_orbitals,
    "density": process_uni,
}


def _getter_arr(obj, pi):
    """Get prop.
    """
    if "." in pi:
        pis = list(pi.split("."))
        pis.reverse()
        while len(pis):
            s = pis.pop()
            obj = _getter_arr(obj, s)
        return obj
    elif "()" in pi:
        return getattr(obj, pi[:-2])()

    else:
        return getattr(obj, pi)


class _StructurePymatgenPropMap(BaseFeature):
    """
    Get property of pymatgen structure preprocessing.

    Examples
    -----------
    tmps = StructurePymatgenPropMap()

    """

    def __init__(self, prop_name=None, func: Callable = None, return_type="df", **kwargs):
        """

        Args:
            prop_name:(str,list of str)
                prop name or list of prop name
                default ["density", "volume", "ntypesp"]
            func:(callable or list of callable)
                please make sure the size of it is the same with prop_name.
        """
        super(_StructurePymatgenPropMap, self).__init__(return_type=return_type, **kwargs)

        if prop_name is None:
            prop_name = ["density", "volume", "ntypesp"]
        if isinstance(prop_name, (list, tuple)):
            self.prop_name = list(prop_name)
        else:
            self.prop_name = [prop_name, ]

        if func is None:
            func = len(self.prop_name) * [process_uni]
        if isinstance(func, (list, tuple)):
            self.func = list(func)
        else:
            self.func = [func, ]

        if len(self.func) == 1 and len(self.prop_name) > 1:
            self.func *= len(self.prop_name)

        assert len(self.prop_name) == len(self.func), "The size of prop and func should be same."
        self.func = [process_uni if i is None else i for i in self.func]

        for i, j in enumerate(self.prop_name):
            if j in after_treatment_func_map_structure:
                self.func[i] = after_treatment_func_map_structure[j]
        self.lengths = []

    def convert(self, structure: [Structure, Lattice]) -> np.ndarray:
        data_all = []
        datai = []
        for pi in self.prop_name:
            datai.append(_getter_arr(structure, pi))
        datai = [self.func[i](j) for i, j in enumerate(datai)]
        if not self.lengths:
            self.lengths = [len(i) if isinstance(i, (np.ndarray, tuple, list)) else 1 for i in datai]
        [data_all.extend(i) for i in datai]

        return np.array(data_all).ravel()

    def __add__(self, other):
        if isinstance(other, AtomPymatgenPropMap):
            self.prop_name.extend(other.prop_name)
            self.func.extend(other.func)
        else:
            raise TypeError("only same class can be added.")
        return self

    @property
    def feature_labels(self):
        nall = []
        for p, l in zip(self.prop_name, self.lengths):
            if l == 1:
                ni = [p, ]
            else:
                ni = []
                for i in range(l):
                    ni.append("_".join([str(p), str(i)]))

            nall.extend(ni)
        self._feature_labels = nall
        return self._feature_labels
