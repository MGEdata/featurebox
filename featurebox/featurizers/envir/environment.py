from pathlib import Path
from typing import Tuple, Union, Dict, List

import numpy as np
from ase import Atoms
from pymatgen.core import Structure

from featurebox.featurizers.base_feature import BaseFeature
from featurebox.featurizers.envir._get_radius_in_spheres import get_radius_in_spheres
from featurebox.featurizers.envir._get_xyz_in_spheres import get_xyz_in_spheres
from featurebox.featurizers.envir.desc_env import DesDict, get_strategy2_in_spheres
from featurebox.featurizers.envir.local_env import NNDict, get_strategy1_in_spheres
from featurebox.utils.general import aaa
from featurebox.utils.predefined_typing import StructureOrMolecule

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


def get_marked_class(nn_strategy, env_dict: Dict = None, instantiation: bool = True):
    """


    Parameters
    ----------
    nn_strategy:Any
        "find_points_in_spheres", "find_xyz_in_spheres" , or nn_strategy
    env_dict:dict
        pre-definition, {"classname": class}.
    instantiation:bool
        return class of object.

    Returns
    -------
    obj:
        object or class in NNDict.

    """
    try:
        # #####old type for compatibility ### #
        if nn_strategy is None:
            return nn_strategy
        elif isinstance(nn_strategy, str) and nn_strategy in ["find_points_in_spheres", "find_xyz_in_spheres"]:
            return nn_strategy
        # ###########by str name############# #
        if isinstance(nn_strategy, str):
            Nei = env_dict[nn_strategy]()
        else:
            try:
                nn_strategy = nn_strategy()
            except TypeError:
                pass
            if nn_strategy.__class__.__name__ in env_dict:
                Nei = nn_strategy
            else:
                raise TypeError("only accept str or object inherit from nn_dict.values()")

        if instantiation:
            return Nei
        else:
            return Nei.__class__

    except (KeyError, TypeError):
        raise TypeError("only accept str or object inherit from nn_dict.values()")


def geo_refine_nn(center_indices, neighbor_indices, vectors, distances,
                  center_prop=None, ele_numbers=None,
                  fill_size=10,
                  dis_sort=True,
                  **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        center_indices: np.ndarray 1d
        neighbor_indices: np.ndarray 1d
        distances: np.ndarray 1d or np.ndarray 2d
        vectors: np.ndarray 2d
        center_prop:np.ndarray 2d
        ele_numbers:np.ndarray 1d
        fill_size: float, not use,
        dis_sort:bool
            sort neighbors with distance.

    Returns:
        (center_indices,center_indices,  neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size,1).
        center_prop: np.ndarray 1d(N,l_c).\n

    where l, and l_c >= 1
    """
    _ = ele_numbers
    _ = fill_size

    if distances.ndim == 2:
        neidx = np.lexsort((center_indices, distances[:, 0],))
    else:
        neidx = np.lexsort((center_indices, distances,))

    cen_nei = center_indices[neidx]
    nei_nei = neighbor_indices[neidx]
    diss = distances[neidx]
    if vectors is not None:
        vecs = vectors[neidx, :]
    else:
        vecs = vectors

    neis = np.vstack((cen_nei, nei_nei))

    toge = np.concatenate((center_indices, neighbor_indices), axis=0).tolist()

    cen = np.array(sorted(set(toge)))

    cen = np.array(cen).ravel()

    if diss.ndim == 1:
        diss = np.array(diss)[..., np.newaxis]
    else:
        diss = np.array(diss)

    if center_prop is None:  # must assort by rank of atom in structure before
        return cen, neis, vecs, diss, np.array(cen).reshape(-1, 1),
    else:
        if center_prop.ndim == 1:
            try:
                center_prop = center_prop[cen]
                center_prop = center_prop.reshape(-1, 1)
            except ValueError:
                raise UserWarning("center_prop is less than center atom number. Try to add cutoff.")
        elif center_prop.ndim == 2 and center_prop.shape[0] == cen.shape[0]:
            center_prop = center_prop[cen]
        else:
            center_prop = np.array(np.NAN)
        return cen, neis, vecs, diss, center_prop


#####################################################################################################################

after_treatment_func_map = {"geo_refine_nn": geo_refine_nn, }
# local env method
env_method = {}
env_method.update(NNDict)
env_method.update(DesDict)


#####################################################################################################################


class _BaseEnvGet(BaseFeature):
    """
    Get properties from Pymatgen.Structure.
    And each structure is convert to data as following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    ``center_properties``:np.ndarray of shape(n,l_c)
        center properties.
    """

    def __init__(self, n_jobs: int, on_errors: str = 'raise', return_type: str = 'any',
                 batch_calculate: bool = False,
                 batch_size: int = 30):
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type,
                         batch_calculate=batch_calculate,
                         batch_size=batch_size)
        self.adaptor_dict = {}  # Dynamic assignment
        self.adaptor = aaa
        self.pbc = True
        self.cutoff = 5.0

    def ase_to_pymatgen(self, atom: Atoms, prop_dict=None):
        """ase_to_pymatgen"""
        if not prop_dict:
            prop_dict = self.adaptor_dict if self.adaptor_dict else {}

        structure = self.adaptor.get_structure(atom)
        if prop_dict != {}:
            [structure.__setattr__(k, v) for k, v in prop_dict.items()]
        return structure

    def pymatgen_to_ase(self, structure: Structure, prop_dict=None):
        """pymatgen_to_ase"""
        if not prop_dict:
            prop_dict = self.adaptor_dict if self.adaptor_dict else {}

        atoms = self.adaptor.get_atoms(structure)
        if prop_dict != {}:
            [atoms.__setattr__(k, v) for k, v in prop_dict.items()]
        return atoms


class GEONNGet(_BaseEnvGet):
    """
    Get properties from Pymatgen.Structure.
    Where the nn_strategy is from ``Pymatgen``.
    And each structure is convert to data as following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    ``center_properties``:np.ndarray of shape(n,l_c)
        center properties.
    """

    def __init__(self, nn_strategy: str = "UserVoronoiNN", refine: str = "geo_refine_nn",
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: Union[List[int], bool] = False, cutoff=5.0, check_align=True,
                 cutoff_name="cutoff",
                 n_jobs=1,
                 on_errors='raise', return_type='any'):
        """

        Parameters
        ----------

        nn_strategy: str
            ["find_points_in_spheres", "find_xyz_in_spheres",
            "BrunnerNN_reciprocal", "BrunnerNN_real", "BrunnerNN_relative",
            "EconNN", "CrystalNN", "MinimumDistanceNNAll", "find_points_in_spheres","UserVoronoiNN",
            "ACSF","BehlerParrinello","EAD","EAMD","SOAP","SO3","SO4_Bispectrum","wACSF",]
        refine:str
            sort method for neighbors of each atom.
            See Also:
            :func:`universe_refine_nn`
        refined_strategy_param:dict
            parameters for refine
        numerical_tol:float
            numerical_tol
        pbc:list
            periodicity in 3 direction
            3-length list,each one is 1 or 0. such as [0,0,0],The 1 mean in this direction is with periodicity.
        cutoff:
            cutoff radius
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.nn_strategy = get_marked_class(nn_strategy, env_method)

        self.refine = after_treatment_func_map[refine]
        self.refined_strategy_param = refined_strategy_param \
            if isinstance(refined_strategy_param, dict) else {}
        self.cutoff = cutoff
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.check_align = check_align

        if isinstance(self.nn_strategy, str):
            self.cutoff_name = cutoff_name

        elif cutoff_name:
            if cutoff:
                setattr(self.nn_strategy, cutoff_name, cutoff)
            self.cutoff_name = cutoff_name

        else:
            if cutoff is not None:
                if hasattr(nn_strategy, "cutoff"):
                    self.nn_strategy.cutoff = cutoff
                    self.cutoff_name = "cutoff"

                elif hasattr(nn_strategy, "Rc"):
                    self.nn_strategy.Rc = cutoff
                    self.cutoff_name = "Rc"

                elif hasattr(nn_strategy, "rcut"):
                    self.nn_strategy.rcut = cutoff
                    self.cutoff_name = "rcut"
                else:
                    raise TypeError("Can't find cut radius name")
            else:
                self.cutoff_name = cutoff_name

        if self.nn_strategy == "find_points_in_spheres":
            self._convert = self.get_radius
        elif self.nn_strategy == "find_xyz_in_spheres":
            self._convert = self.get_xyz
        elif self.nn_strategy.__class__.__name__ in NNDict:
            self._convert = self.get_strategy1
        elif self.nn_strategy.__class__.__name__ in DesDict:
            self._convert = self.get_strategy2
        else:
            raise NameError("can't determining ", self.nn_strategy)

    def convert(self, structure: StructureOrMolecule) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            structure:Structure,pymatgen Structure

        Returns:
            center_indices,center_prop,neighbor_indices,images,distances
        """
        return self._convert(structure, self.cutoff)

    def _check(self, structure, result, cutoff) -> bool:
        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return True
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2)
            )
            return False

    def get_xyz(self, structure: StructureOrMolecule, cutoff
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""
        if cutoff > 15:
            raise ValueError("The cutoff is to large than cutoff.")

        numerical_tol = self.numerical_tol

        result = get_xyz_in_spheres(structure, nn_strategy=None, cutoff=cutoff, numerical_tol=numerical_tol,
                                    pbc=self.pbc)
        # Result: center_indices, neighbor_indices, images, distances, center_prop
        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param, )

        # Return: center_indices, neighbor_indices, images, distances, center_prop
        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return result
        else:

            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2))
            return self.get_xyz(structure, cutoff + 2)

    def get_radius(self, structure: StructureOrMolecule, cutoff
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""
        if cutoff > 15:
            raise ValueError("The cutoff is to large than cutoff.")

        numerical_tol = self.numerical_tol

        result = get_radius_in_spheres(structure, nn_strategy=None, cutoff=cutoff, numerical_tol=numerical_tol,
                                       pbc=self.pbc)
        # Result: center_indices, neighbor_indices, images, distances, center_prop
        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param, )

        # Return: center_indices, neighbor_indices, images, distances, center_prop
        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return result
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2))
            return self.get_radius(structure, cutoff + 2)

    def get_strategy1(self, structure: StructureOrMolecule, cutoff
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_staagy could be rewrite."""
        # assert hasattr(self.nn_strategy, "cutoff")
        if cutoff > 15:
            raise ValueError("The cutoff is to large than cutoff.")

        numerical_tol = self.numerical_tol

        result = get_strategy1_in_spheres(structure, nn_strategy=self.nn_strategy, cutoff=cutoff,
                                          numerical_tol=numerical_tol,
                                          pbc=self.pbc)

        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param)

        if self._check(structure, result, cutoff):
            return result
        else:
            return self.get_strategy1(structure, cutoff + 2)

    def get_strategy2(self, structure: StructureOrMolecule, cutoff
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_staagy could re-write."""

        if cutoff > 15:
            raise ValueError("The cutoff is to large than cutoff.")

        numerical_tol = self.numerical_tol

        result = get_strategy2_in_spheres(structure, nn_strategy=self.nn_strategy, cutoff=cutoff,
                                          numerical_tol=numerical_tol, pbc=self.pbc,
                                          cutoff_name=self.cutoff_name)

        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param)

        if len(result[0]) == len(structure.species) or not self.check_align:
            return result
        else:

            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2)
            )
            return self.get_strategy2(structure, cutoff + 2)
