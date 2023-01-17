from collections import abc
from typing import Tuple, Union, List

import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from featurebox.utils.predefined_typing import StructureOrMolecule


def _re_pbc(pbc: Union[bool, List[bool], Tuple[bool], np.ndarray], return_type="bool"):
    if pbc is True:
        pbc = [1, 1, 1]
    elif pbc is False:
        pbc = [0, 0, 0]
    elif isinstance(pbc, abc.Iterable):
        pbc = [1 if i is True or i == 1 else 0 for i in pbc]
    else:
        raise TypeError("Can't accept {}".format(pbc))
    if return_type == "bool":
        pbc = np.array(pbc) == 1
    else:
        pbc = np.array(pbc)
    return pbc


def get_radius_in_spheres(
        structure: StructureOrMolecule, nn_strategy=None, cutoff: float = 5.0,
        numerical_tol: float = 1e-6,
        pbc: Union[bool, Tuple[bool]] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff.

    Args:
        pbc (tuple of bool): pbc
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance
        nn_strategy(str,None):not used
    Returns:
        center_indices, neighbor_indices, images, distances, center_prop


    """
    _ = nn_strategy

    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        if pbc is not False:
            pbc = _re_pbc(pbc, return_type="int")
        else:
            pbc = np.array([0, 0, 0])
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0])
    else:
        raise ValueError("structure type not supported")

    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    # images = images.astype(np.int64)
    distances = distances.astype(np.float32)
    exclude_self = (distances > numerical_tol)
    # exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)

    return center_indices[exclude_self], neighbor_indices[exclude_self], \
           distances[exclude_self].reshape(-1, 1), distances[exclude_self], np.array(np.NaN)

# if __name__ == "__main__":
#     from mgetool.tool import tt
#
#     structure = Structure.from_file("../../data/temp_test_structure/W2C.cif")
#     tt.t
#     get_points_ = get_radius_in_spheres(structure,
#                                         cutoff=5.0, )
#     tt.t  #
#     tt.p
