import itertools
from collections import abc
from typing import Tuple, Union, List

import math
import numpy as np

from featurebox.utils.predefined_typing import StructureOrMolecule


def _re_pbc(pbc: Union[bool, List[bool], np.ndarray], return_type="bool"):
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


def get_xyz_in_spheres(structure: StructureOrMolecule, nn_strategy=None, cutoff: float = 5.0,
                       numerical_tol: float = 1e-8,
                       pbc=True,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff.

    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        pbc:bool or list of bool
        numerical_tol (float): numerical tolerance
        nn_strategy(str,None):not used
    Returns:
        center_indices, neighbor_indices, images, distances,center_prop
    """
    _ = nn_strategy
    pbc = _re_pbc(pbc, return_type="bool")
    return not_structure_get_xyz_in_spheres(structure.cart_coords,
                                            reciprocal_lattice_abc=structure.lattice.reciprocal_lattice.abc,
                                            matrix=structure.lattice.matrix,
                                            inv_matrix=structure.lattice.inv_matrix,
                                            pbc=pbc,
                                            cutoff=cutoff, numerical_tol=numerical_tol)


def not_structure_get_xyz_in_spheres(
        all_coords: np.ndarray,
        reciprocal_lattice_abc,
        matrix,
        inv_matrix,
        cutoff: float,
        pbc: Union[bool, List[bool]] = True,
        numerical_tol: float = 1e-8,

):
    """
    For each point in `center_coords`, get all the neighboring points in `all_coords` that are within the
    cutoff radius `r`.

    Args:
        all_coords: (list of cartesian coordinates) all available points
        cutoff: (float) cutoff radius
        pbc: (bool or a list of bool) whether to set periodic boundaries
        numerical_tol: (float) numerical tolerance
        matrix:lattice.matrix
        inv_matrix: lattice.inv_matrix
        reciprocal_lattice_abc:lattice.reciprocal_lattice.abc
    Returns:
        center_indices, neighbor_indices, images, distances,center_prop

    """

    center_coords = all_coords
    pbc = np.array(pbc, dtype=np.bool_)  # type: ignore
    center_coords_min = np.min(center_coords, axis=0)
    center_coords_max = np.max(center_coords, axis=0)
    # The lower bound of all considered atom coords
    global_min = center_coords_min - cutoff - numerical_tol
    global_max = center_coords_max + cutoff + numerical_tol
    ind = np.arange(len(all_coords))
    if np.any(pbc):
        recp_len = np.array(reciprocal_lattice_abc)
        maxr = np.ceil((cutoff + 0.15) * recp_len / (2 * math.pi))
        frac_coords = np.dot(center_coords, inv_matrix)
        nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
        nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
        nmin = np.zeros_like(nmin_temp)
        nmin[pbc] = nmin_temp[pbc]
        nmax = np.ones_like(nmax_temp)
        nmax[pbc] = nmax_temp[pbc]
        all_ranges = [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax)]

        # temporarily hold the fractional coordinates
        image_offsets = np.dot(all_coords, inv_matrix)
        all_fcoords = []
        # only wrap periodic boundary
        for k in range(3):
            if pbc[k]:  # type: ignore
                all_fcoords.append(np.mod(image_offsets[:, k: k + 1], 1))
            else:
                all_fcoords.append(image_offsets[:, k: k + 1])
        all_fcoords = np.concatenate(all_fcoords, axis=1)
        image_offsets = image_offsets - all_fcoords
        coords_in_cell = np.dot(all_fcoords, matrix)
        # Filter out those beyond max range
        valid_coords = []
        valid_images = []
        valid_indices = []

        for image in itertools.product(*all_ranges):
            coords = np.dot(image, matrix) + coords_in_cell
            valid_index_bool = np.all(
                np.bitwise_and(coords > global_min[None, :], coords < global_max[None, :]),
                axis=1,
            )
            if np.any(valid_index_bool):
                valid_coords.append(coords[valid_index_bool])
                valid_images.append(np.tile(image, [np.sum(valid_index_bool), 1]) - image_offsets[valid_index_bool])
                valid_indices.extend([k for k in ind if valid_index_bool[k]])
        if len(valid_coords) < 1:
            raise ValueError("No valid coords element, in cutcoff {}".format(cutoff))
        valid_coords = np.concatenate(valid_coords, axis=0)
        valid_images = np.concatenate(valid_images, axis=0)
        valid_indices = np.array(valid_indices)

    else:
        valid_coords = all_coords  # type: ignore
        valid_images = np.array([[0, 0, 0]] * len(valid_coords))
        valid_indices = np.arange(len(valid_coords))

    center = []
    neighbor = []
    images = []
    coords = []
    lc = center_coords.shape[0]
    lv = valid_coords.shape[0]
    for i in range(lc):
        ci = center_coords[i]
        v = valid_coords - ci
        rs = (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2) ** 0.5
        index = rs <= cutoff

        center.append(np.full(lv, i)[index])
        neighbor.append(valid_indices[index])
        images.append(valid_images[index, :])
        coords.append(v[index, :])

    center_indices = np.concatenate(center, axis=0)
    neighbor_indices = np.concatenate(neighbor, axis=0)
    images = np.concatenate(images, axis=0)
    distances = np.concatenate(coords, axis=0)

    exclude_self = (distances[:, 0] > numerical_tol)

    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[
        exclude_self], np.array(np.NaN)

# if __name__ == "__main__":
#     structure = Structure.from_file("../../data/temp_test_structure/W2C.cif")
#     tt.t
#     get_points_ = get_xyz_in_spheres(structure,
#                                      cutoff=5.0, pbc=True,
#                                      )
#     tt.t  #
#     tt.p
