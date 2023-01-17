"""Use descriptors form ``pyXtal_FF``, in :mod:`featurebox.test_featurizers.descriptors`
all with ``calculate`` method.
"""
import copy
from typing import Dict, Tuple, List

import numpy as np
from pymatgen.core import Structure, Molecule

from featurebox.featurizers.envir._get_radius_in_spheres import get_radius_in_spheres
from featurebox.featurizers.envir.descriptors.ACSF import ACSF
from featurebox.featurizers.envir.descriptors.EAD import EAD
from featurebox.featurizers.envir.descriptors.EAMD import EAMD
from featurebox.featurizers.envir.descriptors.SO3 import SO3
from featurebox.featurizers.envir.descriptors.SO4 import SO4_Bispectrum
from featurebox.featurizers.envir.descriptors.SOAP import SOAP
from featurebox.featurizers.envir.descriptors.behlerparrinello import BehlerParrinello
from featurebox.featurizers.envir.descriptors.wACSF import wACSF
from featurebox.utils.general import aaa


def mark_classes(classes: List):
    return {i.__name__: i for i in classes}


DesDict = mark_classes([
    ACSF,
    BehlerParrinello,
    EAD,
    EAMD,
    SOAP,
    SO3,
    SO4_Bispectrum,
    wACSF,
])


def get_strategy2_in_spheres(structure, nn_strategy, cutoff, numerical_tol=None, cutoff_name="cutoff", pbc=True):
    _ = numerical_tol
    if isinstance(structure, Structure) or isinstance(structure, Molecule):
        atoms = aaa.get_atoms(structure)
    else:
        atoms = structure

    nn_strategy_ = copy.copy(nn_strategy)

    setattr(nn_strategy_, cutoff_name, cutoff)

    result = nn_strategy_.calculate(atoms)

    *_, b5 = get_5_result(result)

    a1, a2, a3, a4, _ = get_radius_in_spheres(structure, nn_strategy=None, cutoff=cutoff,
                                              numerical_tol=numerical_tol,
                                              pbc=pbc,
                                              )

    return a1, a2, a3, a4, b5


def get_5_result(d: Dict, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        d: dict, dict of descriptor. at lest contain "x" and "dxdr"
        fill_size: int, unstable.

    Returns:
        (center_indices,center_prop, neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size), None.
        center_prop: np.ndarray 1d(N,l_c).\n

    """
    center = d["x"]
    atom_len = center.shape[0]

    center_indices = np.array(range(atom_len))

    return center_indices, np.array(np.NaN), np.array(np.NaN), np.array(np.NaN), center
