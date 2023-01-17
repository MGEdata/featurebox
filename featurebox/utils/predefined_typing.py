"""Define several types for convenient use"""

from typing import Union, List, Tuple

from pymatgen.core import Structure, Molecule

StructureOrMolecule = Union[Structure, Molecule]
ListTuple = Union[List, Tuple]
