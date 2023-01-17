Pymatgen Data
=============


The data are using the inner periodical_data.json in pymatgen elemental data.
::

>>> from featurebox.featurizers.atom.mapper import AtomPymatgenPropMap
>>> tmps = AtomPymatgenPropMap(search_tp="name",prop_name = [ "atomic_radius", "atomic_mass", "number", "max_oxidation_state"])
>>> s = [{"H": 2, }, {"Po": 1}, {"C": 2}] # [i.species.as_dict() for i in pymatgen.structure.sites]
>>> a2 = tmps.convert(s) # or
>>> a2 = tmps.convert(structurei)

In addition, we could get structure state data by structure.
::

>>> from featurebox.featurizers.state.state_mapper import StructurePymatgenPropMap
>>> tmps = StructurePymatgenPropMap(prop_name = ["density", "volume", "ntypesp"])
>>> a2 = tmps.convert(structurei)

This second class is for structure but for atoms, and the first one return the each atom features
and the second return the whole feature of structure.



