Custom Features
===============

Multiple strategy combinations::

    >>> ### atom part ####
    >>> func_map = [
    ...    "atomic_radius",
    ...    "max_oxidation_state",
    ...    "min_oxidation_state",
    ...     "atomic_radius_calculated",
    ...    "critical_temperature",
    ...    "density_of_solid",
    ...    "average_ionic_radius",
    ...    "average_cationic_radius",
    ...    "average_anionic_radius",]

Custom atom Features::

>>> from featurebox.featurizers.atom import mapper
>>> from featurebox.featurizers.base_feature import ConverterCat
>>> appa1 = mapper.AtomPymatgenPropMap(prop_name=func_map, search_tp="number")
>>> appa2 = mapper.AtomTableMap(tablename="ele_table.csv", search_tp="number")
>>> appa = ConverterCat(appa1, appa2)

Custom state Features::

>>> apps1 = state_mapper.StructurePymatgenPropMap(prop_name=["density", "volume", "ntypesp"])

Custom bond Features::

>>> appb1 = BaseDesGet(nn_strategy="SOAP", numerical_tol=1e-8, cutoff=None, cut_off_name=None)

