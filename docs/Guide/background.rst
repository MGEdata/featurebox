Sample Data and Background
===========================

Sample data
::::::::::::

Download data from the following link: `Structure List <https://github.com/boliqq07/featurebox/blob/master/test/structure_data/sample_data.pkl_pd>`_ .

Usage:

    >>> import pandas as pd
    >>> from pymatgen.core import Structure
    >>> structures = structure_list = pd.read_pickle("sample_data.pkl_pd")
    >>> structure = structurei = structure_list[0]


Background
::::::::::::

The ``Structure`` from ``pymatgen`` is one class to represent the crystal structure data, which contain all message
of atoms and their sites. More details link:
`pymatgen Structure <https://pymatgen.org/usage.html#reading-and-writing-structures-molecules>`_ .

From this type data, we could extract atom/element names and atom/element numbers message by following code.

Such as for single case (built in ``convert`` function):

  >>> structure_1 = structure_list[0]
  >>> name_1 = [{str(i.symbol): 1} for i in structure_1.species]
  >>> number_1 = [i.specie.Z for i in structure_1]

Such as for batch data (built in ``transform`` function):

  >>> name_list = [[{str(i.symbol): 1} for i in si.species] for si in structure_list]
  >>> number_list = [[i.specie.Z for i in si] for si in structure_list]

In this packages, we accept data with type like ``name_list`` , ``number_list``  as input data for ``transform`` .
Meanwhile, The above code are built in package, thus we could accept ``structure_list`` as input data directly.

.. note::

    In addition, the ``ase.Atoms`` could convert by Adaptor ``AseAtomsAdaptor`` from pymatgen or ``featurebox.utils.general.AAA`` .
    Of course, The data ``name`` data , ``number`` data could build by yourself from you code.
