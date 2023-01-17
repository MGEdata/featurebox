Install
==================

Method 1
::::::::::::

Install with pip ::

    pip install featurebox

.. note::

    If VC++ needed for windows, Please download the dependent packages from
    `Python Extension Packages <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ and install offline.
    Such as ``Spglib``. And try again, or reference to Method 2.

Method 2
::::::::::::

Requirements Packages:

============= =================  ============
 Dependence   Name               Version
------------- -----------------  ------------
 necessary    sympy              >=1.6
 necessary    deap               >=1.3.1
 necessary    scikit-learn       >=0.22.1
 necessary    torch              >=1.5.0
 necessary    ase                \
 necessary    pymatgen           \
 recommend    scikit-image       \
 recommend    minepy             \
 recommend    torch_geometric    \
============= =================  ============


Install by step:

1. sympy ::

    pip install sympy>=1.6

Reference: https://www.sympy.org/en/index.html

2. deap ::

    pip install deap

Reference: https://github.com/DEAP/deap

3. pymatgen ::

    conda install --channel conda-forge pymatgen

Reference: https://github.com/materialsproject/pymatgen

.. note::

    If ``Spblib`` needed, which need C++ to compiled, please download
    `Python Extension Packages <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_
    and pip install locally.

Such as ::

    pip install /your/local/path/spglib-1.16.1-cp38-cp38-win_amd64.whl

4. scikit-learn ::

    conda install sklearn

Reference: https://github.com/materialsproject/pymatgen

5. mgetool::

    pip install mgetool

Reference: https://github.com/Mgedata/mgetool

6. featurebox::

    pip install featurebox

7. ase::

    pip install ase

Reference: https://wiki.fysik.dtu.dk/ase/ , not necessary, just for network.
