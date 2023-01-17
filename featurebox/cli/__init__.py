# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:27
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

__doc__ = """
The ``cli`` part include:

    ``bandgap``, ``dbc``, ``bader``, ``cohp``, ``dos``, ``general``, ``diff``, ``converge``

1. Run in command line mode (suggested). All message (help) could get by ``'-h'`` .

Examples::

    $ featurebox bandgap -h 
    
    $ fbx bandgap -h
    
    $ featurebox bandgap -p /home/parent_dir
    
    $ featurebox bandgap -f /home/parent_dir/paths.temp

2. Run in python.

>>> from featurebox.cli.vasp_dos import DosxyzPathOut
>>> dosxyz = DosxyzPathOut(n_jobs=4, store_single=True)
>>> result = dosxyz.transform(paths_list)

"""
