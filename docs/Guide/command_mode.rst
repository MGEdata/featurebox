Command Mode for Extractor
===========================

The Extractor include:

``bandgap``, ``dbc``, ``bader``, ``cohp``, ``dos``, ``general``, ``diff``, ``converge``

Using
---------------

1. Run in command line mode (suggested). All message (help) could get by ``'-h'`` .

Examples::

    $ featurebox bandgap -h

    $ fbx bandgap -h

    $ featurebox bandgap -p /home/parent_dir

    $ featurebox bandgap -f /home/parent_dir/paths.temp

Use ``fbx -h`` or ``fbx {sub_cmd} -h`` for more details.

2. Run in python for more customization.

>>> from featurebox.cli.vasp_dos import DosxyzPathOut
>>> dosxyz = DosxyzPathOut(n_jobs=4, store_single=True)
>>> result = dosxyz.transfrom(paths_list)
>>> # More part: The following is not in command model.
>>> # final treatment to extractor need message and formatting.
>>> features = dosxyz.extractor(result,atoms=[0, 1, 2, 3], ori=["p-x","d-xy"],format_path=None)

Key
---------------
1. All the sub-extractor such as 'bader' are offer 2 input method in command mode.

**For single case**:

If one path is offered by ``-p`` for ``Single Case`` (default, and in WORKPATH), please make sure the necessary files
under the path exists.

**For batching**:

If paths is offered by ``-f`` for ``Batching``, please make sure the file such as 'paths.temp' exits and not empty.

Generated one file by ``findpath`` command in mgetool package  is suggest. use ``findpath`` command
directly now, to get all sub-folder in current path.

Or use ``findpath -h`` for more help.

2. Some Extractor tools are need third-party tools. please download and installed them in advance.


=========    ================================================
Property     Name
---------    ------------------------------------------------
dbc          `vaspkit <https://vaspkit.com/installation.html#download>`_ <=1.2.1
cohp         `lobster <http://www.cohp.de/>`_
bader        `bader <http://theory.cm.utexas.edu/henkelman/code/bader>`_
bader        `chgsum.pl <http://theory.cm.utexas.edu/vtsttools/download.html>`_
chg_diff     `chgdiff.pl <http://theory.cm.utexas.edu/vtsttools/download.html>`_
=========    ================================================












