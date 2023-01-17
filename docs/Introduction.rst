Introduction
==================

.. image:: img.jpg

Featurebox contains some tools (**Generation** and **Selection**) for material features.
**Generation** is used for feature generation in batch model. **Selection** is used for feature selection.

And one **Extractor** in command line mode is add to obtain some special properties in batch model.
The special properties need certain subsequent computational processing or third-party software participation.

In total, ``Batching`` is the central idea of this module. All works are for convenient data manipulation.

Generation tools
-----------------------

================================================         =========================================
 Name                                                    Application
------------------------------------------------         -----------------------------------------
 :mod:`featurebox.featurizers.atom.mapper`               ``atom`` Getting each element data of compound.
 :mod:`featurebox.featurizers.envir`                     ``bond`` Getting local environment data of compound.
 :mod:`featurebox.featurizers.state`                     ``state`` Getting holistic compound data.
 :mod:`featurebox.featurizers.bond.expander`             Tools to transforming pure bond data.
 :mod:`featurebox.featurizers.batch_feature`             A built-in goofy tool for generating features.
 :class:`featurebox.data.namesplit.NameSplit`            Dividing compound names to elemental proportion table.
 :class:`featurebox.data.mp_access.MpAccess`             Getting data from pymatgen conveniently.
================================================         =========================================

All the **Generation** tools with  ``convert`` method for single case.
and ``fit_transform`` methods for case list.

Guide: :doc:`Guide/data_type`


Binding selection tools
-----------------------------------------

======================================================= =========================================
 Name                                                   Application
------------------------------------------------------- -----------------------------------------
 :class:`featurebox.selection.backforward.BackForward`  Backforward selection
 :class:`featurebox.selection.corr.Corr`                Correlation selection.
 :class:`featurebox.selection.exhaustion.Exhaustion`    Exhaustion selection.
 :class:`featurebox.selection.ga.GA`                    Genetic algorithm selection.
======================================================= =========================================

All the **Selection** tools are ``sklearn-type``, with ``fit``, ``fit_transform`` methods .etc.

.. note::

    Where the binding means treat the binding features as one feature.
    And the binding features are selected or deleted synchronously.

Guide: :doc:`Guide/bind_selection`


Property batching extractor
-----------------------------------------

======================================================= =========================================
 Name                                                   Application
------------------------------------------------------- -----------------------------------------
 :mod:`featurebox.cli.vasp_bader`                       Bader Charge
 :mod:`featurebox.cli.vasp_cohp`                        COHP
 :mod:`featurebox.cli.vasp_dbc`                         band center
 :mod:`featurebox.cli.vasp_dos`                         DOS for plot
 :mod:`featurebox.cli`                                  **More** ...
======================================================= =========================================

All the **Extractor**  tools with  ``convert`` method for single case.
and ``fit_transform`` methods for case list.

Guide: :doc:`Guide/command_mode`

.. note::

    The properties batching extractor are suggested to use ``Command line mode`` .
    But interactive model is still available for more customized operation.

.. note::

    The **Graph neural network** have been removed to ``pyg_extension`` package,
    which employ **envir**, **bond** and **atom** .etc to build input data.




