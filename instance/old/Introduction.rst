Introduction
==================

.. image:: img.jpg

Featurebox contains some tools (**Generation** and **Selection**) for material features.

Generation tools
-----------------------

================================================         =========================================
 Name                                                    Application
------------------------------------------------         -----------------------------------------
 :mod:`featurebox.featurizers.atom.mapper`               ``atom`` Getting each element data of compound.
 :mod:`featurebox.featurizers.envir`                     ``bond`` Getting local environment data (bond and state) of compound.
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

All the selection tools are ``sklearn-type``, with ``fit``, ``fit_transform`` methods .etc.

Guide: :doc:`Guide/bind_selection`

.. note::

    Where the binding means treat the binding features as one feature.
    And the binding features are selected or deleted synchronously.


Graph neural network tools
-------------------------------------------

Featurebox integrated with **Graph neural network**.

===================================================================== =========================================
 Name                                                                 Application
--------------------------------------------------------------------- -----------------------------------------
 :class:`featurebox.featurizers.base_graph_geo.StructureGraphGEO`     Integrated structure data tool. (high dimensional data)
 :class:`featurebox.models_geo.cgcnn.CrystalGraphConvNet`             Traditional graph neural network.
 :class:`featurebox.models_geo.flow_geo.LearningFlow`                 Script for modeling (recommended customization by user).
 :class:`featurebox.featurizers.generator_geo.InMemoryDatasetGeo`     ``Dataset`` for data data in a non-uniform format. (Torch-like)
===================================================================== =========================================


The neural network tools are ``torch-type``, with ``forward`` methods .etc.

Guide: :doc:`Guide/geo_data`

.. note::

    The **Graph neural network** employ **envir**, **bond** and **atom** .etc to build input data.




