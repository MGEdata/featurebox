# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pymatgen.core import Structure

structures = structure_list = pd.read_pickle(r"./structure_data/sample_data.pkl_pd")

name_data = [[{str(i.symbol): 1} for i in si.species] for si in structure_list]
number_data = [[i.specie.Z for i in si] for si in structure_list]

structure_1 = structure = structurei = structure_list[0]
name_single = [{str(i.symbol): 1} for i in structure_1.species]
number_single = [i.specie.Z for i in structure_1]

from featurebox.featurizers.atom.mapper import AtomTableMap

tmps = AtomTableMap(search_tp="number")
single_sample = [1, 1, 1, 76, 76]
multi_sample = [[1, 1, 1, 76, 76], [3, 3, 4, 4]]
a = tmps.convert(single_sample)

b = tmps.transform(multi_sample)

from featurebox.featurizers.atom.mapper import AtomJsonMap

tmps = AtomJsonMap(search_tp="name", return_type="np")
single_sample = [{"H": 2}, {"P": 1}]
single_sample2 = {"H": 2, "P": 1}
multi_sample = [[{"H": 2}, {"P": 1}], [{"He": 3}, {"P": 4}]]  # or
multi_sample2 = [{"H": 2, "P": 1}, {"He": 3, "P": 4}]
a = tmps.convert(single_sample)
a = tmps.convert(single_sample2)
b = tmps.transform(multi_sample)
b = tmps.transform(multi_sample2)

from pymatgen.core.structure import Structure

structurei = structure_list[0]

from featurebox.featurizers.state.state_mapper import StructurePymatgenPropMap

tmps = StructurePymatgenPropMap(prop_name=["density", "volume", "ntypesp"])
a = tmps.convert(structurei)
b = tmps.transform([structurei] * 10)

from pymatgen.core.structure import Structure

from featurebox.featurizers.atom.mapper import AtomTableMap
from featurebox.featurizers.state.statistics import WeightedAverage

data_map = AtomTableMap(search_tp="name", n_jobs=1)
wa = WeightedAverage(data_map, n_jobs=1, return_type="df")
x3 = [{"H": 2, "Pd": 1}, {"He": 1, "Al": 4}]
wa.fit_transform(x3)
x4 = [structurei] * 5
wa.fit_transform(x4)

from featurebox.featurizers.atom.mapper import AtomJsonMap
from featurebox.featurizers.state.union import UnionFeature
from featurebox.featurizers.state.statistics import DepartElementFeature

data_map = AtomJsonMap(search_tp="name", embedding_dict="ele_megnet.json",
                       n_jobs=1)  # keep this n_jobs=1 and return_type="np"
wa = DepartElementFeature(data_map, n_composition=2, n_jobs=1, return_type="pd")
comp = [{"H": 2, "Pd": 1}, {"He": 1, "Al": 4}]
wa.set_feature_labels(
    ["fea_{}".format(_) for _ in range(16)])  # 16 this the feature number of built-in "ele_megnet.json"
couple_data = wa.fit_transform(comp)

uf = UnionFeature(comp, couple_data, couple=2, stats=("mean", "maximum"))
state_data = uf.fit_transform()

import numpy as np
from featurebox.featurizers.state.union import PolyFeature

state_features = np.array([[0, 1, 2, 3, 4, 5], [0.422068, 0.360958, 0.201433, -0.459164, -0.064783, -0.250939]]).T
state_features = pd.DataFrame(state_features, columns=["f1", "f2"], index=["x0", "x1", "x2", "x3", "x4", "x5"])
pf = PolyFeature(degree=[1, 2])
pf.fit_transform(state_features)

from featurebox.featurizers.batch_feature import BatchFeature

bf = BatchFeature(data_type="structures", return_type="df")
data = bf.fit_transform(structure_list)

from featurebox.featurizers.batch_feature import BatchFeature

bf = BatchFeature(data_type="compositions")
com = [[{str(i.symbol): 1} for i in structurei.species] for structurei in structure_list]
# where com is element list
data = bf.fit_transform(com)

from featurebox.featurizers.batch_feature import BatchFeature

bf = BatchFeature(data_type="elements")
aas = [[{str(i.symbol): 1} for i in structurei.species] for structurei in structure_list]
data = bf.fit_transform(aas)
bf.element_c.search_tp = "number"
aas = [[i.specie.Z for i in structure] for structure in structure_list]
# where aas is element list
data = bf.fit_transform(aas)

from featurebox.featurizers.atom.mapper import AtomJsonMap

tmps = AtomJsonMap(search_tp="number", embedding_dict="ele_megnet.json")
a = tmps.convert(structure)

from featurebox.featurizers.atom.mapper import AtomJsonMap

tmps = AtomJsonMap(search_tp="number", embedding_dict="ele_megnet.json")
s = [1, 76]
a = tmps.convert(s)

from featurebox.featurizers.atom.mapper import AtomJsonMap

tmps = AtomJsonMap(search_tp="name")
s = [{"H": 2, }, {"Al": 1}]
a = tmps.convert(s)

from featurebox.featurizers.atom.mapper import AtomJsonMap

tmps = AtomJsonMap(search_tp="name")
s = [[{"H": 2, }, {"Ce": 1}], [{"H": 2, }, {"Al": 1}]]
a = tmps.transform(s)

tmps = AtomTableMap(search_tp="number", tablename="oe.csv")
a = tmps.convert(structure)

com = [i.species.as_dict() for i in structure.sites]
com = [{str(i.symbol): 1} for i in structure.species]

from featurebox.featurizers.atom.mapper import AtomPymatgenPropMap

tmps = AtomPymatgenPropMap(search_tp="name", prop_name=["atomic_radius", "atomic_mass", "number", "max_oxidation_state",
                                                        "min_oxidation_state", "row", "group",
                                                        "atomic_radius_calculated",
                                                        "mendeleev_no", "critical_temperature", "density_of_solid",
                                                        "average_ionic_radius", "average_cationic_radius",
                                                        "average_anionic_radius", ])
s = [{"H": 2, }, {"Po": 1}, {"C": 2}]  # [i.species.as_dict() for i in pymatgen.structure.sites]
a2 = tmps.convert(s)  # or
a2 = tmps.convert(structurei)

from featurebox.featurizers.state.state_mapper import StructurePymatgenPropMap

tmps = StructurePymatgenPropMap(prop_name=["density", "volume", "ntypesp"])
a2 = tmps.convert(structurei)

from featurebox.featurizers.state.union import PolyFeature

n = np.array([[0, 1, 2, 3, 4, 5], [0.422068, 0.360958, 0.201433, -0.459164, -0.064783, -0.250939]]).T
ps = pd.DataFrame(n, columns=["f1", "f2"], index=["x0", "x1", "x2", "x3", "x4", "x5"])
pf = PolyFeature(degree=[1, 2])
pf.fit_transform(n)

from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap

data_map = AtomJsonMap(search_tp="name", n_jobs=1)
wa = WeightedAverage(data_map, n_jobs=1, return_type="df")
x3 = [{"H": 2, "Pd": 1}, {"He": 1, "Al": 4}]
wa.fit_transform(x3)  # or
wa.fit_transform(structure_list)
wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)])
wa.fit_transform(x3)

### atom part ####
func_map = [
    "atomic_radius",
    "max_oxidation_state",
    "min_oxidation_state",
    "atomic_radius_calculated",
    "critical_temperature",
    "density_of_solid",
    "average_ionic_radius",
    "average_cationic_radius",
    "average_anionic_radius", ]

from featurebox.featurizers.atom import mapper
from featurebox.featurizers.base_feature import ConverterCat

appa1 = mapper.AtomPymatgenPropMap(prop_name=func_map, search_tp="number")
appa2 = mapper.AtomTableMap(tablename="ele_table.csv", search_tp="number")
appa = ConverterCat(appa1, appa2)

from featurebox.data.namesplit import NameSplit
import os

os.chdir(r'../../../Desktop')
name = ['(Ti1.24La3)2', "((Ti1.24)2P2)1H0.2", "((Ti1.24)2)1H0.2", "((Ti1.24))1H0.2", "((Ti)2P2)1H0.2", "((Ti))1H0.2"]
NSp = NameSplit()
NSp.transform(name)

from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from featurebox.selection.backforward import BackForward

X, y = fetch_california_housing(return_X_y=True)
X = X[:100]
y = y[:100]
svr = SVR()
bf = BackForward(svr, primary_feature=4, random_state=1)
new_x = bf.fit_transform(X, y)
bf.support_

from sklearn.datasets import fetch_california_housing
from featurebox.selection.corr import Corr

x, y = fetch_california_housing(return_X_y=True)
x = x[:100]
y = y[:100]
co = Corr(threshold=0.7, multi_index=[0, 8], multi_grade=2)
newx = co.fit_transform(x)
print(x.shape)
print(newx.shape)
# (506, 13)
# (506, 9)

from sklearn.datasets import fetch_california_housing
from featurebox.selection.corr import Corr

x, y = fetch_california_housing(return_X_y=True)
co = Corr(threshold=0.7, multi_index=[0, 8], multi_grade=2)
x = x[:100]
y = y[:100]
co.fit(x)
Corr(multi_index=[0, 8], threshold=0.7)
group = co.count_cof()
group[1]
co.remove_coef(group[1])
