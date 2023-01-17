import unittest

import numpy as np
import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap, AtomPymatgenPropMap
from featurebox.featurizers.base_feature import ConverterCat
from test.structure_data.get_dataset import data01


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = ce.check(self.data)[:10]
        self.data0_checked = ce.check(self.data)[:10]

    def test_get9(self):
        tmps = AtomTableMap(tablename=None)
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)

    def test_get66(self):
        tmps = AtomTableMap(tablename=None)
        s = [i.species.as_dict() for i in self.data0_3[0].sites]
        b = tmps.convert(s)
        print(b.shape)

    def test_get67(self):
        tmps = AtomTableMap(search_tp="name", tablename="ele_table.csv",n_jobs=2)
        s = [[{"H": 2, }, {"Po": 1}], [{"H": 2, }, {"Po": 1}]]
        a = tmps.transform(s)

    def test_cat(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomTableMap(search_tp="name"))
        s = [{"H": 2, }, {"Po": 1}]
        b = tmps.convert(s)
        assert b.shape[1] > 20

    def test_cat2(self):
        tmps = ConverterCat(AtomTableMap(search_tp="number"), AtomTableMap(search_tp="number"))
        s = [1, 2]
        b = tmps.convert(s)
        assert b.shape[1] > 20

    def test_cat3(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomJsonMap())
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] > 20

    def test_cat4(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomJsonMap("ie.json"))
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] >= 20

    def test_cat5(self):
        tmps = ConverterCat(AtomJsonMap(), AtomJsonMap("ie.json"))
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] >= 17

    def test_equal(self):
        tmps = AtomJsonMap("ie.json",n_jobs=2)
        s = [{"H": 1, }, {"Pd": 1}]
        b = tmps.convert(s)
        tmps = AtomJsonMap("ie.json", search_tp="number")
        s = [1, 46]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))

    def test_equal2(self):
        tmps = AtomTableMap(search_tp="name",n_jobs=2)
        s = [{"As": 1, }, {"U": 1}]
        b = tmps.convert(s)
        tmps = AtomTableMap(search_tp="number",n_jobs=2)
        s = [33, 92]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))

    def test_equal3(self):
        tmps = AtomPymatgenPropMap("X", search_tp="name",n_jobs=2)
        s = [{"As": 1, }, {"U": 1}]
        b = tmps.convert(s)
        tmps = AtomPymatgenPropMap("X", search_tp="number")
        s = [33, 92]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))

    def test_equal4(self):
        tmps = AtomPymatgenPropMap("X", search_tp="name",n_jobs=2)
        s = [{"As": 1, }, {"U": 1}]
        b = tmps.convert(s)
        tmps = AtomPymatgenPropMap("X", search_tp="number")
        s = [33, 92]
        c = tmps.convert(s)
        s = [[{"As": 1, }, {"U": 1}], [{"As": 1, }, {"U": 1}]]
        self.assertTrue(np.all(np.equal(b, c)))
        tmps = AtomPymatgenPropMap("X", search_tp="name")
        c = tmps.transform(s)


if __name__ == '__main__':
    unittest.main()
