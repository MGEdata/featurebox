import unittest

from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap
from test.structure_data.get_dataset import data03


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.st = data03
        self.st2 = [data03, data03]

    def test_convert(self):
        atm = AtomTableMap()
        a = atm.convert(self.st)
        print(a)

    def test_convert3(self):
        atm = AtomTableMap(search_tp="ion_name")
        try:
            a = atm.convert(self.st)
        except (KeyError, ValueError) as e:
            print(e)

    def test_convert2(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.convert(self.st)

    def test_tra(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.transform(self.st2)
