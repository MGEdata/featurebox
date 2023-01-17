import os
import pathlib
import re
import unittest

import pandas as pd

from mgetool.tool import cmd_sys


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        path = pathlib.Path(__file__).parent.parent
        self.path = path / "structure_data" / "data2"
        os.chdir(self.path)
        self.wd = ["Ag/pure_static", "Au/pure_static"]
        self.ff= lambda x: re.split(r" |-|/|\\", x)[-2]+"-Sall"
        self.doping=["Ag","Au"]

    def test_something1(self):
        from featurebox.cli.vasp_bader import BaderStartInter
        bst = BaderStartInter(store_single=True)
        res = bst.transform(self.wd)
        data = bst.extract(res, [0,1,2,3])
        # data = bst.extract(res, [0,1,2,3], format_path=self.ff)
        res = pd.read_csv("bader_all.csv")
        data = bst.extract(res, [0,1,2,3])
        pass

    def test_something2(self):
        from featurebox.cli.vasp_dbc import DBCPy
        method = DBCPy(method="ele")
        res = method.transform(self.wd)
        data = method.extract(res, ele_and_orbit=["C-p","O-p"], join_ele=["Ag","Au"],
                              format_path=self.ff)
        res = pd.read_csv("dbc_py_all.csv")
        data = method.extract(res, ele_and_orbit=["C-p","O-p"], join_ele=["Ag","Au"],)
        pass

    def test_something3(self):
        from featurebox.cli.vasp_dbc import DBCPy
        method = DBCPy(method="atom")
        res = method.transform(self.wd)
        data = method.extract(res, atoms=[0,1,2,3], format_path=self.ff)
        res = pd.read_csv("dbc_py_all.csv")
        data = method.extract(res, atoms=[0, 1, 2, 3], )
        pass

    def test_something4(self):
        from featurebox.cli.vasp_dbc import DBCStartInter
        method = DBCStartInter()
        res = method.transform(self.wd)
        data = method.extract(res, atoms=[0,1,2,3], format_path=self.ff)
        res = pd.read_csv("dbc_all.csv")
        data = method.extract(res,ele=["Mo1","Mo2"],join_ele=[f"{i}1" if i !="Mo" else "Mo18" for i in self.doping],
                              format_path=self.ff)
        pass

    def test_something5(self):
        from featurebox.cli.vasp_dbc import DBCxyzPathOut
        method = DBCxyzPathOut()
        res = method.transform(self.wd)
        data = method.extract(res, atoms=[0, 1, 2, 3], orbit=None)
        res = pd.read_csv("dbc_xyz_all.csv")
        data = method.extract(res, atoms=[0, 1, 2, 3], orbit=["p-x","d-xy"],format_path=self.ff)
        pass

    def test_something6(self):
        from featurebox.cli.vasp_bgp import BandGapPy
        method = BandGapPy()
        res = method.transform(self.wd)
        data = method.extract(res, format_path=None)
        res = pd.read_csv("bgp_all.csv")
        data = method.extract(res, format_path=self.ff)
        pass

    def test_something7(self):

        from featurebox.cli.vasp_cohp import COHPStartInter
        method = COHPStartInter(store_single=True)
        res = method.transform(self.wd)
        data2 = method.extract(res, format_path=self.ff)
        res = pd.read_csv("ICOHP_all.csv")
        data2 = method.extract(res, format_path=None)
        pass


if __name__ == '__main__':
    unittest.main()
