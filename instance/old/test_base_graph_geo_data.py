import unittest

import pandas as pd
from mgetool.tool import def_pwd
from torch_geometric.data import DataLoader

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.base_graph_geo import BaseStructureGraphGEO, StructureGraphGEO
from test.structure_data.get_dataset import data01


class TestGraph3(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_3 = self.data[:3]
        ce = CheckElements.from_pymatgen_structures()

        self.data0_checked = ce._check(self.data)[:10]

    def test_data2(self):
        def_pwd("./raw", change=False)

        sg1 = BaseStructureGraphGEO()

        data_list = sg1.transform_and_to_data(self.data0_checked)
        loader = DataLoader(data_list, batch_size=3)
        for i in loader:
            print(i)

    def test_data(self):
        def_pwd("./raw", change=False)

        sg1 = StructureGraphGEO(nn_strategy="find_points_in_spheres",
                                bond_generator=None,
                                atom_converter=None,
                                bond_converter=None,
                                state_converter=None,
                                cutoff=5.0, )
        sg1.transform_and_save(self.data0_checked, save_mode="i")


if __name__ == '__main__':
    unittest.main()
