import unittest

import pandas as pd
from mgetool.tool import def_pwd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.base_graph_geo import StructureGraphGEO
from featurebox.featurizers.generator_geo import InMemoryDatasetGeo, DatasetGEO
from test.structure_data.get_dataset import data01


class TestGraph3(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_3 = self.data[:3]
        ce = CheckElements.from_pymatgen_structures()

        self.data0_checked = ce._check(self.data[:20])

    def test_CrystalGraph(self):
        for i in self.data0_checked:
            sg1 = StructureGraphGEO(nn_strategy="find_points_in_spheres",
                                    bond_generator=None,
                                    atom_converter=None,
                                    bond_converter=None,
                                    state_converter=None,
                                    cutoff=5.0, )
            s12 = sg1(i)
            # print(s12)
            print(s12["edge_index"].shape[-1])
            print(s12["edge_attr"].shape)

    def test_CrystalGraph3(self):
        imdg = InMemoryDatasetGeo(".", load_mode="i")

        l = imdg[2]

    def test_CrystalGraph32(self):
        imdg = InMemoryDatasetGeo(".", load_mode="i", re_process_init=False)

        l = imdg[2]

    def test_CrystalGraph4(self):
        def_pwd("./raw", change=False)

        sg1 = StructureGraphGEO(nn_strategy="find_points_in_spheres",
                                bond_generator=None,
                                atom_converter=None,
                                bond_converter=None,
                                state_converter=None,

                                cutoff=5.0, )
        sg1.transform_and_save(self.data0_checked, save_mode="i")

        imdg = DatasetGEO(".", load_mode="i", re_process_init=True)

        l = imdg[2]
        l = imdg[2]

    def test_CrystalGraph42(self):
        def_pwd("./raw", change=False)

        sg1 = StructureGraphGEO(nn_strategy="find_xyz_in_spheres",
                                bond_generator=None,
                                atom_converter=None,
                                bond_converter=None,
                                state_converter=None,

                                cutoff=2.0, )
        sg1.transform_and_save(self.data0_checked, save_mode="i")
        imdg = DatasetGEO(".", load_mode="i", re_process_init=False)

        l = imdg[2]
        l = imdg[2]

    def test_CrystalGraph43(self):
        def_pwd("./raw", change=False)

        sg1 = StructureGraphGEO(nn_strategy="SOAP",
                                bond_generator=None,
                                atom_converter=None,
                                bond_converter=None,
                                state_converter=None,

                                cutoff=2.0, )
        sg1.transform_and_save(self.data0_checked, save_mode="i")
        imdg = DatasetGEO(".", load_mode="i", re_process_init=False)

        l = imdg[2]
        l = imdg[2]

    def test_CrystalGraph44(self):
        def_pwd("./raw", change=False)

        sg1 = StructureGraphGEO(nn_strategy="CrystalNN",
                                bond_generator=None,
                                atom_converter=None,
                                bond_converter=None,
                                state_converter=None,

                                cutoff=2.0, )
        sg1.transform_and_save(self.data0_checked, save_mode="i")
        imdg = DatasetGEO(".", load_mode="i", re_process_init=False)

        l = imdg[2]
        l = imdg[2]


if __name__ == '__main__':
    unittest.main()
