import unittest

import pandas as pd

from featurebox.featurizers.state.state_mapper import StructurePymatgenPropMap
from test.structure_data.get_dataset import data01


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_something(self):
        sppm = StructurePymatgenPropMap()
        data = sppm.fit_transform(self.data0_3)

    def test_something2(self):
        sppm = StructurePymatgenPropMap(prop_name=["lengths", "angles", "get_brillouin_zone()"])
        data = sppm.fit_transform([i._lattice for i in self.data0_3])

    def test_something3(self):
        sppm = StructurePymatgenPropMap(
            prop_name=["_lattice.lengths", "_lattice.angles", "_lattice.get_brillouin_zone()"])
        data = sppm.fit_transform(self.data0_3)


if __name__ == '__main__':
    unittest.main()
