import unittest

try:
    import numba
    nb = True
    from featurebox.featurizers.envir.environment import GEONNGet
    from featurebox.featurizers.envir.local_env import UserVoronoiNN
except ImportError:
    nb = False


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        from featurebox.data.check_data import CheckElements
        from test.structure_data.get_dataset import data01, data02
        ce = CheckElements.from_pymatgen_structures()
        self.data = data01
        self.data2 = data02
        self.data0 = self.data[0]
        self.data0_3 = ce.check(self.data)[:10]
        self.data0_checked = ce.check(self.data)[:10]

    @unittest.skipUnless(nb, "")
    def test_size_xyz(self):
        bag = GEONNGet(cutoff=5.0, nn_strategy="find_xyz_in_spheres")
        for i in self.data0_3:
            center_indices, atom_nbr_idx, bond_states, bonds, center_prop = bag.convert(i)
            print(center_indices.shape)
            print(atom_nbr_idx.shape)
            print(bond_states.shape)
            print(bonds.shape)
            print(center_prop.shape)
            print("next")

    @unittest.skipUnless(nb, "")
    def test_size_radius(self):
        bag = GEONNGet(cutoff=5.0, nn_strategy="find_points_in_spheres")
        for i in self.data0_3:
            center_indices, atom_nbr_idx, bond_states, bonds, center_prop = bag.convert(i)
            print(center_indices.shape)
            print(atom_nbr_idx.shape)
            print(bond_states.shape)
            print(bonds.shape)
            print(center_prop.shape)
            print("next")

    @unittest.skipUnless(nb, "")
    def test_size_strategy(self):
        bag = GEONNGet(cutoff=5.0, nn_strategy="MinimumDistanceNNAll")
        for i in self.data0_3:
            center_indices, atom_nbr_idx, bond_states, bonds, center_prop = bag.convert(i)
            print(center_indices.shape)
            print(atom_nbr_idx.shape)
            print(bond_states.shape)
            print(bonds.shape)
            print(center_prop.shape)
            print("next")
