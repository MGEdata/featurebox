import unittest

try:
    import numba
    nb = True
    from featurebox.featurizers.envir.environment import GEONNGet
except ImportError:
    nb = False


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:

        from featurebox.data.check_data import CheckElements
        from test.structure_data.get_dataset import data02, data01
        ce = CheckElements.from_pymatgen_structures()
        self.data = data01
        self.data2 = data02
        self.data0 = self.data[0]
        self.data0_3 = ce.check(self.data)[:10]
        self.data0_checked = ce.check(self.data)[:10]


    @unittest.skipUnless(nb, "")
    def test_get(self):
        bag = GEONNGet(nn_strategy="VoronoiNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get2(self):
        bag = GEONNGet(nn_strategy="UserVoronoiNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get4(self):
        bag = GEONNGet(nn_strategy="MinimumDistanceNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get19(self):
        bag = GEONNGet(cutoff=5.0, nn_strategy="find_points_in_spheres")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get9(self):
        bag = GEONNGet(nn_strategy="BrunnerNN_reciprocal")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get91(self):
        bag = GEONNGet(nn_strategy="BrunnerNN_real")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get10(self):
        bag = GEONNGet(nn_strategy="EconNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get11(self):
        bag = GEONNGet(nn_strategy="CrystalNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get44(self):
        bag = GEONNGet(nn_strategy="MinimumDistanceNN")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get199(self):
        bag = GEONNGet(cutoff=5.0, nn_strategy="find_points_in_spheres")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get111(self):
        bag = GEONNGet(nn_strategy="BrunnerNN_reciprocal")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get911(self):
        bag = GEONNGet(nn_strategy="BrunnerNN_real")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get110(self):
        bag = GEONNGet(nn_strategy="EconNN")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)

    @unittest.skipUnless(nb, "")
    def test_get1111(self):
        bag = GEONNGet(nn_strategy="CrystalNN")
        for i in self.data0_checked:
            resultt = bag.convert(i)
            print(resultt)


if __name__ == '__main__':
    unittest.main()
