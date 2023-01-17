import unittest

from featurebox.models_geo.basemodel import GeneralReadOutLayer


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        module = GeneralReadOutLayer((128, 64, "sum", 32, 1))
        print(module)

    def test_something2(self):
        module = GeneralReadOutLayer(("sum", 32, 1))
        print(module)

    def test_something3(self):
        module = GeneralReadOutLayer((128, 64, "sum"))
        print(module)

    def test_something4(self):
        module = GeneralReadOutLayer((128, 64, "sum", 22))
        print(module)


if __name__ == '__main__':
    unittest.main()
