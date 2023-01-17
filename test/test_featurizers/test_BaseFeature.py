import unittest

import numpy as np

from featurebox.featurizers.base_feature import BaseFeature, DummyConverter


class MyTestCase(unittest.TestCase):

    def test_transofrm(self):
        bf = BaseFeature(n_jobs=1)
        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [1, 2, 3, 4, 5, 6.0]
        x3 = np.array([[1, 2, 3, 4, 5, 6.0], [1, 2, 3, 4, 5, 6.0]])
        x4 = np.array([[1, 2, 3, 4, 5, 6.0]])
        x5 = np.array([1, 2, 3, 4, 5, 6.0])
        newx = bf.fit_transform(x1)
        print(newx)
        newx = bf.fit_transform(x2)
        print(newx)
        newx = bf.fit_transform(x3)
        print(newx)
        newx = bf.fit_transform(x4)
        print(newx)
        newx = bf.fit_transform(x5)
        print(newx)

    def test_transofrm2(self):
        bf = DummyConverter(n_jobs=1)
        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [1, 2, 3, 4, 5, 6.0]
        x3 = np.array([[1, 2, 3, 4, 5, 6.0], [1, 2, 3, 4, 5, 6.0]])
        x4 = np.array([[1, 2, 3, 4, 5, 6.0]])
        x5 = np.array([1, 2, 3, 4, 5, 6.0])
        newx = bf.fit_transform(x1)
        print(newx)
        newx = bf.fit_transform(x2)
        print(newx)
        newx = bf.fit_transform(x3)
        print(newx)
        newx = bf.fit_transform(x4)
        print(newx)
        newx = bf.fit_transform(x5)
        print(newx)

    def test_fea(self):
        bf = DummyConverter(n_jobs=1)
        bf.set_feature_labels(["name"])
        print(bf.feature_labels)


if __name__ == '__main__':
    unittest.main()
