import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from featurebox.selection.ga import GA


class MyTestCase(unittest.TestCase):
    def test_something(self):
        param_grid1 = {'C': [50, 1, 0.1]}
        X, y = load_iris(return_X_y=True)
        svc = SVC()
        gd = GridSearchCV(svc, cv=5, param_grid=param_grid1)
        bf = GA(estimator=gd, random_state=3, pop_n=200, ngen=2)
        bf.fit(X[:, :5], y)


if __name__ == '__main__':
    unittest.main()
