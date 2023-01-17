import unittest

from featurebox.selection.backforward import BackForward, BackForwardStable


class MyTestCase(unittest.TestCase):
    def test_something(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.svm import SVR
        X, y = fetch_california_housing(return_X_y=True)
        svr = SVR()
        bf = BackForward(svr, primary_feature=4, random_state=1)
        new_x = bf.fit_transform(X[:100, :6], y[:100])
        bf.support_
        print(bf)

    def test_something2(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.svm import SVR
        X, y = fetch_california_housing(return_X_y=True)
        X = X[:100, :6]
        y=y[:100]
        svr = SVR()
        bf = BackForward(svr, primary_feature=4, random_state=1, refit=True)
        new_x = bf.fit_transform(X[:50], y[:50])
        test_score = bf.score(X[50:], y[50:])
        print(bf)

    def test_something3(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.svm import SVR
        X, y = fetch_california_housing(return_X_y=True)
        X = X[:100, :3]
        y=y[:100]
        svr = SVR()

        bf = BackForwardStable(svr, primary_feature=2, random_state=1, refit=True, n_jobs=4, times=4)
        new_x = bf.fit(X, y)
        test_score = bf.score(X, y)
        print(test_score)

    def test_something4(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.svm import SVR
        X, y = fetch_california_housing(return_X_y=True)
        X = X[:100, :4]
        y=y[:100]
        svr = SVR()
        bf = BackForward(svr, primary_feature=2, random_state=1, refit=True)
        new_x = bf.fit_transform(X, y)
        test_score = bf.score(X, y)

    def test_something5(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.svm import SVR
        from sklearn import model_selection
        X, y = fetch_california_housing(return_X_y=True)
        X = X[:100, :4]
        y = y[:100]
        svr = SVR()
        gd = model_selection.GridSearchCV(svr, param_grid=[{"C": [1, 10]}], n_jobs=1, cv=3)
        bf = BackForward(gd, primary_feature=2, random_state=1, refit=True)
        new_x = bf.fit(X, y)
        test_score = bf.score(X, y)
        print(test_score)


if __name__ == '__main__':
    unittest.main()
