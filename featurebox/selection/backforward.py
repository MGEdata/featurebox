# -*- coding: utf-8 -*-

# @Time   : 2019/5/23 20:40
# @Author : Administrator
# @Project : feature_preparation
# @FileName: backforward.py
# @Software: PyCharm

"""Forward_and_back feature elimination for feature ranking"""

import copy
import warnings
from functools import partial
from typing import List

import numpy as np
from mgetool.tool import parallelize
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._validation import _score, cross_val_score
from sklearn.utils.metaestimators import if_delegate_has_method, _safe_split
from sklearn.utils.validation import check_is_fitted, check_X_y, check_random_state

from featurebox.selection.multibase import MultiBase


class BackForward(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MultiBase):
    """
    BackForward method to selected features.
    

    Attributes
    ----------
    n_feature_: int
        The number of selected features finally.

    support_: array of shape [n_feature]
        The mask of selected features finally.

    estimator_: object
        The best model with the best features finally (refited with all data.).

    best_score_: float
        Best score of best model of best features.

    Examples
    --------

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> bf = BackForward(svr,primary_feature=4, random_state=1)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False,  True,  True, False, False, False, False,  True])

    Examples
    --------
    If ``score`` and ``predict`` is used, the ``refit`` should be set True,
    the refit used all data in ``fit`` function, that is, it is not test score/predict.

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> from sklearn.model_selection import cross_val_score
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> bf = BackForward(svr,primary_feature=4, random_state=1, refit=True,cv=5)
    >>> bf = bf.fit(X[:50],y[:50])
    >>> bf.best_score_         # cv score
    -3.0552830696940037
    >>> train_score = bf.score(X[:50],y[:50])  # train score
    >>> test_score = bf.score(X[-50:],y[-50:]) # test score in more data.
    >>> np.mean(cross_val_score(bf.estimator_,X[:50,bf.support_],y[:50],cv=5)) # re cv_score in manually.
    -3.0552830696940037

    Examples
    --------
    If GridSearchCV, the refit should be set True and return the cv score.

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> from sklearn import model_selection
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> gd = model_selection.GridSearchCV(svr,param_grid={"C":[1,10]},n_jobs=1)  # keep n_jobs=1 there.
    >>> bf = BackForward(gd,primary_feature=4, random_state=1, refit=True,scoring="neg_root_mean_squared_error",cv=5)
    >>> bf = bf.fit(X[:50],y[:50])
    >>> bf.best_score_         # cv score
    -0.5919173121895709
    >>> train_score = bf.score(X[:50],y[:50])  # train score
    >>> test_score = bf.score(X[-50:],y[-50:]) # test score in more data.
    >>> bf.estimator_.best_score_ # re cv_score in manually.
    -0.5919173121895709
    """

    def __init__(self, estimator: BaseEstimator, n_type_feature_to_select: int = None, primary_feature: int = None,
                 multi_grade: int = 2, multi_index: List = None, refit=False, cv=5, min_type_feature_to_select: int = 3,
                 must_index: List = None, tolerant: float = 0.01, verbose: int = 1, random_state: int = None,
                 scoring: str = None):
        """

        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
            A supervised sklearn learning estimator with ``fit`` method.
        n_type_feature_to_select:int
            The max number of feature to selection. If ``None``, select the features with best score.
        min_type_feature_to_select:int
            force select number min.
        primary_feature:int
            primary features to start loop, default initial n_features//2.
        multi_grade: int
            group number.
        multi_index:
            group index.
        must_index:
            must selection index.
        tolerant:
            tolerant for rank compare.
        verbose:int
            print or not.
        random_state:int
            random_state.
        refit:bool
            refit or not. if refit, the model would use all data.
        scoring:None,str
            scoring method name.
        """
        super().__init__(multi_grade=multi_grade, multi_index=multi_index, must_index=must_index)
        if any((hasattr(estimator, "max_features") and refit,
                isinstance(estimator, BaseSearchCV) and hasattr(estimator.estimator, "max_features") and refit)):
            print("For estimator with 'max_features' attribute, the 'max_features' would changed with each sub-data. \n"
                  "Please change and define 'max_features' by manual testing after Backforward!\n")
        if refit:
            if isinstance(estimator, BaseSearchCV) and estimator.refit is True:
                warnings.warn(
                    "\nThe self.estimator_ :{} would used all the X, y data if refit! \n"
                    "Please be careful with the 'score' and 'predict' if use, "
                    "which are 'train' score/predict if inputs not changed!!!\n"
                    "Check 'self.estomator_.cv_result' to get CV result,"
                    " such as 'self.estomator_.best_score_' for evaluation instead.".format(
                        estimator.__class__.__name__), UserWarning)
            else:
                warnings.warn(
                    "\nThe self.estimator_ :{} would used all the X, y data with refit! \n"
                    "Please be careful with the 'score' and 'predict' functions."
                    "if inputs not changed, the 'score' and 'predict' are training!!!\n"
                    "Thus:\n"
                    "Use 'cross_val_score(self.estimator_,X[:, self.support_],y)' for evaluation instead,\n"
                    "Use 'cross_val_predict(self.estimator_,X[:, self.support_],y)' for plot instead."
                    "".format(
                        estimator.__class__.__name__), UserWarning)

        assert cv >= 3

        if isinstance(estimator, BaseSearchCV):
            print(f"Using scoring:{scoring},and cv:{cv}")
            estimator.scoring = scoring
            estimator.cv = cv
        self.scoring = scoring
        self.estimator = estimator

        self.n_type_feature_to_select = n_type_feature_to_select
        self.primary_feature = primary_feature
        self.verbose = verbose
        self.score_ = []
        self.random_state = random_state
        self.tolerant = tolerant
        self.refit = refit
        self.min_type_feature_to_select = min_type_feature_to_select
        self.cv = cv
        if isinstance(n_type_feature_to_select, int):
            assert n_type_feature_to_select >= min_type_feature_to_select, "Max numbers should be large than Min numbers."

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the baf model and then the underlying estimator on the selected
           feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            The training input0 samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, x, y):

        estimator = clone(self.estimator)

        def add_slice(slice10, slice20):
            best0 = score(slices=slice10)

            for add1 in list(slice20):
                # or random.choice() >= score(slice1):
                slice10.append(add1)
                test = score(slices=slice10)
                slice1_, test_ = sub_slice(slice10)
                if test_ >= max(best0, test):
                    best0 = test_
                    slice10 = slice1_
                elif test > max(best0, test_) + 2 * self.tolerant:
                    best0 = test
                else:
                    slice10.remove(add1)
                if self.verbose > 0:
                    print("Fitting estimator with {} feature {}".format(len(slice10), best0))
                self.score_.append((tuple(slice10), best0))
            return slice10, best0

        def sub_slice(slice10):
            best0 = score(slices=slice10)
            slice0 = list(copy.deepcopy(slice10))
            slice10 = list(slice10)
            ran.shuffle(slice10)
            if self.check_must:
                slice10 = [_ for _ in slice10 if _ not in self.must_fold_add]

            for sub in list(slice10):
                if len(slice0) <= self.min_type_feature_to_select:
                    pass
                else:
                    slice0.remove(sub)
                    test0 = score(slices=slice0)
                    if test0 > best0 - self.tolerant:
                        best0 = test0
                    else:
                        slice0.append(sub)
                    # print(slice0, best)
            return slice0, best0

        def sub_slice_force(slice10):
            best0 = score(slices=slice10)
            slice0 = list(copy.deepcopy(slice10))
            ran.shuffle(slice0)

            while len(slice0) > self.n_type_feature_to_select:
                slice_all = []
                slice10 = list(copy.deepcopy(slice0))
                if self.check_must:
                    slice10 = [_ for _ in slice10 if _ not in self.must_fold_add]
                le = len(slice10)
                while le:
                    slice_all.append(copy.deepcopy(slice0))
                    le -= 1
                for slice_, sub in zip(slice_all, slice10):
                    slice_.remove(sub)
                test0 = [score(slices=i) for i in slice_all]
                index = int(np.argmax(test0))
                best0 = max(test0)
                slice0 = slice_all[index]
                # print(slice0, best0)
                if self.verbose > 0:
                    print("Fitting estimator with {} feature {}".format(len(slice0), best0))
                self.score_.append((tuple(slice0), best0))
            return slice0, best0

        def score_pri(slices, x0, y0):
            slices = list(slices)
            if len(slices) <= 1:
                score0 = - np.inf
            else:
                slices = self.feature_unfold(slices)
                data_x0 = x0[:, slices]

                if hasattr(self.estimator, "max_features"):
                    estimator_ = clone(self.estimator)
                    estimator_.max_features = data_x0.shape[1]  # force to data feature size.

                elif isinstance(self.estimator, BaseSearchCV) and hasattr(self.estimator.estimator, "max_features"):
                    estimator_ = clone(self.estimator)
                    estimator_.estimator.max_features = data_x0.shape[1]  # force to data feature size.
                else:
                    estimator_ = estimator

                estimator_.fit(data_x0, y0)

                if hasattr(estimator_, 'best_score_') or isinstance(estimator_, BaseSearchCV):
                    score0 = np.mean(estimator_.best_score_)
                else:
                    score0 = np.mean(cross_val_score(estimator_, data_x0, y0, cv=self.cv, scoring=self.scoring))

            return score0

        score = partial(score_pri, x0=x, y0=y)

        self.score_ = []
        x, y = check_X_y(x, y, "csc")
        assert all((self.check_must, self.check_multi)) in [True, False]
        # Initialization
        if self.check_multi:
            n_feature = (self.multi_index[1] - self.multi_index[0]) // self.multi_grade + self.multi_index[0]
        else:
            n_feature = x.shape[1]
        if self.primary_feature is None:
            primary_feature = n_feature // 2
        else:
            primary_feature = self.primary_feature
        assert primary_feature < n_feature, "Too large for primary_feature."

        feature_list = list(range(n_feature))
        fold_feature_list = self.feature_fold(feature_list)

        ran = check_random_state(self.random_state)
        slice1 = ran.choice(fold_feature_list, primary_feature, replace=False)
        slice1 = list(self.feature_fold(slice1))
        slice2 = list(set(fold_feature_list) - set(slice1))
        ran.shuffle(slice2)

        slice1, best = add_slice(slice1, slice2)
        if isinstance(self.n_type_feature_to_select, int) and len(slice1) > self.n_type_feature_to_select:
            slice1, best = sub_slice_force(slice1)

        slice1.sort()
        select_feature = np.array(self.feature_unfold(slice1))
        sup = np.zeros(x.shape[1], dtype=np.bool)
        sup[select_feature] = 1
        self.best_score_ = best
        self.support_ = sup
        self.estimator_ = clone(self.estimator)

        if self.refit:
            if hasattr(self.estimator, "max_features"):
                self.estimator_.max_features = select_feature.shape[0]
            elif isinstance(self.estimator, BaseSearchCV) and hasattr(self.estimator.estimator, "max_features"):
                self.estimator_.estimator.max_features = select_feature.shape[0]
            self.estimator_.fit(x[:, select_feature], y)
        self.n_feature_ = len(select_feature)

        return self

    @if_delegate_has_method(delegate='estimator_')
    def predict(self, X):
        """Reduce X to the selected feature and then using the underlying estimator to predict.
         Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, y, scoring=None):
        """Reduce X to the selected feature and then return the score of the underlying estimator.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.

        scoring : str, callable, default=None
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:
            a single string (see :ref:`scoring_parameter`)

            The score defined by ``scoring`` if provided, and the
            ``estimator_.score`` method otherwise else raise error.
        """
        check_is_fitted(self, 'estimator_')
        scoring = scoring if scoring is not None else self.scoring
        scorer = check_scoring(self.estimator_, scoring=scoring, allow_none=False)
        return scorer(self.estimator_, self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_


def _baf_single_fit(train, test, baf, estimator, X, y, scorer, random_state):
    """"""
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    baf_i = clone(baf)
    baf_i.random_state = random_state
    baf_i.refit = True
    baf_i._fit(X_train, y_train)
    return baf_i.support_, _score(baf_i.estimator_, baf_i.transform(X_test, ), y_test, scorer), baf_i.score_


def _multi_time_fit(random_state, baf, X, y, scorer):
    """"""
    baf_i = clone(baf)
    baf_i.random_state = random_state
    baf_i.refit = True
    baf_i._fit(X, y)
    return baf_i.support_, _score(baf_i.estimator_, baf_i.transform(X, ), y, scorer), baf_i.score_


class BackForwardStable(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    BackForwardStable.
    Run with different order for more Stable (Just for test).

    Attributes
    ----------
    n_feature_ : int
        The number of selected feature with cross-validation.

    support_ : array of shape [n_feature]
        The mask of selected feature.

    estimator_ : object
        The model with the best features finally (refited with all data.).

    best_score_: float
        Best score of best model of best features.

    Examples
    --------

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> bf = BackForwardStable(svr,primary_feature=3,  random_state=1)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False,  True, False, False, False,  True,  True, False])
    >>> bf.best_score_
    -0.09122826477472024

    If score and predict is used, the refit could be set True and make sure the data is splited, due to the refit
    used all data in fit() function.

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> bf = BackForwardStable(svr,primary_feature=4, random_state=1, refit=True)
    >>> new_x = bf.fit_transform(X[:50],y[:50])
    >>> train_score = bf.score(X[50:],y[50:])
    >>> cv_score = bf.best_score_
    ...

    If GridSearchCV, the refit could be set True and return the cv score.

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> from sklearn import model_selection
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()
    >>> gd = model_selection.GridSearchCV(svr,param_grid={"C":[1,10]},n_jobs=1,cv=5)
    >>> bf = BackForward(gd,primary_feature=4, random_state=1, refit=True)
    >>> new_x = bf.fit_transform(X,y)
    ...
    """

    def __init__(self, estimator: BaseEstimator, n_type_feature_to_select: int = None,
                 min_type_feature_to_select: int = 3,
                 primary_feature: int = None, multi_grade: int = 2, multi_index: List = None,
                 must_index: List = None, verbose: int = 0, random_state: int = None,
                 tolerant: float = 0.001, cv: int = 5,
                 times: int = 5, scoring: str = None, n_jobs: int = None, refit=False):
        """

        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
            A supervised sklearn learning estimator with ``fit`` method.
        n_type_feature_to_select:int
            The max number of feature to selection. If ``None``, select the features with best score.
        min_type_feature_to_select:int
            force select number min.
        primary_feature:int
            primary features to start loop, default initial n_features//2.
        multi_grade: int
            group number.
        multi_index:
            group index.
        must_index:
            must selection index.
        tolerant:
            tolerant for rank compare.
        verbose:int
            print or not.
        random_state:int
            random_state.
        refit:bool
            refit or not. if refit, the model would use all data.
        n_jobs : int or None
            Number of cores to run in parallel while fitting across folds.
            ``None`` means 1 and ``-1`` means using all processors.
        scoring: None,str
            scoring method.

        """
        if isinstance(estimator, BaseSearchCV):
            warnings.warn("The 'estimator' of BackForwardStable not suggested BaseSearchCV, "
                          "because the BackForwardStable is one BaseSearchCV itself.")

        self.estimator = estimator
        self.times = times
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_type_feature_to_select = n_type_feature_to_select
        self.min_type_feature_to_select = min_type_feature_to_select
        self.primary_feature = primary_feature
        self.multi_grade = multi_grade
        self.multi_index = multi_index
        self.must_index = must_index
        self.score_ = []
        self.random_state = random_state
        self.refit = refit
        self.tolerant = tolerant
        self.cv = cv

    def fit(self, X, y, groups=None):
        """Fit the baf model and automatically tune the number of selected feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            Training vector, where `n_samples` is the number of samples and
            `n_feature` is the total number of feature.

        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like, shape = [n_samples], optional
            cal_group labels for the samples used while splitting the dataset into
            train/test set.
        """
        X, y = check_X_y(X, y, "csr")
        # Initialization
        estimator = clone(self.estimator)
        scorer = check_scoring(estimator, scoring=self.scoring)
        ran = check_random_state(self.random_state)

        baf = BackForward(estimator=estimator,
                          n_type_feature_to_select=self.n_type_feature_to_select,
                          min_type_feature_to_select=self.min_type_feature_to_select,
                          verbose=self.verbose, primary_feature=self.primary_feature,
                          multi_grade=self.multi_grade, multi_index=self.multi_index, cv=self.cv,
                          must_index=self.must_index, random_state=ran, tolerant=self.tolerant)
        rans = ran.randint(0, 1000, self.times)

        func = partial(_multi_time_fit, baf=baf, X=X, y=y, scorer=scorer)

        scores = parallelize(n_jobs=self.n_jobs, func=func, iterable=rans, respective=False)

        support, scores, score_step = zip(*scores)
        best_support = support[np.argmax(scores)]
        best_score = scores[np.argmax(scores)]
        # Re-execute an elimination with best_k over the whole set

        # Set final attributes
        self.support_step = score_step
        self.support_ = best_support
        self.best_score_ = best_score
        self.estimator_ = clone(self.estimator)
        if self.refit:
            if hasattr(self.estimator_, "max_features"):
                self.estimator_.max_features = np.sum(self.support_)
            self.estimator_.fit(X[:, self.support_], y)
        self.n_feature_ = np.count_nonzero(support)
        return self

    @if_delegate_has_method(delegate='estimator_')
    def predict(self, X):
        """Reduce X to the selected feature and then Fit using the underlying estimator, only with refit.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator, only with refit.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_
