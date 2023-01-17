#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
This is script, copy for using, rather than call.**
"""

import warnings
from functools import partial

import numpy as np
from sklearn import kernel_ridge, gaussian_process, neighbors
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    RandomForestRegressor, GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, Lasso, ElasticNet, Perceptron, \
    SGDClassifier, PassiveAggressiveRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

kernel = 1.0 * RBF(1.0)
kernel2 = Matern(nu=0.5)
kernel3 = Matern(nu=1.5)
kernel4 = Matern(nu=2.5)
kernel5 = Matern(nu=0.5, length_scale=0.8)
kernel6 = Matern(nu=1.5, length_scale=0.8)
kernel7 = Matern(nu=2.5, length_scale=0.8)
kernel8 = 2 * Matern(nu=0.5)
kernel9 = 2 * Matern(nu=1.5)
kernel10 = 2 * Matern(nu=2.5)

ker = [kernel, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9, kernel10]


def dict_method_clf():
    """many clf method."""
    dict_method = {}
    # 1st part
    """4KNC"""
    me4 = neighbors.KNeighborsClassifier(n_neighbors=5)
    cv4 = StratifiedKFold(5, shuffle=False, random_state=0)
    scoring4 = 'balanced_accuracy'

    param_grid4 = [{'n_neighbors': [3, 4, 5, 6, 7], "weight": ['uniform', "distance"], "leaf_size=30": [10, 20, 30],
                    'metric': ['seuclidean', "manhattan"]}, ]

    dict_method.update({"KNC-set": [me4, cv4, scoring4, param_grid4]})

    """1SVC"""
    me1 = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
              coef0=0.0, shrinking=True, probability=False,
              tol=1e-3, cache_size=200, class_weight='balanced',
              verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    cv1 = StratifiedKFold(5, shuffle=False)
    scoring1 = 'accuracy'

    param_grid1 = [{'C': [1.0e8, 1.0e6, 10000, 100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01], 'kernel': ker}]

    dict_method.update({'SVC-set': [me1, cv1, scoring1, param_grid1]})

    """5GPC"""
    me5 = gaussian_process.GaussianProcessClassifier(kernel=kernel)
    cv5 = StratifiedKFold(5, shuffle=False)
    scoring5 = 'balanced_accuracy'
    param_grid5 = [{"kernel": ker}, ]

    dict_method.update({'GPC-set': [me5, cv5, scoring5, param_grid5]})

    # 2nd part
    '''TreeC'''
    me6 = DecisionTreeClassifier(
        criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, class_weight="balanced")
    cv6 = StratifiedKFold(5, shuffle=False)
    scoring6 = 'accuracy'
    param_grid6 = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4]}]
    dict_method.update({'TreeC-em': [me6, cv6, scoring6, param_grid6]})

    '''GBC'''
    me7 = GradientBoostingClassifier(
        loss='deviance', learning_rate=0.1, n_estimators=100,
        subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.,
        max_depth=3, min_impurity_decrease=0.,
        init=None,
        random_state=None, max_features=None, verbose=0,
        max_leaf_nodes=None, warm_start=False,
    )
    cv7 = StratifiedKFold(5, shuffle=False)
    scoring7 = 'balanced_accuracy'
    param_grid7 = [{'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split': [2, 3, 4], 'learning_rate': [0.1, 0.05]}]
    dict_method.update({'GBC-em': [me7, cv7, scoring7, param_grid7]})

    '''RFC'''
    me8 = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                 max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                                 bootstrap=True, oob_score=False,
                                 random_state=None, verbose=0, warm_start=False,
                                 class_weight="balanced")
    cv8 = StratifiedKFold(5, shuffle=False)
    scoring8 = 'accuracy'
    param_grid8 = [{'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split': [2, 3, 4], 'learning_rate': [0.1, 0.05]}]
    dict_method.update({"RFC-em": [me8, cv8, scoring8, param_grid8]})

    "AdaBC"
    dt = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=5, min_samples_split=4)
    me9 = AdaBoostClassifier(dt, n_estimators=100, learning_rate=1., algorithm='SAMME.R',
                             random_state=0)
    cv9 = StratifiedKFold(5, shuffle=False)
    scoring9 = 'accuracy'
    param_grid9 = [{'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split': [2, 3, 4], 'learning_rate': [0.1, 0.05]}]
    dict_method.update({"AdaBC-em": [me9, cv9, scoring9, param_grid9]})

    # the 3rd

    "Per"
    me14 = Perceptron(penalty="l1", alpha=0.0001, fit_intercept=True, max_iter=None, tol=None,
                      shuffle=True, verbose=0, eta0=1.0, random_state=0,
                      class_weight=None, warm_start=False)
    cv14 = StratifiedKFold(5, shuffle=False)
    scoring14 = 'accuracy'
    param_grid14 = [{'alpha': [0.0001, 0.001, 0.01]}, ]
    dict_method.update({"Per-L1": [me14, cv14, scoring14, param_grid14]})

    """LogRL1"""
    me15 = LogisticRegression(penalty='l1', solver='liblinear', dual=False, tol=1e-3, C=1.0, fit_intercept=True,
                              intercept_scaling=1, class_weight='balanced', random_state=0)
    cv15 = StratifiedKFold(5, shuffle=False)
    scoring15 = 'accuracy'
    param_grid15 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2], 'penalty': ["l1", "l2"]}, ]
    dict_method.update({"LogR-L1": [me15, cv15, scoring15, param_grid15]})

    """3SGDCL2"""
    me3 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                        fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                        verbose=0, epsilon=0.1, random_state=0,
                        learning_rate='optimal', eta0=0.0, power_t=0.5,
                        class_weight="balanced", warm_start=False, average=False)
    cv3 = StratifiedKFold(5, shuffle=False)
    scoring3 = 'accuracy'

    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05], 'loss': ['squared_loss', "huber"],
                    "penalty": ["l1", "l2"]}, ]

    dict_method.update({"SGDC-set": [me3, cv3, scoring3, param_grid3]})

    return dict_method


def dict_method_reg():
    """many reg method."""
    dict_method = {}
    # 1st part

    """4KNR"""
    me4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski')
    cv4 = 5
    scoring4 = 'r2'
    param_grid4 = [{'n_neighbors': [3, 4, 5, 6, 7], "weights": ['uniform', "distance"], "leaf_size": [10, 20, 30]
                    }]
    dict_method.update({"KNR-set": [me4, cv4, scoring4, param_grid4]})

    """1SVR"""
    me1 = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=False, max_iter=2000)
    cv1 = 5
    scoring1 = 'r2'
    param_grid1 = [{'C': [10000, 100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01], 'kernel': ker}]
    dict_method.update({"SVR-set": [me1, cv1, scoring1, param_grid1]})

    """5kernelridge"""
    me5 = kernel_ridge.KernelRidge(alpha=1, gamma="scale", degree=3, coef0=1, kernel_params=None)
    cv5 = 5
    scoring5 = 'r2'
    param_grid5 = [{'alpha': [100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01, 0.001, 1e-4, 1e-5], 'kernel': ker}]
    dict_method.update({'KRR-set': [me5, cv5, scoring5, param_grid5]})

    """6GPR"""
    me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                                    n_restarts_optimizer=0,
                                                    normalize_y=False, copy_X_train=True, random_state=0)
    cv6 = 5
    scoring6 = 'r2'
    param_grid6 = [{'alpha': [1e-3, 1e-2], 'kernel': ker}]
    dict_method.update({"GPR-set": [me6, cv6, scoring6, param_grid6]})

    # 2nd part

    """6RFR"""
    me7 = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                bootstrap=True, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    cv7 = 5
    scoring7 = 'r2'
    param_grid7 = [{'max_depth': [4, 5, 6, 7], }]
    dict_method.update({"RFR-em": [me7, cv7, scoring7, param_grid7]})

    """7GBR"""
    me8 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                    subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                    max_depth=3, min_impurity_decrease=0.,
                                    init=None, random_state=None,
                                    max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                    warm_start=False, )
    cv8 = 5
    scoring8 = 'r2'
    param_grid8 = [{'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 3],
                    'learning_rate': [0.1, 0.05]}]
    dict_method.update({'GBR-em': [me8, cv8, scoring8, param_grid8]})

    "AdaBR"
    dt3 = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=7, min_samples_split=4)
    me9 = AdaBoostRegressor(dt3, n_estimators=200, learning_rate=0.05, random_state=0)
    cv9 = 5
    scoring9 = 'explained_variance'
    param_grid9 = [{"base_estimator": [dt3]}]
    dict_method.update({"AdaBR-em": [me9, cv9, scoring9, param_grid9]})

    '''DTR'''
    me10 = DecisionTreeRegressor(
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,

    )
    cv10 = 5
    scoring10 = 'r2'
    param_grid10 = [
        {'max_depth': [2, 3, 4, 5, 6, 7, 8], "min_samples_split": [2, 3, 4], "min_samples_leaf": [1, 2]}]
    dict_method.update({'DTR-em': [me10, cv10, scoring10, param_grid10]})

    'ElasticNet'
    me11 = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
                      copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)

    cv11 = 5
    scoring11 = 'r2'
    param_grid11 = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.3, 0.5, 0.8]}]
    dict_method.update({"EN-L1": [me11, cv11, scoring11, param_grid11]})

    'Lasso'
    me12 = Lasso(alpha=1.0, fit_intercept=True, normalize=True, precompute=False, copy_X=True, max_iter=3000,
                 tol=0.001,
                 warm_start=False, positive=False, random_state=None, )

    cv12 = 5
    scoring12 = 'r2'
    param_grid12 = [
        {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000], "tol": [0.001, 0.01, 0.1]}, ]
    dict_method.update({"LASSO-L1": [me12, cv12, scoring12, param_grid12]})

    """2BayesianRidge"""
    me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                        n_iter=300, normalize=False, tol=0.01, verbose=False)
    cv2 = 5
    scoring2 = 'r2'
    param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05], 'alpha_2': [1e-07, 1e-06, 1e-05]}]
    dict_method.update({'BRR-L1': [me2, cv2, scoring2, param_grid2]})

    """3SGDRL2"""
    me3 = SGDRegressor(alpha=0.0001, average=False,
                       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                       penalty='l2', power_t=0.25,
                       random_state=0, shuffle=True, tol=0.01,
                       verbose=0, warm_start=False)
    cv3 = 5
    scoring3 = 'r2'
    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05], 'loss': ['squared_loss', "huber"],
                    "penalty": ["l1", "l2"]}]
    dict_method.update({'SGDR-L1': [me3, cv3, scoring3, param_grid3]})

    """PassiveAggressiveRegressor"""
    me14 = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False,
                                      validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
                                      loss='epsilon_insensitive', epsilon=0.1, random_state=None,
                                      warm_start=False, average=False)
    cv14 = 5
    scoring14 = 'r2'
    param_grid14 = [{'C': [1.0e8, 1.0e6, 10000, 100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01]}]
    dict_method.update({'PAR-L1': [me14, cv14, scoring14, param_grid14]})

    return dict_method


def dict_me(me="clf"):
    if me == "clf":
        dict_method_ = dict_method_clf()
    else:
        dict_method_ = dict_method_reg()
    return dict_method_


def method_pack(method_all, me="reg", scoring=None, gd=True, cv=10):
    """return cv or gd."""
    if not method_all:
        method_all = ['KNR-set', 'SVR-set', "KRR-set", "GPR-set",
                      "RFR-em", "AdaBR-em", "DTR-em",
                      "LASSO-L1", "BRR-L1", "SGDR-L1", "PAR-L1"]
    dict_method = dict_me(me=me)

    print(dict_method.keys())
    if gd:
        estimator = []
        for method_i in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method_i]
            if me == "clf":
                scoring2 = scoring if scoring else 'balanced_accuracy'
            if me == "reg":
                scoring2 = scoring if scoring else 'r2'
            cv2 = cv if cv else cv2
            gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=10)
            estimator.append(gd2)
        return estimator
    else:
        estimator = []
        for method_i in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method_i]
            if me == "clf":
                scoring2 = scoring if scoring else 'balanced_accuracy'
            if me == "reg":
                scoring2 = scoring if scoring else 'r2'
            cv2 = cv if cv else cv2
            gd2 = partial(cross_val_score, estimator=me2, cv=cv2, scoring=scoring2)
            # gd2 = cross_val_score(me2, cv=cv2, scoring=scoring2)
            estimator.append(gd2)
        return estimator


def cv_predict(x, y, s_estimator, kf):
    y_test_predict_all = []
    for train, test in kf:
        X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
        s_estimator.fit(X_train, y_train)
        y_test_predict = s_estimator.predict(X_test)
        y_test_predict_all.append(y_test_predict)

    test_index = [i[1] for i in kf]
    y_test_true_all = [y[_] for _ in test_index]

    return y_test_true_all, y_test_predict_all


def pack_score(y_test_true_all, y_test_predict_all, scoring):
    if isinstance(y_test_true_all, np.ndarray) and isinstance(y_test_predict_all, np.ndarray):
        y_test_true_all = [y_test_true_all, ]
        y_test_predict_all = [y_test_predict_all, ]
    if scoring == "rmse":
        scoring2 = 'neg_mean_squared_error'
    else:
        scoring2 = scoring

    scorer = get_scorer(scoring2)

    scorer_func = scorer._score_func

    score = [scorer_func(i, j) for i, j in zip(y_test_true_all, y_test_predict_all)]

    if scoring == "rmse":
        score = np.sqrt(score)
    score_mean = np.mean(score)
    return score_mean

# def my_score(gd_method, train_X, test_X, train_Y, test_Y):
#     # train_X, test_X = my_pca(train_X, test_X, 0.9)
#
#     # s_X = preprocessing.StandardScaler()
#     # train_X = s_X.fit_transform(train_X)
#     # test_X = s_X.transform(test_X)
#
#     # train_X,train_Y,= utils.shuffle(train_X,train_Y, random_state=3)
#
#     grid = gd_method
#     n_splits = 5
#     kf = KFold(n_splits=n_splits, shuffle=False)
#     grid.cv = kf
#     grid.fit(train_X, train_Y)
#
#     print("最好的参数：", grid.best_params_)
#     print("最好的模型：", grid.best_estimator_)
#     print("\ngrid.best_score:\n", grid.best_score_)
#
#     metrics_method1 = "rmse"
#     metrics_method2 = "r2"
#
#     # cv_train
#     kf = KFold(n_splits=n_splits, shuffle=False)
#     kf = list(kf.split(train_X))
#     y_train_true_all, y_train_predict_all = cv_predict(train_X, train_Y, grid.best_estimator_, kf)
#
#     cv_pre_train_y = np.concatenate([i.ravel() for i in y_train_predict_all]).T
#     score1 = pack_score(y_train_true_all, y_train_predict_all, metrics_method1)
#     score2 = pack_score(y_train_true_all, y_train_predict_all, metrics_method2)
#     print("train_X's cv score %s" % score1, "train_X's cv score %s" % score2)
#
#     # all_train
#     grid = grid.best_estimator_
#     # grid.fit(train_X,train_Y)
#     pre_train_y = grid.predict(train_X)
#     score3 = pack_score(train_Y, pre_train_y, metrics_method1)
#     score4 = pack_score(train_Y, pre_train_y, metrics_method2)
#     print("train_X's score %s " % score3, "train_X's score %s" % score4)
#     # test
#     pre_test_y = grid.predict(test_X)
#     score5 = pack_score(test_Y, pre_test_y, metrics_method1)
#     score6 = pack_score(test_Y, pre_test_y, metrics_method2)
#     print("test_X's score %s" % score5, "test_X's score %s" % score6)
#
#     return cv_pre_train_y, pre_train_y, pre_test_y

# if __name__ == "__main__":
#     import pandas as pd
#
#     com_data_raw = pd.read_csv('wxxwcx.csv')
#
#     com_data = com_data_raw
#
#     select_X5 = ['O-M-outer', 'a Lattice parameter', 'M Electron affinity mean',
#     'M first ionization potential differ',
#                  'valence X atom']
#     target_y = 'Hydrogen adsorption energy'
#     x = com_data[select_X5].values
#     y = com_data[target_y].values
#     method = ["SVR-set", 'KRR-set', "GPR-set", "RFR-em", 'GBR-em', "AdaBR-em", 'TreeC-em']
#     gd_method = method_pack(method, me="reg", gd=True)
#     train_X, test_X, train_Y, test_Y = x[42:, :], x[:42, :], y[42:], y[:42]
#     cv_pre_train_y, pre_train_y, pre_test_y = my_score(gd_method[5], train_X, test_X, train_Y, test_Y)
