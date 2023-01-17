#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Calculate the correction of columns.
"""
import copy
from typing import List

import numpy as np
from mgetool.tool import name_to_name
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from featurebox.selection.multibase import MultiBase


class Corr(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MultiBase):
    """
    Calculate correlation. (Where the result are changed with random state.)

    Examples
    ---------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from featurebox.selection.corr import Corr
    >>> x, y = fetch_california_housing(return_X_y=True)
    >>> x = x[:100]
    >>> y = y[:100]
    >>> co = Corr(threshold=0.5)
    >>> nx = co.fit_transform(x)

    **1. Used for get group exceeding the threshold**

    Examples
    ---------

    >>> from sklearn.datasets import fetch_california_housing
    >>> from featurebox.selection.corr import Corr
    >>> x, y = fetch_california_housing(return_X_y=True)
    >>> x = x[:100]
    >>> y = y[:100]
    >>> co = Corr(threshold=0.5)
    >>> groups = co.count_cof(np.corrcoef(x[:,:7], rowvar=False))
    >>> groups[1]
    [[0, 6], [1], [2], [3], [4], [5], [0, 6]]
    >>> groups[0]
    [[1.0, 0.554], [1.0], [1.0], [1.0], [1.0], [1.0], [0.554, 1.0]]

    Where the (0,6) are with correlation more than 0.7.

    **2. Used for filter automatically by machine**

    Examples
    -----------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from featurebox.selection.corr import Corr
    >>> x, y = fetch_california_housing(return_X_y=True)
    >>> x = x[:100]
    >>> y = y[:100]
    >>> co = Corr(threshold=0.5)
    >>> co.fit(x)
    Corr(threshold=0.5)
    >>> group = co.count_cof()
    >>> group[1]
    [[0, 6, 7], [1], [2], [3], [4], [5], [0, 6, 7], [0, 6, 7]]
    >>> co.remove_coef(group[1]) # Filter automatically by machine.
    [0, 1, 2, 3, 4, 5]

    Where the remove_coef are changed with random state.

    **3. Used for binding correlation**

    Examples
    -----------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from featurebox.selection.corr import Corr
    >>> x, y = fetch_california_housing(return_X_y=True)
    >>> x = x[:100]
    >>> y = y[:100]
    >>> co = Corr(threshold=0.3,multi_index=[0,8],multi_grade=2)
    >>> # in range [0,8], the features are binding in to 2 sized: [[0,1],[2,3],[4,5],[6,7]]
    >>> co.fit(x)
    Corr(multi_index=[0, 8], threshold=0.3)
    >>> group = co.count_cof()
    >>> group[1]
    [[0, 1, 3], [0, 1, 3], [2], [0, 1, 3]]
    >>> co.remove_coef(group[1]) # Filter automatically by machine.
    [0, 2]
    """

    def __init__(self, threshold: float = 0.85, multi_grade: int = 2, multi_index: List = None, must_index: List = None,
                 random_state: int = 0):
        """

        Parameters
        ----------
        threshold:float
            ranking threshold.
        multi_grade:
            binding_group size, calculate the correction between binding.
        multi_index:list
            the range of multi_grade:[min,max).
        must_index:list
            the columns force to index.
        random_state: int
        """

        super().__init__(multi_grade=multi_grade, multi_index=multi_index, must_index=must_index)
        self.threshold = threshold
        self.cov = None
        self.cov_shrink = None
        self.shrink_list = []
        self.random_state = random_state
        self.nan_index_mark = None

    def fit(self, data, pre_cal=None, method="mean"):
        if pre_cal is None:

            cov = np.corrcoef(data, rowvar=False, )

        elif isinstance(pre_cal, np.ndarray) and pre_cal.shape[0] == data.shape[1]:
            cov = pre_cal
        else:
            raise TypeError("pre_cal is None or coef of data_cluster with shape(data_cluster[0],data_cluster[0])")

        # for nan
        self.nan_index_mark = ~np.array([np.all(np.isnan(cov[i])) for i in range(cov.shape[0])])
        if not np.all(self.nan_index_mark):
            print("There are some NAN values in correlation coefficient matrix, which could be constant features.\n"
                  "The NAN features would removed later. See more in 'nan_index_mark' attribute.")

        cov = np.nan_to_num(cov)
        self.cov = cov
        self.data = data
        self.shrink_list = list(range(self.cov.shape[0]))
        self._shrink_coef(method=method)
        self.filter()
        return self

    def _shrink_coef(self, method="mean" or "max"):

        if self.check_multi:
            self.shrink_list = list(range(self.cov.shape[0]))
            self.shrink_list = list(self.feature_fold(self.shrink_list))

            cov = self.cov
            single = tuple([i for i in self.shrink_list if i not in self.multi_index])
            multi = tuple([i for i in self.shrink_list if i in self.multi_index])

            cov_multi_all = []
            le = self.multi_grade
            while le:
                index = []
                index.extend(single)
                index.extend([i + le - 1 for i in multi])
                index.sort()
                cov_multi_all.append(cov[index][:, index])
                le -= 1
            cov_multi_all = np.array(cov_multi_all)
            if method == "mean":
                cov_new = np.mean(cov_multi_all, axis=0)
            else:
                cov_new = np.max(cov_multi_all, axis=0)
            self.cov_shrink = cov_new
            return self.cov_shrink
        else:
            self.cov_shrink = self.cov

    def count_cof(self, cof=None):
        """Check cof and count the number."""
        if cof is None:
            cof = self.cov_shrink
        if cof is None:
            raise NotImplemented("imported cof is None")

        list_count = []
        list_coef = []
        g = np.where(abs(cof) >= self.threshold)
        for i in range(cof.shape[0]):
            e = np.where(g[0] == i)
            com = list(g[1][e])
            # ele_ratio.remove(i)
            list_count.append(com)
            list_coef.append([round(cof[i, j], 3) for j in com])
        self.list_coef = list_coef
        self.list_count = list_count
        return list_coef, list_count

    def remove_coef(self, cof_list_all):
        """Delete the index of feature with repeat coef."""
        ran = check_random_state(self.random_state)
        reserve = []
        for i in cof_list_all:
            if not cof_list_all:
                reserve.append(i)

        for cof_list in cof_list_all:
            if not cof_list:
                pass
            else:
                if reserve:
                    candi = []
                    for j in cof_list:

                        con = any([[False, True][j in cof_list_all[k]] for k in reserve])
                        if not con:
                            candi.append(j)
                    if any(candi):
                        a = ran.choice(candi)
                        reserve.append(a)
                    else:
                        pass
                else:
                    a = ran.choice(cof_list)
                    reserve.append(a)
                cof_list_t = copy.deepcopy(cof_list)
                for dela in cof_list_t:
                    for cof_list2 in cof_list_all:
                        if dela in cof_list2:
                            cof_list2.remove(dela)
        return sorted(list(set(reserve)))

    def filter(self):
        list_coef, list_count = self.count_cof()
        index = self.remove_coef(list_count)
        support_ = [self.shrink_list[i] for i in index]
        support_ = self.feature_unfold(support_)
        support_ = np.array([True if i in support_ else False for i in range(self.data.shape[1])])
        if self.nan_index_mark is None:
            self.support_ = support_
        else:
            self.support_ = support_ * self.nan_index_mark
        return support_

    @staticmethod
    def cov_y(x_, y_):
        cov = np.corrcoef(x_, y_, rowvar=False)
        cov = cov[:, -1][:-1]
        return cov

    def remove_by_y(self, y_):
        corr = self.cov_y(self.data, y_)
        corr = self.feature_fold(corr)
        lcount = self.list_count
        fea_all = []
        score = name_to_name(corr, search=lcount, search_which=0, return_which=(1,), two_layer=True)
        for score_i, list_i in zip(score, lcount):
            indexs = np.argmax(score_i)
            feature_index = list_i[int(indexs)]
            fea_all.append(feature_index)

        fea_all = sorted(list(set(fea_all)))
        return fea_all

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def inverse_transform_index(self, index):
        """inverse the selected index to origin index by support."""
        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            index = np.where(index)[0]
        index = np.array(list(index))

        return np.where(self.support_)[0][index]

    def transform_index(self, index):
        """Get support index."""
        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            index = np.where(index)[0]

        return np.array([i for i in index if self.support_[i]])

    def transform(self, data):
        if isinstance(data, (list, tuple)):
            return data[self.support_]
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            return data[self.support_]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            return data[:, self.support_]
        else:
            pass

#
# if __name__ == "__main__":
#     # x, y = fetch_california_housing(return_X_y=True)
#     # co = Corr(threshold=0.7)
#     # c = co.count_cof(np.corrcoef(x, rowvar=False))[1]
#     from sklearn.datasets import fetch_california_housing
#
#     x, y = fetch_california_housing(return_X_y=True)
#     x = x[:100]
#     y = y[:100]
#     co = Corr(threshold=0.5, multi_index=[0, 8], multi_grade=2)
#
#     nx = co.fit_transform(x)
