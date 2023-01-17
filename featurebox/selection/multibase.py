# -*- coding: utf-8 -*-

# @Time   : 2019/5/26 0:47
# @Author : Administrator
# @Project : feature_preparation
# @FileName: multibase.py
# @Software: PyCharm
import itertools
from typing import List

import numpy as np


class MultiBase(object):
    """Base method for binding"""

    def __init__(self, multi_grade: int = 2, multi_index: List = None, must_index: List = None):
        """

        Parameters
        ----------
        multi_grade:
            binding_group size, calculate the correction between binding
        multi_index:list
            the range of multi_grade:[min,max)
        must_index:list,None
            the columns force to index
        """
        self.multi_grade = multi_grade
        self.multi_index = multi_index
        self.must_index = must_index

    @property
    def check_multi(self):
        multi_index = self.multi_index
        if multi_index is None:
            return False
        elif isinstance(multi_index, (list, tuple)):
            if len(multi_index) == 2 and isinstance(multi_index[0], int) and isinstance(multi_index[1], int):
                return True
            else:
                raise TypeError("multi_index should be None or iterable type with 2 number")
        else:
            raise TypeError("multi_index should be None or iterable type with 2 number")

    @property
    def check_must(self):
        must_index = self.must_index
        if must_index is None:
            return False
        elif isinstance(must_index, (list, tuple)):
            return True
        else:
            raise TypeError("must_index should be None or iterable type with less than 1 number")

    @property
    def must_fold_add(self):
        if self.check_must:
            must_index = self.must_index
            if self.check_multi:
                def ff(mi):
                    data = mi - self.multi_index[0]
                    data = (data // self.multi_grade) * self.multi_grade + self.multi_index[0]
                    return data

                com_mark = [self.multi_index[0] <= mi < self.multi_index[1] for mi in must_index]
                must_feature = [ff(mi) if com_mark is True else mi for com_mark_i, mi in zip(com_mark, must_index)]
                return must_feature
            else:
                return list(must_index)
        else:
            return []

    @property
    def must_unfold_add(self):
        if self.check_must:
            must_index = self.must_index
            if self.check_multi:
                def ff2(mi):
                    data = mi - self.multi_index[0]
                    data = (data // self.multi_grade) * self.multi_grade + self.multi_index[0]
                    return list(range(data, data + self.multi_grade))

                com_mark = [self.multi_index[0] <= mi < self.multi_index[1] for mi in must_index]
                must_feature = [ff2(mi) if com_mark is True else [mi, ] for com_mark_i, mi in zip(com_mark, must_index)]
                must_feature = list(itertools.chain(*must_feature))
                return must_feature
            else:
                return list(must_index)
        else:
            return []

    def _feature_fold(self, feature, raw=False):
        multi_grade, multi_index = self.multi_grade, self.multi_index
        if self.check_multi:
            feature = np.sort(feature)
            single = np.array([_ for _ in feature if _ < multi_index[0] or _ >= multi_index[1]])
            com_com = np.array([_ for _ in feature if multi_index[1] > _ >= multi_index[0]])
            com_sin = com_com[::multi_grade]
            res = np.hstack((single, com_sin))
        else:
            res = np.array(feature)
        if not raw:
            return np.sort(res).astype(int)
        else:
            return res.astype(int)

    def feature_fold(self, feature):
        fea2 = list(self._feature_fold(feature, raw=True))
        add = self.must_fold_add
        fea2.extend(add)
        return np.array(list(set(fea2)))

    def _feature_unfold(self, feature, raw=False):
        multi_grade, multi_index = self.multi_grade, self.multi_index
        if self.check_multi:
            single = np.array([_ for _ in feature if _ < multi_index[0] or _ >= multi_index[1]])
            com_sin = np.array([_ for _ in feature if multi_index[1] > _ >= multi_index[0]])
            com_com = list(com_sin)
            while multi_grade - 1:
                com_com.extend(com_sin + (multi_grade - 1))
                multi_grade -= 1
            res = np.array(list(set(np.hstack((single, np.array(com_com))))))
        else:
            res = np.array(feature)
        if not raw:
            return np.sort(res).astype(int)
        else:
            return res.astype(int)

    def feature_unfold(self, feature):
        fea2 = list(self._feature_unfold(feature, raw=True))
        add = self.must_unfold_add
        fea2.extend(add)
        return np.array(list(set(fea2)))

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
