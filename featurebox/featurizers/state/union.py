# -*- coding: utf-8 -*-

from abc import ABC
# @Time    : 2019/11/1 15:57
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause
from collections import Counter
from itertools import chain, combinations_with_replacement
from typing import List, Any, Union, Dict

import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from featurebox.featurizers.base_feature import BaseFeature
from featurebox.featurizers.state.extrastats import PropertyStats
from featurebox.featurizers.state.statistics import DepartElementFeature


class UnionFeature(BaseFeature):
    """
    Transform method should input0 comp_index rather than entries.

    Examples
    ---------
    >>> from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap
    >>> data_map = AtomJsonMap(search_tp="name", n_jobs=1)
    >>> wa = DepartElementFeature(data_map,n_composition=2, n_jobs=1,return_type="df")
    >>> x3 = [{"H": 2, "Pd": 1},{"He":1,"Al":4}]
    >>> wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)])
    >>> wa.fit_transform(x3)
        fea_0_0   fea_0_1   fea_1_0  ...  fea_14_1  fea_15_0  fea_15_1
    0  0.352363  0.561478  0.635952  ... -0.236541 -0.270104 -0.212607
    1 -0.067220  0.025758  0.141113  ... -0.092577 -0.042185  0.080350
    <BLANKLINE>
    [2 rows x 32 columns]

    >>> couple_data = wa.fit_transform(x3)
    >>> uf = UnionFeature(x3,couple_data,couple=2,stats=("mean","maximum"))
    >>> uf.fit_transform()
        feamean  feamaximum   feamean  ...  feamaximum   feamean  feamaximum
    0  0.422068    0.360958  0.201433  ...   -0.113506  0.021095   -0.212607
    1  0.007163   -0.471498 -0.072860  ...    0.312183  0.165278    0.080350
    <BLANKLINE>
    [2 rows x 32 columns]

    >>> couple_data = wa.fit_transform(x3)
    >>> uf = UnionFeature(x3,couple_data,couple=2,stats=("std_dev",))
    >>> uf.fit_transform()
       feastd_dev  feastd_dev  feastd_dev  ...  feastd_dev  feastd_dev  feastd_dev
    0    0.147867    0.583352    0.033739  ...    0.366625    0.182177    0.040657
    1    0.065745    0.541477    0.209795  ...    0.374331    0.182331    0.086646
    <BLANKLINE>
    [2 rows x 16 columns]

    """

    def __init__(self, comp: List[Dict], couple_data: Union[pd.DataFrame, np.ndarray], couple=2, stats=("mean",),
                 n_jobs: int = 1, on_errors: str = 'raise',
                 return_type: str = 'df'):
        super(UnionFeature, self).__init__(n_jobs, on_errors=on_errors, return_type=return_type)
        self.couple = couple
        self.comp = comp
        self.stats = stats
        self.elem_data = couple_data
        self.elem_data_np = couple_data.values if isinstance(couple_data, pd.DataFrame) else couple_data
        if isinstance(self.elem_data, pd.DataFrame):
            self.set_feature_labels(list(self.elem_data.columns))

        # Initialize stats computer

    def convert(self, comp_number=0):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
            :param comp_number:
        """
        comp = self.comp[comp_number]
        elem_data = self.elem_data_np[comp_number]

        # Get the element names and fractions
        elements, fractions = zip(*comp.items())
        elem_data = np.reshape(elem_data, (self.couple, -1), order="F")
        all_attributes_all = []
        for stat in self.stats:
            all_attributes = [PropertyStats.calc_stat(elem_data_i, stat, fractions) for elem_data_i in elem_data.T]
            all_attributes_all.extend(all_attributes)
        return np.array(all_attributes_all)

    def transform(self, entries: List = None) -> Any:
        ll = len(self.comp)
        return super(UnionFeature, self).transform(list(range(ll)))

    def set_feature_labels(self, self_elem_data_columns_values: List):
        """
        Generate attribute names.

        Parameters
        ----------
        self_elem_data_columns_values:List
            name
        Returns
        ---------
            ([str]) attribute labels.
        """
        name = np.array(self_elem_data_columns_values)[::self.couple]
        name = [i.split("_")[0] + "%s" % j for i in name for j in self.stats]
        self._feature_labels = name
        return name

    fit_transform = transform


class PolyFeature(BaseFeature, ABC):
    """
    Extension method.

    Such as degree = 2 means (x1x2,x1**2,x2**2)

    Examples
    -----------
    >>> n = np.array([[0,1,2,3,4,5],[0.422068,0.360958,0.201433,-0.459164,-0.064783,-0.250939]]).T
    >>> ps = pd.DataFrame(n,columns=["f1","f2"],index= ["x0","x1","x2","x3","x4","x5"])
    >>> pf = PolyFeature(degree=[1,2])
    >>> pf.fit_transform(n)

    n   f0^1     f1^1  f0^2  f0^1*f1^1      f1^2
    0   0.0  0.422068   0.0   0.000000  0.178141
    1   1.0  0.360958   1.0   0.360958  0.130291
    2   2.0  0.201433   4.0   0.402866  0.040575
    3   3.0 -0.459164   9.0  -1.377492  0.210832
    4   4.0 -0.064783  16.0  -0.259132  0.004197
    5   5.0 -0.250939  25.0  -1.254695  0.062970

    """

    def __init__(self, *, degree: Union[int, List[int]] = 3, n_jobs=1, on_errors='raise', return_type='df'):
        super(PolyFeature, self).__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

        if isinstance(degree, int):
            degree = [degree, ]
        self.degrees = degree

    @staticmethod
    def _combinations(n_features, degree):
        assert len(degree) ** n_features <= 1e6, "too much degree to calculate, plese depress the degree"
        return chain(*[combinations_with_replacement(range(n_features), i) for i in degree])

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y=None, **kwargs):

        if isinstance(X, pd.DataFrame):
            f_name = list(X.columns)
            s_name = list(X.index)
            X = X.values

        else:
            s_name = None
            f_name = None

        n_samples, n_features = check_array(X, accept_sparse=True).shape
        self.n_input_features_ = n_features
        self.set_feature_labels(f_name)

        combinations = list(self._combinations(n_features, self.degrees))
        columns = []
        for comb in combinations:
            out_col = 1
            x = X[:, comb]
            if x.ndim > 1:
                out_col = np.prod(x, axis=1)
            columns.append(out_col)
        ret = np.vstack(columns).T

        try:
            labels_len = len(self.feature_labels)
            if labels_len > 0:
                labels = self.feature_labels
            else:
                labels = None
        except (NotImplementedError, TypeError):
            labels = None

        if self.return_type == 'any':
            return ret

        if self.return_type == 'array':
            return np.array(ret)

        if self.return_type == 'df':
            if isinstance(f_name, List):
                return pd.DataFrame(ret, index=s_name, columns=labels)
            return pd.DataFrame(ret, columns=labels)

    def set_feature_labels(self, input_features=None):

        check_is_fitted(self, 'n_input_features_')
        if input_features is None:
            input_features = ['f%d' % i for i in range(self.n_input_features_)]
        else:
            if input_features is None:
                input_features = input_features
        combinations = self._combinations(self.n_input_features_, self.degrees)
        feature_names = []
        for rows in combinations:
            rows = Counter(rows).items()
            nas = ["{}^{}".format(input_features[k], v) for k, v in rows]
            names = "*".join(nas)

            feature_names.append(names)
        self._feature_labels = feature_names

# n = np.array([[0, 1, 2, 3, 4, 5], [0.422068, 0.360958, 0.201433, -0.459164, -0.064783, -0.250939]]).T
# ps = pd.DataFrame(n, columns=["f1", "f2"], index=["x0", "x1", "x2", "x3", "x4", "x5"])
# pf = PolyFeature(degree=[2, 3])
# a = pf.fit_transform(n)
