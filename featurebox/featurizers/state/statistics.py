# -*- coding: utf-8 -*-

# @Time    : 2019/11/1 13:18
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause

from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
from pymatgen.core.composition import Composition as PMGComp
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.util.string import formula_double_format

from featurebox.featurizers.atom.mapper import AtomTableMap, BinaryMap
from featurebox.featurizers.state.extrastats import PropertyStats


class BaseCompositionFeature(BinaryMap):
    """
    BaseCompositionFeature is the basis for composition data.
    the subclass should be re-implemented, such as:
    ::

        def mix_function(self, elems:List, nums:List):
            w_ = np.array(nums)
            return w_.dot(elems)

    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        """
        Base class for composition feature.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        if data_map is None:
            data_map = AtomTableMap(tablename="oe.csv", search_tp="name")
        self.data_map = data_map
        # change
        self.data_map.weight = False
        self.data_map.n_jobs = 1

        self.search_tp = self.data_map.search_tp

    def convert_dict(self, atoms: dict) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        if isinstance(atoms, dict):
            atoms = [{k: v} for k, v in atoms.items()]

        numbers = np.array([list(ai.values())[0] for ai in atoms])

        ele = self.data_map.convert(atoms)
        if len(atoms) == 1:
            ele = np.array(ele).reshape((len(atoms), -1))
        return self.mix_function(ele, numbers)

    def convert_number(self, atoms: List) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        numbers = np.ones(len(atoms))

        ele = self.data_map.convert(atoms)
        if len(atoms) == 1:
            ele = np.array(ele).reshape((len(atoms), -1))
        return self.mix_function(ele, numbers)

    @abstractmethod
    def mix_function(self, elems: List, nums: Union[List, np.ndarray]):
        """

        Parameters
        ----------
        elems: list
            Elements in compound.
        nums: list
            Number of each element.

        Returns
        -------
        descriptor: numpy.ndarray
        """


class WeightedAverage(BaseCompositionFeature):
    """
    Examples
    ---------
    >>> from featurebox.featurizers.atom import AtomTableMap, AtomJsonMap
    >>> data_map = AtomJsonMap(search_tp="name", n_jobs=1)
    >>> wa = WeightedAverage(data_map, n_jobs=1,return_type="df")
    >>> x3 = [{"H": 2, "Pd": 1},{"He":1,"Al":4}]
    >>> wa.fit_transform(x3)
             0         1         2   ...        13        14        15
    0  0.422068  0.360958  0.201433  ... -0.459164 -0.064783 -0.250939
    1  0.007163 -0.471498 -0.072860  ...  0.206306 -0.041006  0.055843
    <BLANKLINE>
    [2 rows x 16 columns]

    >>> wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)])
    >>> wa.fit_transform(x3)
          fea_0     fea_1     fea_2  ...    fea_13    fea_14    fea_15
    0  0.422068  0.360958  0.201433  ... -0.459164 -0.064783 -0.250939
    1  0.007163 -0.471498 -0.072860  ...  0.206306 -0.041006  0.055843
    <BLANKLINE>
    [2 rows x 16 columns]

    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        w_ = nums / np.sum(nums)
        return w_.dot(elems)


class WeightedSum(BaseCompositionFeature):
    """
    Examples
    --------
    >>> from featurebox.featurizers.atom import AtomTableMap, AtomJsonMap
    >>> data_map = AtomJsonMap(search_tp="name", n_jobs=1)
    >>> wa = WeightedSum(data_map, n_jobs=1,return_type="df")
    >>> x3 = [{"H": 2, "Pd": 1},{"He":1,"Al":4}]
    >>> wa.fit_transform(x3)
             0         1         2   ...        13        14        15
    0  1.266204  1.082873  0.604300  ... -1.377492 -0.194350 -0.752816
    1  0.035813 -2.357490 -0.364302  ...  1.031530 -0.205029  0.279215
    <BLANKLINE>
    [2 rows x 16 columns]

    >>> wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)])
    >>> wa.fit_transform(x3)
          fea_0     fea_1     fea_2  ...    fea_13    fea_14    fea_15
    0  1.266204  1.082873  0.604300  ... -1.377492 -0.194350 -0.752816
    1  0.035813 -2.357490 -0.364302  ...  1.031530 -0.205029  0.279215
    <BLANKLINE>
    [2 rows x 16 columns]

    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        w_ = np.array(nums)
        return w_.dot(elems)


class GeometricMean(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems: np.ndarray, nums):
        w_ = np.array(nums).reshape(-1, 1)
        tmp = elems ** w_
        return np.power(tmp.prod(axis=0), 1 / sum(w_))


class HarmonicMean(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        w_ = np.array(nums)
        tmp = w_.dot(elems)
        return sum(w_) / tmp


class WeightedVariance(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems: np.ndarray, nums):
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems)
        var_ = elems - mean_
        return w_.dot(var_ ** 2)


class MaxPooling(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        return np.max(elems, axis=0)


class MinPooling(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map=data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        return np.min(elems, axis=0)


class ExtraMix(BaseCompositionFeature):
    """
    See Also:
        :class:`WeightedSum`
    """

    def __init__(self, data_map: BinaryMap, stats: Tuple[str] = ("mean",), n_jobs: int = 1,
                 on_errors: str = 'raise', return_type: str = 'df'):
        super().__init__(data_map=data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.stats = stats

    def mix_function(self, elems, nums):
        all_attributes = []
        for stat in self.stats:
            all_attributes.append(PropertyStats.calc_stat(elems, stat, nums))

        return np.array(all_attributes).ravel()


class DepartElementFeature(BaseCompositionFeature):
    """
    Get the table of element data.

    Examples
    ----------
    >>> from featurebox.featurizers.atom.mapper import AtomJsonMap
    >>> from featurebox.featurizers.state.union import UnionFeature
    >>> from featurebox.featurizers.state.statistics import DepartElementFeature
    >>> data_map = AtomJsonMap(search_tp="name",embedding_dict="ele_megnet.json", n_jobs=1) # keep this n_jobs=1 and return_type="np"
    >>> wa = DepartElementFeature(data_map,n_composition=2, n_jobs=1, return_type="pd")
    >>> comp = [{"H": 2, "Pd": 1},{"He":1, "Al":4}]
    >>> wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)]) # 16 this the feature number of built-in "ele_megnet.json"
    >>> couple_data = wa.fit_transform(comp)
        fea_0_0   fea_0_1   fea_1_0  ...  fea_14_1  fea_15_0  fea_15_1
    0  0.352363  0.561478  0.635952  ... -0.236541 -0.270104 -0.212607
    1 -0.067220  0.025758  0.141113  ... -0.092577 -0.042185  0.080350
    <BLANKLINE>
    [2 rows x 32 columns]

    """

    def __init__(self, data_map: BinaryMap, n_composition: int, n_jobs: int = 1, on_errors: str = 'raise',
                 return_type: str = 'df'):

        super().__init__(data_map=data_map, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.n_composition = n_composition

    def mix_function(self, elems: np.ndarray, nums=None):

        return np.array(elems).ravel(order='F')

    def convert_dict(self, atoms: Union[dict, PMGComp]) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        if isinstance(atoms, PMGComp):
            sym_amt = atoms.get_el_amt_dict()
            syms = sorted(sym_amt.keys(), key=lambda sym: get_el_sp(sym).X)
            atoms = {s: formula_double_format(sym_amt[s], False) for s in syms}
        if isinstance(atoms, dict):
            atoms = [{k: v} for k, v in atoms.items()]

        numbers = np.array([list(ai.values())[0] for ai in atoms])

        ele = self.data_map.convert(atoms)
        if len(atoms) == 1:
            ele = np.array(ele).reshape((len(atoms), -1))
        return self.mix_function(ele, numbers)

    def convert_number(self, atoms: List) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        numbers = np.ones(len(atoms))

        ele = self.data_map.convert(atoms)
        if len(atoms) == 1:
            ele = np.array(ele).reshape((len(atoms), -1))
        return self.mix_function(ele, numbers)

    def set_feature_labels(self, values):
        self._feature_labels = [str(s) + "_" + str(n) for s in values for n in range(self.n_composition)]
