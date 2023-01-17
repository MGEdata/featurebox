# -*- coding: utf-8 -*-

# @Time    : 2019/10/18 14:27
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import copy
import os
import re
from itertools import chain
from typing import List

import pandas as pd


class _Ele:
    """Element type."""

    def __init__(self, name):
        self.name = [name, ]
        self.num = [1, ]

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            self_new = copy.deepcopy(self)
            self_new.num = [other * _ for _ in self.num]
            return self_new
        else:
            raise TypeError("can not mul {} and  {}".format(self.__str__, other.__str__))

    def __str__(self):
        return "".join(["{}{}".format(i, j) for i, j in zip(self.name, self.num)])

    __repr__ = __str__

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            self_new = copy.deepcopy(self)
            self_new.num = [other * _ for _ in self.num]
            return self_new
        else:
            raise TypeError("can not mul {} and  {}".format(self.__str__, other.__str__))

    def __add__(self, other):
        if isinstance(other, (float, int)):
            raise TypeError("can not add {} and number {}".format(self.__str__, other))
            pass
        elif isinstance(other, _Ele):
            self_new = copy.deepcopy(self)
            for i, j in zip(other.name, other.num):
                if i in self.name:
                    index = self.name.index(i)
                    self_new.num[index] += j
                else:
                    self_new.name.append(i)
                    self_new.num.append(j)
            return self_new

    def __pos__(self):
        return copy.deepcopy(self)

    @property
    def to_dict(self):
        return {i: j for i, j in zip(self.name, self.num)}

    @property
    def to_item(self):
        # print(list(zip(spath.x_name, spath.num)))
        return list(chain.from_iterable(zip(self.name, self.num)))


name_ele = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K',
            'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
            'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
            'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
            'Pu']

for ele_i in name_ele:
    locals()[ele_i] = _Ele(ele_i)


def _init_dict(element_n=None):
    if not element_n:
        element_n = len(name_ele)
    dict0 = dict()
    name_list_n = ["Abandon"] + name_ele
    for i in name_list_n[:element_n]:
        dict0[i] = 0
    return dict0


def _fea_dict(ele, element_n=None):
    dict0 = _init_dict(element_n=element_n)
    dicti = copy.copy(dict0)
    num = ele.num
    sname = ele.name
    for j, k in zip(sname, num):
        if j in dicti:
            dicti[j] = k
        else:
            dicti["Abandon"] = 1
    return dicti


def _transform(names, folds_name='folds.csv', expands_name=r'expands.csv'):
    c = []
    for j, i in enumerate(names):
        try:
            com = eval(i)
        except (NameError, SyntaxError) as e:
            print("{}, at {}, which is  {} ".format(e, j, i))
            com = _Ele(i)
        c.append(com)
    folds = pd.DataFrame([_i.to_item for _i in c], [str(_) for _ in c])
    expands = pd.DataFrame([_fea_dict(_i) for _i in c], [str(_) for _ in c])
    folds.to_csv(folds_name)
    expands.to_csv(expands_name)


def _pre_add(namei):
    a_ = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
          'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
          'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
          'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu']
    b = ['H', 'B', 'C', 'numeric', 'O', 'F', 'K', 'P', 'S', 'V', 'Y', 'I', 'W', 'U']

    num = re.findall(r"\d+\.?\d*", namei)

    namei_new = ""
    namei3 = namei
    j = 0
    while j < len(num):
        s1, s2, namei3 = namei3.partition(num[j])
        namei_new = r"".join([namei_new, s1, "_", s2])
        j += 1
    if namei3:
        namei_new += namei3

    namej = namei_new

    for x in a_:
        namej = re.sub(x, x + '1', namej)
    for x in b:
        namej = re.sub(x, x + '1', namej)
    needsub = re.findall(r'[A-Z]1[][a-z]', namej)
    for i in needsub:
        namej = re.sub(i, i[0] + i[2], namej)
    namej = re.sub(r'1?_', "", namej)

    fin = re.findall(r'\)[A-Z][a-z]*', namej)
    for i in fin:
        namej = re.sub(re.escape(i), ")1" + i[1:], namej, 1)

    return namej


def _bracket_follow(s):
    """
    for situation that the number of () before the ()
    if add this function, make sure the first number below ")" have "+" !!!, if not please add by human.
    """
    pre = re.findall(r'\d+\.?\d*\(.+\)\D*', s)
    long = len(pre)
    while long > 0:
        i = pre[0]
        a_, b, c = i.partition("(")
        d, e, f = c.partition(")")
        res = r"(" + d + r")" + a_ + f
        # s = re.sub(i, res, s, 1)
        s = s.replace(i, res, 1)
        pre = re.findall(r'\d+\.?\d*\(.\)\D*', s)
        long = len(pre)
    return s


def _add_mul(s):
    ret2 = re.findall(r'\d+\.?\d*', s)
    s_new = ""
    s3 = s
    j = 0
    while j < len(ret2):
        s1, s2, s3 = s3.partition(ret2[j])
        s_new = r"".join([s_new, "+", s1, "*", s2])
        j += 1
    if s3:
        s_new += "+" + s3
    s_new = s_new.replace(r"+ +", r"+", 1)
    s_new = s_new.replace(r"+)", r")")
    return s_new


def _substitued(s):
    # print(_bracket_follow.__doc__)
    s = _bracket_follow(s)
    s = _pre_add(s)
    s = _add_mul(s)
    return s


class NameSplit:
    """
    Split the name to table, return ``expand.csv`` and ``folds.csv``.

    .. note::

        Make sure the number below the element or (),
        For situation that the number of () before the (), such as "0.9(BaZr)+0.1(BaZrO3)".

        Please set the bracket_follow=True,
        and for this situation make sure the first number below ``)`` have ``+`` !!!.

    Examples
    --------
    >>> from featurebox.data.name_split import NameSplit
    >>> import os
    >>> os.chdir(r'.')
    >>> name = ['(Ti1.24La3)2',"((Ti1.24)2P2)1H0.2", "((Ti1.24)2)1H0.2",
    ... "((Ti1.24))1H0.2", "((Ti)2P2)1H0.2",  "((Ti))1H0.2"]
    >>> NSp = NameSplit()
    >>> NSp.transform(name)
    ...
    """

    def __init__(self, bracket_follow: bool = False):
        self.brack = bracket_follow

    def _substitute(self, s: List):
        if self.brack:
            print(_bracket_follow.__doc__)
            s = [_bracket_follow(si) for si in s]
        s = [_pre_add(si) for si in s]
        s = [_add_mul(si) for si in s]
        return s

    def transform(self, names: List[str], folds_name: str = 'folds.csv', expands_name: str = r'expands.csv'):
        """

        Parameters
        ----------
        names:list of str
            composition names list.
        folds_name:str
            return name, default is 'folds.csv'
        expands_name:str
            return name, default is 'expands.csv'

        Returns
        -------
        None
            return tables.csv.
        """
        names = self._substitute(names)
        _transform(names, folds_name, expands_name)


if __name__ == '__main__':
    a = [
        '(BaZr0.2Ti0.8O3)0.9(Ba0.7Ca0.3TiO3)0.1',
        '(Ba0.95Ca0.05)+(Zr0.1Ti0.9)O3',
        '(Ba(Zr0.2Ti0.8)O3)0.5 + ((Ba0.7Ca0.3)TiO3)0.5',
        'Ba0.95Sr0.05+(Ti0.95Zr0.05)0.975Sn0.025O3',
        "Ba0.95Sr0.05(Ti0.95Zr0.05)O3",
        "Ba0.9Sr0.1(Ti0.9Zr0.1)0.95Sn0.05O3",
        '(Ti1.24La3)2',
        "((Ti1.24)2P2)1H0.2", "((Ti1.24)2)1H0.2", "((Ti1.24))1H0.2",
        r"(TiLa)H2", r"(TiLa)0.2H2",
        r"(TiLa)0.2", '(TiLa)',
        '(TiLa3)', '(TiLa)', '(TiLa3)2',
        "((Ti)2P2)1H0.2", "((Ti)2)1H0.2", "((Ti))1H0.2"
    ]
    os.chdir(r'')

    NS = NameSplit()

    NS.transform(a)
