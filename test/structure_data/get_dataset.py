# -*- coding: utf-8 -*-

# @Time    : 2021/8/5 22:55
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pathlib

import pandas as pd
from pymatgen.core import Structure

path = pathlib.Path(__file__).parent
data01 = pd.read_pickle(path / "data_structure.pkl_pd")
data02 = pd.read_pickle(path / "data_structure2.pkl_pd")
data03 = Structure.from_file(str(path / "674718.cif"))
