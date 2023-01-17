# -*- coding: utf-8 -*-

# @Time    : 2020/11/27 11:27
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

from itertools import zip_longest
from typing import List, Dict

import pandas as pd
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm


class MpAccess:
    """
    API for pymatgen database, access pymatgen to get data.

    Examples
    --------
    >>> mpa = MpAccess("Di28ZMunseR8vr57") # change yourself key.
    >>> ids = mpa.get_ids({"elements": {"$in": ["Al","O"]},'nelements': {"$lt": 2, "$gte": 1}})
    number 29
    >>> df = mpa.data_fetcher(mp_ids=ids, mp_props=['material_id', "cif"])
    Will fetch 29 inorganic compounds from Materials Project
    >>> structures_list = mpa.cifs_to_structures()
    ...

    """

    def __init__(self, api_key: str = "Di28ZMunseR8vr46"):
        """

        Parameters
        ----------
        api_key:str:
            pymatgen key.
        """
        self.m = MPRester(api_key)
        self.dff = None
        self.ids = None

    def data_fetcher(self, mp_ids: List[str] = None, mp_props: List[str] = None,
                     elasticity: bool = False) -> pd.DataFrame:
        """
        Fetch file from pymatgen.

        prop_name=['band_gap','density',"icsd_ids"'volume','material_id','pretty_formula','elements',"energy",
        'efermi','e_above_hull','formation_energy_per_atom','final_energy_per_atom','unit_cell_formula',
        'spacegroup','nelements'"nsites","final_structure","cif","piezo","diel"]

        Parameters
        ----------
        mp_ids:list of str
            list of MP id of pymatgen.
        mp_props:list of str
            prop_names
        elasticity:bool
            obtain elasticity or not.

        Returns
        -------
        pandas.DataFrame
            properties Table.
        """

        print('Will fetch %s inorganic compounds from Materials Project' % len(mp_ids))

        def grouper(iterable, n, fillvalue=None):
            """"
            Split requests into fixed number groups
            eg: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
            Collect data_cluster into fixed-length chunks or blocks"""
            args = [iter(iterable)] * n
            return zip_longest(fillvalue=fillvalue, *args)

        if mp_ids is None:
            mp_ids = self.ids
        # the following props will be fetched
        if mp_props is None:
            mp_props = [
                'band_gap',
                'density',
                "icsd_ids"
                'volume',
                'material_id',
                'pretty_formula',
                'elements',
                "energy",
                'efermi',
                'e_above_hull',
                'formation_energy_per_atom',
                'final_energy_per_atom',
                'unit_cell_formula',
                'spacegroup',
                'nelements'
                "nsites",
                "final_structure",
                "cif",
                "piezo",
                "diel"
            ]

        if elasticity:
            mp_props.append("elasticity")

        entries = []
        mpid_groups = [g for g in grouper(mp_ids, 40)]

        with self.m as mpr:
            for group in tqdm(mpid_groups):
                mpid_list = [ids for ids in filter(None, group)]
                chunk = mpr.query({"material_id": {"$in": mpid_list}}, mp_props)
                entries.extend(chunk)

        if 'material_id' in entries[0]:
            df = pd.DataFrame(entries, index=[e['material_id'] for e in entries])
        else:
            df = pd.DataFrame(entries)
        if 'unit_cell_formula' in df.columns:
            df = df.rename(columns={'unit_cell_formula': 'composition'})

        df = df.T

        self.dff = df

        return df

    def cifs_to_structures(self, cifs: List[str] = None) -> List[Structure]:
        """Get structures from cifs"""
        if cifs is None:
            if "cif" in self.dff.index:
                cifs = self.dff.loc["cif"]
            else:
                raise TypeError("cif is not in data")

        return [Structure.from_str(i, "cif") for i in cifs]

    def get_ids(self, criteria: Dict = None):
        """
        Search id by criteria.

        support_property = ['energy', 'energy_per_atom', 'volume', 'formation_energy_per_atom', 'nsites',
        'unit_cell_formula','pretty_formula', 'is_hubbard', 'elements', 'nelements', 'e_above_hull', 'hubbards',
        'is_compatible', 'spacegroup', 'task_ids',  'band_gap', 'density', 'icsd_id', 'icsd_ids', 'cif',
        'total_magnetization','material_id', 'oxide_type', 'tags', 'elasticity']

        Examples
        --------
        >>> from itertools import combinations
        >>> name_list = ["NaCl","CaCo3"]
        >>> criteria = {
        ... 'pretty_formula': {"$in": name_list},
        ... 'nelements': {"$lt": 3, "$gte": 3},
        ... 'spacegroup.number': {"$in": [225]},
        ... 'crystal_system': "cubic",
        ... 'nsites': {"$lt": 20},
        ... 'formation_energy_per_atom': {"$lt": 0},
        ... # "elements": {"$all": "O"},
        ... # "piezo":{"$ne": None}
        ... # "elements": {"$all": "O"},
        ... "elements": {"$in": list(combinations(["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], 5))}}

        where, ``"$gt"`` >,  ``"$gte"`` >=,  ``"$lt"`` <,  ``"$lte"`` <=,  ``"$ne"`` !=,  ``"$in"``,  ``"$nin"`` (not in),
        ``"$or"``,  ``"$and"``,  ``"$not"``,  ``"$nor"`` ,  ``"$all"``
        """

        if criteria is None:
            criteria = {}

        ids = self.m.query(
            criteria=criteria,
            properties=["material_id"])
        print("number %s" % len(ids))
        ids = pd.DataFrame(ids).values.ravel().tolist()
        self.ids = ids
        return ids

    def get_ids_from_web_table(self, path_file: str = None) -> List[str]:
        """This is method to read csv file download from web,the file name is '_Materials Project.csv',
        which contains "Materials Id" columns.
        """
        if path_file is None:
            path_file = "_Materials Project.csv"
        ids_data = pd.read_csv(path_file)
        ids = ids_data["Materials Id"]
        self.ids = ids
        return ids


if __name__ == "__main__":
    #############

    mpa = MpAccess("Di28ZMunseR8vr56")
    ids = mpa.get_ids({"elements": {"$in": ["Al", "O"]}, 'nelements': {"$lt": 2, "$gte": 1}})

    df = mpa.data_fetcher(mp_ids=ids, mp_props=['material_id', "cif"])
    structures_list = mpa.cifs_to_structures()
