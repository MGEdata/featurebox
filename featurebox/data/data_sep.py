# -*- coding: utf-8 -*-

# @Time  : 2022/8/1 15:56
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import warnings
from copy import copy
from typing import Union, Dict, Any, Tuple

import pandas as pd


class DataSameSep:
    """Settle data, dispatch data with "all" mark to each site.
    Make sure the values of dict are Immutable type,such as float,init. Otherwise, the stored data would
    change with the input data, even if later than the call of this class/function.

    Examples:
    --------------
    >>> d1 = {"Ta-S1":{"bond1":3.4,"bond2":3.5},"Co-S2":{"bond1":3.2,"bond2":3.1},"Fe-Sall":{"bond1":3.2,"bond2":3.1}}
    >>> dss = DataSameSep(d1)
    >>> dss["Ta-S1"]={"bond1":3.2,"bond2":3.5} # cover the old.
    >>> dss.replace({"Ta-S1":{"bond1":3.4,"bond2":3.5},"Co-S2":{"bond1":3.2,"bond2":3.1}}) # cover the old.
    >>> dss.replace_entry(label="Ta",site=1,entry={"bond1":3.2,"bond2":3.5}) # cover the old.

    >>> dss.update({"Ta-S1":{"bond1":3.4,"bond2":3.5},"Co-S2":{"bond1":3.2,"bond2":3.1}}) # add
    >>> dss.update_entry(label="Co",site=0,entry={"bond1":3.2}) # add
    >>> dss.update_entry_kv(label="Mg",site="all",key="bond1",value=3.2) # add
    >>> dict_data = dss.settle()
    >>> pd_data = dss.settle_to_pd(sort=True)
    >>> print(pd_data)
           bond1  bond2
    Co-S0    3.2    NaN
    Co-S2    3.2    3.1
    Fe-S0    3.2    3.1
    Fe-S1    3.2    3.1
    Fe-S2    3.2    3.1
    Mg-S0    3.2    NaN
    Mg-S1    3.2    NaN
    Mg-S2    3.2    NaN
    Ta-S1    3.2    3.5
    """

    def __init__(self, data: Dict = None, sep="-", sites_name="S", dup=3, prefix=None):
        """
        Make sure the key of data are formatted by {label}-{Si or Sall} !!! and all values is dict type.
        The 'S' is the same with sites_name.

        Parameters
        ----------
        data: (dict of dict)
            first key are formated by {label}{sep}{Si or Sall}.
        sep: (str)
            default "-".
        sites_name: (str)
            default "S".
        dup: (int)
            default 3.
        prefix: (str)
            the class prefix of one batch data.

        """

        if data is None:
            data = {}
        self.data = copy(data)
        self.sites_name = sites_name
        self.dup = dup
        self.data_settled = {}
        self.sep = sep
        self.label = []
        self.mark = [f"{sites_name}{i}" for i in range(dup)] + [f"{sites_name}all"]
        self._check()
        self.prefix = prefix

    def _check(self):
        key = list(self.data.keys())
        if len(key) > 0:
            try:
                labels, marks = list(zip(*[i.split(self.sep) for i in key]))
            except BaseException as e:
                print(e)
                print(f"Split name to labels and marks by failed using {self.sep}")
                print(key)
                print("Try to split name to prefixs, labels, marks now.")
                try:
                    prefixs, labels, marks = list(zip(*[i.split(self.sep) for i in key]))
                    prefix = list(set(prefixs))
                    if len(prefix) >= 1:
                        print(f"There are {len(prefix)} prefix.")
                except BaseException as e:
                    print(e)
                    raise ValueError(f"The key force keep naming '{{label}}{self.sep}{{Sx}}' "
                                     f"or '{{prefix}}{self.sep}{{label}}{self.sep}{{Sx}}, no more '{self.sep}' please.")

            self.label = list(set(labels))

            assert set(marks) | set(self.mark) == set(self.mark), f"The {set(marks)} are out of range. " \
                                                                  f"please using in {self.mark}."

    def spilt(self, prefix_label_site="") -> Tuple:
        """Try to get prefix,label,site_number."""
        try:
            labels, site = prefix_label_site.split(self.sep)
            prefix = None
        except ValueError:
            try:
                prefix, labels, site = prefix_label_site.split(self.sep)
            except BaseException:
                raise ValueError("The key should named '{label}-{Sx}' or '{prefix}-{label}-{Sx}'. ")
        site_num = site.replace(self.sites_name, "")
        if site_num == "all":
            pass
        else:
            try:
                site_num = int(site_num)
            except ValueError:
                raise ValueError(f"The '{site}' are not consistent with {self.mark}.")
            if site_num >= self.dup:
                warnings.warn(f"The 'site number' ({site_num}) should be less than 'self.dup' ({self.dup}).")
        return prefix, labels, site_num

    def __setitem__(self, key, value: Dict):
        assert self.sep in key
        assert key.split(self.sep)[-1] in self.mark
        assert len(key.split(self.sep)) in [2, 3], "The key should named '{label}-{Sx}' or '{prefix}-{label}-{Sx}'."
        if self.prefix:
            if len(key.split(self.sep)) != 3:
                raise UserWarning(f"There are {self.prefix} in definition but the key  {key} is with out prefix."
                                  f"Advise use {{prefix}}-{{label}}-{{Sx}}")
        self.data.update({key: copy(value)})

    def update(self, data: Dict):
        """
        Add dict data.

        Parameters
        ----------
        data:dict
            {entry_key: entry}.
        """
        for ki, vi in data.items():
            if ki not in self.data:
                self.data[ki] = copy(vi)
            else:
                self.data[ki].update(vi)

    def replace(self, data: Dict):
        """
        Replace dict data.

        Parameters
        ----------
        data:dict
            {entry_key: entry}.
        """
        self.data.update(copy(data))

    def _get_entry_name(self, label, site, prefix):
        if isinstance(site, int):
            pass
        else:
            assert f"{self.sites_name}{site}" in self.mark, "Keep site is int or 'all'."
        if prefix:
            entry_key = f"{prefix}{self.sep}{label}{self.sep}{self.sites_name}{site}"
        elif self.prefix:
            entry_key = f"{self.prefix}{self.sep}{label}{self.sep}{self.sites_name}{site}"
        else:
            entry_key = f"{label}{self.sep}{self.sites_name}{site}"
        return entry_key

    def update_entry(self, label: str, site: Union[int, str], entry: Dict, prefix=None):
        """
        Add dict data to entry.

        Parameters
        ----------
        label:str
            label name.
        site:int
            number small than self.dup, or "all".
        entry:dict
            entry data.
        prefix:str
            prefix name for batch of data.
        """
        entry_key = self._get_entry_name(label, site, prefix)
        if entry_key not in self.data:
            self.data[entry_key] = copy(entry)
        else:
            self.data[entry_key].update(entry)

    def update_entry_kv(self, label: str, site: Union[int, str], key: str, value: Any, prefix=None):
        """
        Add dict data to entry.

        Parameters
        ----------
        label:str
            label name.
        site:int
            number small than self.dup, or "all".
        key:str
            name of property.
        value:any
            value (float, int, str)
        prefix:str
            prefix name for batch of data.
        """
        return self.update_entry(label, site, {key: value}, prefix)

    def replace_entry(self, label: str, site: Union[int, str], entry: Dict, prefix=None):
        """
        Replace entry!! This would cover the old entry.

        Parameters
        ----------
        label:str
            label name.
        site:int
            number small than self.dup, or "all".
        entry:dict
            entry data.
        prefix:str
            prefix name for batch of data.
        """
        entry_key = self._get_entry_name(label, site, prefix)
        self.data.update({entry_key: copy(entry)})

    def settle(self, sort=False) -> Dict:
        """
        Settle data and return a formed dict.

        Parameters
        ----------
        sort:bool
            sort the entry keys or not.

        Returns
        -------
        data_settled:dict
            new dict.
        """
        self._check()
        for entry_key in self.data.keys():
            if "all" in entry_key:
                label = entry_key.replace(f"{self.sep}{self.sites_name}all", "")
                for site in range(self.dup):
                    nk = f"{label}{self.sep}{self.sites_name}{site}"
                    if nk in self.data_settled:
                        self.data_settled[nk].update(self.data[entry_key])
                    else:
                        self.data_settled.update({nk: copy(self.data[entry_key])})
            else:
                if entry_key in self.data_settled:
                    self.data_settled[entry_key].update(self.data[entry_key])
                else:
                    self.data_settled.update({entry_key: copy(self.data[entry_key])})
        if sort:
            self.data_settled = {k[0]: k[1] for k in sorted(self.data_settled.items(), key=lambda x: x[0])}

        return self.data_settled

    def settle_to_pd(self, sort=False) -> pd.DataFrame:
        """
        Settle data and return a formed pd.Dataframe.

        Parameters
        ----------
        sort:bool
            sort the entry keys or not.

        Returns
        -------
        data_settled:pd.Dataframe
            new table.
        """
        data = self.settle(sort=sort)
        return pd.DataFrame.from_dict(data).T

    def update_from_pd(self, df: Union[pd.DataFrame, str]):
        """
        Read table and update to data.
        The table must be the formed by self.settle_to_pd function.

        if df is str, try: df = pd.read_csv("df_name", index_col=0).T

        Parameters
        ----------
        df:(pd.DataFrame,str)

        """
        if isinstance(df, str):
            df = pd.read_csv(df, index_col=0)
        df_dict = df.T.to_dict()
        self.update(df_dict)
