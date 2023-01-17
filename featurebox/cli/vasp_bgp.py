# -*- coding: utf-8 -*-
# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os
import re
from typing import List, Callable

import numpy as np
import pandas as pd
from path import Path
from pymatgen.io.vasp import Vasprun

from featurebox.cli._basepathout import _BasePathOut


class BandGapPy(_BasePathOut):
    """
    Get band gap from vasprun.xml by pymatgen.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BandGapPy, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["vasprun.xml"]
        self.out_file = "bgp_all.csv"
        self.software = []

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        vasprun = Vasprun(path / "vasprun.xml")
        ef = vasprun.efermi
        data = {"efermi": ef}
        res = vasprun.eigenvalue_band_properties
        [data.update({k: v}) for k, v in zip(["band_gap", "cbm", "vbm", "is_band_gap_direct"], res)]

        if self.store_single:
            result = pd.DataFrame.from_dict({f"{path}": data}).T
            print("'{}' are sored in '{}'".format("bgp_single.csv", os.getcwd()))
            result.to_csv("bgp_single.csv")

        return data

    def batch_after_treatment(self, paths, res_code):  # fill
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        # 合并多个文件到一个csv.
        # Sample
        data_all = {pi: ri for pi, ri in zip(paths, res_code)}
        result = pd.DataFrame(data_all).T
        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


class BandGapStartZero(_BasePathOut):
    """Get band gap from paths and return csv file.
    VASPKIT Version: 1.2.1 or below. Download vaspkit from
    https://vaspkit.com/installation.html#download

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BandGapStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["EIGENVAL", "INCAR", "DOSCAR"]
        self.out_file = "bgp_kit_all.csv"
        self.software = ["vaspkit"]
        self.key_help = "Make sure the Vaspkit <= 1.2.1."

    @staticmethod
    def read(path, store=False, store_name="bgp_kit_single.csv"):
        file_name = path / "BAND_GAP"
        with open(file_name) as f:
            res = f.readlines()

        res = [i for i in res if "(eV)" in i]

        name = [i.split(":")[0].replace("  ", "") for i in res]
        name = [i[1:] if i[0] == " " else i for i in name]
        value = [float(i.split(" ")[-1].replace("\n", "")) for i in res]

        result = {"File": str(path)}
        for ni, vi in zip(name, value):
            result.update({ni: vi})

        if round(value[1], 2) <= round(value[3], 2) <= round(value[2], 2):
            result.update({'Fermi Energy (eV) Center': (value[1] + value[2]) / 2})
        else:
            result.update({'Fermi Energy (eV) Center': value[3]})

        result = {path: result}

        result = pd.DataFrame.from_dict(result).T

        if store:
            result.to_csv(store_name)
            print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

        return result

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # 1.外部软件
        cmds = ("vaspkit -task 911 > BAND_GAP",)
        for i in cmds:
            os.system(i)

        return self.read(path, store=self.store_single, store_name="bgp_kit_single.csv")

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = []
        col = None
        for res in res_code:
            if isinstance(res, pd.DataFrame):
                col = res.columns
                res = res.values
                data_all.append(res)

        data_all = np.concatenate(data_all, axis=0)
        result = pd.DataFrame(data_all, columns=col, index=paths)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result

    @staticmethod
    def extract(data, *args, format_path: Callable = None, **kwargs):
        """
        Extract the message in data, and formed it.

        Parameters
        ----------
        data:pd.DateFrame
            transformed data.
        format_path:Callable
            function to deal with each path, for better shown.

        Returns
        -------
        res_data:pd.DateFrame
            extracted and formed data.

        """
        if format_path == "default":
            format_path = lambda x: re.split(r" |-|/|\\", x)[-2]
        elif format_path is None:
            format_path = lambda x: x

        if "Unnamed: 0" in data:
            data["File"] = data["Unnamed: 0"]  # File are sole
            del data["Unnamed: 0"]
            data = data.set_index("File")

        data = data[["Band Gap (eV)", "Eigenvalue of CBM (eV)", "Eigenvalue of VBM (eV)", "Fermi Energy (eV)",
                     "Fermi Energy (eV) Center"]]
        data.index = [format_path(ci) for ci in data.index]
        return data


class BandGapStartInter(BandGapStartZero):
    """
    For some system can't run this BandGapStartZero.
    Download vaspkit from
    https://vaspkit.com/installation.html#download

    1. Copy follow code to form one 'bg.sh' file, and 'sh bg.sh':

    Notes::

        #!/bin/bash

        old_path=$(cd "$(dirname "$0")"; pwd)

        for i in $(cat paths.temp)

        do

        cd $i

        echo $(cd "$(dirname "$0")"; pwd)

        vaspkit -task 911 > BAND_GAP

        cd $old_path

        done

    2. tar -zcvf data.tar.gz BAND_GAP.

    3. Move to other system and run  'tar -zxvf data.tar.gz'.

    4. Run with this class.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BandGapStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["BAND_GAP"]
        self.out_file = "bgp_kit_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        return self.read(path, store=self.store_single, store_name="bgp_kit_single.csv")


class BandGapStartSingleResult(BandGapStartZero):
    """Avoid Double Calculation. Just reproduce the 'results_all' from a 'result_single' files.
    keeping the 'result_single.csv' files exists.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BandGapStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["bgp_kit_single.csv"]
        self.out_file = "bgp_kit_all.csv"
        self.software = []

    def read(self, path, **kwargs):
        pii = path / self.necessary_files[0]
        if pii.isfile():
            result = pd.read_csv(pii)
            return result
        else:
            return None

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        return self.read(path)


class _CLICommand:
    """
    批量提取 band gap center，保存到当前工作文件夹。 查看参数帮助使用 -h。

    Notes:

        1.1 前期准备
        "EIGENVAL", "INCAR", "DOSCAR","POSCAR"

        1.2 INCAR准备
        INCAR文件参数要求：
        LORBIT=11
        NSW = 0
        IBRION = -1

        2.运行文件要求:
        vaspkit <= 1.2.1, for -j in (0,1)

    -j 参数说明：

        0                 1                  2                         3
        BandGapStartZero, BandGapStartInter, BandGapStartSingleResult, BandGapPy

        0: 调用vaspkit软件运行。（首次启动）
        1: 调用vaspkit软件分析结果运行。（热启动）
        2: 调用单个bandgap_single.csv运行。（热启动）
        3: 调用pymatgen运行。

    补充:

        在 featurebox 中运行，请使用 featurebox bandgap ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox bandgap -f /home/sdfa/paths1.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox bandgap -p /home/sdfa/
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=3)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        methods = [BandGapStartZero, BandGapStartInter, BandGapStartSingleResult, BandGapPy]
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods[args.job_type](n_jobs=4, store_single=True)
            with open(pf) as f:
                wd = f.readlines()
            assert len(wd) > 0, f"No path in file {pf}"
            bad.transform(wd)
        elif pn.isdir():
            bad = methods[args.job_type](n_jobs=1, store_single=True)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"Get data by {__file__}. Examples:\n"
                                                 "python this.py -p /home/dir_name , or\n"
                                                 "python this.py -f /home/dir_name/paths.temp")
    _CLICommand.add_arguments(parser=parser)
    args = _CLICommand.parse_args(parser=parser)
    _CLICommand.run(args=args, parser=parser)
