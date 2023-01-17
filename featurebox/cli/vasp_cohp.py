# -*- coding: utf-8 -*-
# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os
from typing import List

import numpy as np
import pandas as pd
from path import Path
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster import Icohplist, Cohpcar

from featurebox.cli._basepathout import _BasePathOut


class COHPStartZero(_BasePathOut):
    """Get d band center from paths and return csv file.
    Download lobster from http://www.cohp.de/

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=True):
        super(COHPStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["WAVECAR", "INCAR", "DOSCAR"]
        self.out_file = "COHP_all.csv"
        self.software = ["lobster"]
        self.key_help = self.__doc__

    @staticmethod
    def read(path, store=False):
        data_all = {}
        if os.path.isfile("ICOHPLIST.lobster"):
            icohplist = Icohplist(are_coops=False, are_cobis=False, filename="ICOHPLIST.lobster")
            data = np.array([
                icohplist.icohplist["1"]["length"],
                -(icohplist.icohplist["1"]["icohp"][Spin.up]),
                -(icohplist.icohplist["1"]["icohp"][Spin.down])
            ]).ravel()
            data_all.update({path: data})

        else:
            print(f"no data for {path}")
            print(f"Check your ``lobsterout``, make sure enough ``NBANDS`` in INCAR")
            print(f"Check your VASP input and output, make sure enough ``LWAVE = .TRUE.`` in INCAR")
            data_all.update({path: None})

        data_all2 = {}
        if os.path.isfile("COHPCAR.lobster"):
            cohpcar = Cohpcar(are_coops=False, are_cobis=False, filename="COHPCAR.lobster")

            data = np.vstack((cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            # data = np.vstack((cohpcar.energies, cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            data_all2.update({path: data})
        else:
            print(f"no data for {path}")
            data_all2.update({path: None})

        result_single1 = pd.DataFrame.from_dict(data_all).T
        result_single1.columns = ["length", "icohp (up)", "icohp (down)"]
        result_single2 = pd.DataFrame.from_dict(data_all2).T

        if store:
            result_single1.to_csv("ICOHP_single.csv")
            print("'{}' are sored in '{}'".format("ICOHP_single.csv", os.getcwd()))
            result_single2.to_csv("COHP_single.csv")
            print("'{}' are sored in '{}'".format("COHP_single.csv", os.getcwd()))

        return result_single1

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # 1.外部软件
        cmds = ("lobster > look",)
        for i in cmds:
            print("Runing waiting for ")
            os.system(i)

        return self.read(path, store=self.store_single)

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = []
        col = None
        for pi in paths:
            try:
                res = pd.read_csv(pi / "ICOHP_single.csv", index_col=0)
                col = res.columns
                res = res.values
                data_all.append(res)
            except BaseException:
                print(f"No data for {pi}")

        data_all = np.concatenate(data_all, axis=0)
        result = pd.DataFrame(data_all, columns=col, index=paths)

        result.to_csv("ICOHP_all.csv")
        print("'{}' are sored in '{}'".format("ICOHP_all.csv", os.getcwd()))

        data_all2 = []
        col = None
        for pi in paths:
            try:
                res = pd.read_csv(pi / "COHP_single.csv", index_col=0)
                col = res.columns
                res = res.values
                data_all2.append(res)
            except BaseException:
                print(f"No data for {pi}")

        data_all2 = np.concatenate(data_all2, axis=0)
        result2 = pd.DataFrame(data_all2, columns=col, index=paths)

        result2.to_csv("COHP_all.csv")
        print("'{}' are sored in '{}'".format("COHP_all.csv", os.getcwd()))

        return result


class COHPStartInter(COHPStartZero):
    """
    For some system can't run this COHPStartZero.
    Download lobster from http://www.cohp.de/

    1. Copy follow code to form one 'lob.sh' file, and 'sh lob.sh' (change the atoms couple):

    Notes::

        #!/bin/bash

        old_path=$(cd "$(dirname "$0")"; pwd)

        for i in $(cat paths.temp)

        do

        cd $i

        echo $(cd "$(dirname "$0")"; pwd)

        echo COHPStartEnergy -10 > lobsterin

        echo COHPEndEnergy 5 >> lobsterin

        echo cohpBetween atom 45 atom 31 >> lobsterin

        lobster > look

        sleep 20m

        cd $old_path


        done

    2. tar -zcvf data.tar.gz "ICOHPLIST.lobster", "COHPCAR.lobster".

    3. Move to other system and run  'tar -zxvf data.tar.gz'.

    4. Run with this class.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=True):
        super(COHPStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["ICOHPLIST.lobster", "COHPCAR.lobster"]
        self.out_file = "ICOHP_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        return self.read(path, store=self.store_single)


class COHPStartSingleResult(COHPStartInter):
    """Avoid Double Calculation. Just reproduce the 'results_all' from a 'result_single' files.
    keeping the 'result_single.csv' files exists.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(COHPStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["COHP_single.csv", "ICOHP_single.csv"]
        self.out_file = "ICOHP_all.csv"
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
    批量提取 ICOHP，保存到当前工作文件夹。 查看参数帮助使用 -h。

    Notes:

        1.1 前期准备
        "WAVECAR", "INCAR", "DOSCAR"

        1.2 INCAR准备
        INCAR文件参数要求：
        LCHARG = .TRUE.
        LWAVE = .TRUE.
        NSW = 0
        IBRION = -1

        2.运行文件要求:
        lobster

    -j 参数说明：

        0               1               2
        [COHPStartZero, COHPStartInter, COHPStartSingleResult]

        0: 调用lobster软件运行。（首次启动）
        1: 调用lobster软件分析结果运行。（热启动,建议）
        2: 调用单个cohp_single.csv运行。（热启动）
        3: 调用pymatgen运行。

    补充:

        在 featurebox 中运行，请使用 featurebox cohp ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox cohp -f /home/sdfa/paths1.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox cohp -p /home/sdfa/
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=1)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        methods = [COHPStartZero, COHPStartInter, COHPStartSingleResult]
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods[args.job_type](n_jobs=1, store_single=True)
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
