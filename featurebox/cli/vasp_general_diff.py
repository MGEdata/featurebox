# -*- coding: utf-8 -*-
# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import os
from typing import List

import pandas as pd
from path import Path

from featurebox.cli._basepathout import _BasePathOut2


class GeneralDiff(_BasePathOut2):
    """Get data from couples of paths and return csv file.

    Notes::

        mod="pymatgen.io.vasp"         # Module to get class.
        cmd="Vasprun"                  # class to get object.
        necessary_files="vasprun.xml"  # class input file.
        prop="final_energy"            # class.property name.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False,
                 mod="pymatgen.io.vasp", cmd="Vasprun", necessary_files="vasprun.xml", prop="final_energy"):
        super(GeneralDiff, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        from importlib import import_module
        mod = import_module(mod)
        self.cmd = getattr(mod, cmd)
        self.necessary_files = [necessary_files, ]
        self.prop = prop
        self.out_file = f"{prop}-diff_all.csv"
        self.software = []

    def run(self, paths: List[Path], files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        res = []
        for path in paths:
            try:
                vasprun = self.cmd(path / self.necessary_files[0])
            except BaseException:
                vasprun = self.cmd.from_file(path / self.necessary_files[0])

            data = getattr(vasprun, self.prop)

            if isinstance(data, (float, int)):
                res.append(data)
            else:
                raise NotImplementedError("Try to transform to float or int but failed."
                                          f"This {self.prop} could not be extracted using this script.")
        data = {f"{self.prop}-diff": res[0] - res[1]}

        if self.store_single:
            result = pd.DataFrame.from_dict({f"{paths[0]}-{paths[1]}": data}).T
            result.to_csv(f"{self.prop}-diff_single.csv")
            print(f"Store {self.prop}-diff_single.csv to {os.getcwd()}.")
            # This file would be covered！！！

        return data

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = {f"{pi[0]}-{pi[1]}": ri for pi, ri in zip(paths, res_code)}
        result = pd.DataFrame(data_all).T
        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


class _CLICommand:
    """
    批量获取性质差。

    本脚本可适用于调取绝大多数pymatgen对象的性质差，请自由搭配。（默认两个vasprun.xml能量差）。

    本脚本较为特殊， -f 或者-p 必须输入两组参数 -f /home/sdfa/paths1.temp /home/sdfa/paths2.temp， 查看参数帮助使用 -h。

    扩展阅读:

        >>> # 实际操作步骤如下所示，默认四个关键参数如下。
        >>> # mod="pymatgen.io.vasp", cmd="Vasprun", necessary_files="vasprun.xml", prop="final_energy"
        >>> from pymatgen.io.vasp import Vasprun
        >>> vr= Vasprun("vasprun.xml") # or
        >>> result = vr.final_energy

    扩展案例:

        featurebox diff -prop efermi -f /home/sdfa/paths.temp

    补充:

        在 featurebox 中运行，请使用 featurebox diff ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定两组路径所在文件:

        $ featurebox diff -f /home/sdfa/paths1.temp /home/sdfa/paths2.temp

        如果在 featurebox 中运行单个案例，请指定两个运算子文件夹:

        $ featurebox diff -p /home/sdfa/ /home/sdfa2/
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default=("./data", "./data2"), nargs=2,
                            help="two paths")
        parser.add_argument('-f', '--paths_file', type=str, default=("paths1.temp", "paths2.temp"), nargs=2,
                            help="two files containing paths.")
        parser.add_argument('-mod', '--mod', type=str, default='pymatgen.io.vasp',
                            help="which module to import. such as: 'pymatgen.io.vasp'.")

        parser.add_argument('-cmd', '--cmd', type=str, default='Vasprun',
                            help="which python class to call the necessary file. such as: 'Vasprun'.")
        parser.add_argument('-nec', '--nec', type=str, default='vasprun.xml',
                            help="necessary file. such as: 'vasprun.xml'.")
        parser.add_argument('-prop', '--prop', type=str, default='final_energy',
                            help="property name. such as: 'final_energy'.")
        # mod = "pymatgen.io.vasp", cmd = "Vasprun", necessary_files = "vasprun.xml", prop = "final_energy"

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):

        pf = [Path(i) for i in args.paths_file]
        pn = [Path(i) for i in args.path_name]
        if pf[0].isfile():
            bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, n_jobs=4)
            with open(pf[0]) as f1:
                wd1 = f1.readlines()
            with open(pf[1]) as f2:
                wd2 = f2.readlines()
            assert len(wd1) > 0, f"No path in file {pf[0]}"
            assert len(wd2) > 0, f"No path in file {pf[1]}"
            bad.transform([list(i) for i in zip(wd1, wd2)])
        elif pn[0].isdir():
            bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, store_single=False)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:
        $ python this.py -p /home/dir_name /home/dir_name2
        $ python this.py -f /home/dir_name/path.temp1  /home/dir_name/path.temp2
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"Get data by {__file__}. Examples:\n"
                                                 "python this.py -p /home/dir_name /home/dir_name2, or\n"
                                                 "python this.py -f /home/dir_name/path.temp1  /home/dir_name/path.temp2")
    _CLICommand.add_arguments(parser=parser)
    args = _CLICommand.parse_args(parser=parser)
    _CLICommand.run(args=args, parser=parser)
