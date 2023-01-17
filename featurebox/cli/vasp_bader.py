# -*- coding: utf-8 -*-

# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import os
from typing import List, Callable

import numpy as np
import pandas as pd
from path import Path
from pymatgen.io.vasp import Potcar, Poscar

from featurebox.cli._basepathout import _BasePathOut


class BaderStartZero(_BasePathOut):
    """Get bader from paths and return csv file.

    1.Download bader from
    http://theory.cm.utexas.edu/henkelman/code/bader/

    2. Download chgsum.pl from
    http://theory.cm.utexas.edu/vtsttools/download.html

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["AECCAR0", "AECCAR2", "CHGCAR", "POTCAR", "CONTCAR"]
        self.out_file = "bader_all.csv"
        self.software = ["chgsum.pl", "bader"]
        self.inter_temp_files = ["ACF.dat", "CHGCAR_sum"]
        self.key_help = "Make 'LAECHG=.TRUE.' 'LCHARG = .TRUE.'  'IBRION = -1'  in VASP INCAR."

    @staticmethod
    def read(path, store=False, store_name="bader_single.csv"):
        """Run linux cmd and return result, make sure the bader is installed."""
        if (path / "POTCAR").isfile() and (path / "CONTCAR").isfile:
            potcar = Potcar.from_file("POTCAR")
            symbols = Poscar.from_file("CONTCAR", check_for_POTCAR=False).structure.atomic_numbers

            zval = []
            for i in symbols:
                for j in potcar:
                    if j.atomic_no == i:
                        zval.append(j.ZVAL)
                        break

            zval = np.array(zval)

            with open(path / "ACF.dat", mode="r") as f:
                msg = f.readlines()

            msg = [i for i in msg if ":" not in i]
            msg = [i for i in msg if "--" not in i]
            msg = [i.replace("\n", "") if "\n" in i else i for i in msg]
            msg = [i.split(" ") for i in msg[1:]]
            msg = [[k for k in i if k != ""] for i in msg]
            msg2 = np.array(msg).astype(float)

            msg_append = (msg2[:, 4] - zval).reshape(-1, 1)
            msg_f = np.full(msg2.shape[0], fill_value=str(path)).reshape(-1, 1)
            msg2 = np.concatenate((msg_f, msg2, msg_append), axis=1)
            ta = np.array(["File", "Atom Number", "X", "Y", "Z", "CHARGE", " MIN DIST", " ATOMIC VOL",
                           "Bader Ele Move"])

            result = pd.DataFrame(msg2, columns=ta)
            if store:
                result.to_csv(store_name)
                print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

            return result
        else:
            return None

    @staticmethod
    def extract(data: pd.DataFrame, atoms, format_path: Callable = None):

        if isinstance(atoms, (list, tuple)):
            data = data.apply(pd.to_numeric, errors='ignore')
            res = []
            if format_path is not None:
                data["File"] = [format_path(ci) for ci in data["File"]]
            for v in atoms:
                sel = data[data["Atom Number"] == v + 1]
                sel = sel[["File", "Bader Ele Move"]].set_index("File")
                n_name = [f"{i}-{v}" for i in sel.columns]
                sel.columns = n_name
                res.append(sel)
            return pd.concat(res, axis=1)
        else:
            raise NotImplementedError("'atoms' just accept list or tuple.")

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""

        # 1.外部软件
        cmds = ("chgsum.pl AECCAR0 AECCAR2", "bader CHGCAR -ref CHGCAR_sum")
        for i in cmds:
            os.system(i)

        return self.read(path, store=self.store_single, store_name="bader_single.csv")

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
        result = pd.DataFrame(data_all, columns=col)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


class BaderStartInter(BaderStartZero):
    r"""Get bader from paths and return csv file. For some system can't run BaderStartZero.

    Download bader from
    http://theory.cm.utexas.edu/henkelman/code/bader/

    Download chgsum.pl from
    http://theory.cm.utexas.edu/vtsttools/download.html

    1. Copy follow code to form one 'badertoACF.sh' file, and 'sh badertoACF.sh':

    Notes::
        
        #!/bin/bash

        old_path=$(cd "$(dirname "$0")"; pwd)

        for i in $(cat paths.temp)

        do

        cd $i

        echo $(cd "$(dirname "$0")"; pwd)

        chgsum.pl AECCAR0 AECCAR2

        bader CHGCAR -ref CHGCAR_sum

        cd $old_path

        done

    2. tar -zcvf data.tar.gz ACF.dat POTCAR POSCAR.

    3. Move to other system and run  'tar -zxvf data.tar.gz'.

    4. Run with this class (-j 1).
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["ACF.dat", "POTCAR", "CONTCAR"]
        self.out_file = "bader_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        return self.read(path, store=self.store_single, store_name="bader_single.csv")


class BaderStartSingleResult(BaderStartZero):
    """Get bader from paths and return csv file.
    Avoid Double Calculation. Just reproduce the 'results_all' from a 'result_single' files.
    keeping the 'result_single.csv' files exists.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["bader_single.csv"]
        self.out_file = "bader_all.csv"
        self.software = []
        self.key_help = self.__doc__

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
    批量提取 bader，保存到当前工作文件夹。 查看参数帮助使用 -h。

    Notes:

        1.1 前期准备
        需要以下文件：~/bin/bader, ~/bin/chgsum.pl
        并赋予权限:
        chmod u+x ~/bin/chgsum.pl
        chmod u+x ~/bin/bader

        1.2 INCAR准备
        INCAR文件参数要求：
        LAECHG = .TRUE.
        LCHARG = .TRUE.
        NSW = 0
        IBRION = -1

        2.运行文件要求:
        chgsum.pl, bader,
        AECCAR0，AECCAR2，CHGCAR

    -j 参数说明：

        0               1                2
        BaderStartZero, BaderStartInter, BaderStartSingleResult

        0: 调用bader软件运行。（首次启动）
        1: 调用bader软件分析结果运行。（热启动）
        2: 调用单个bader_single.csv运行。（热启动）

    补充:

        在 featurebox 中运行，请使用 featurebox bader ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox bader -f /home/sdfa/paths1.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox bader -p /home/sdfa/
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=0)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        methods = [BaderStartZero, BaderStartInter, BaderStartSingleResult]
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
