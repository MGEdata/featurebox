# -*- coding: utf-8 -*-

# @Time  : 2022/8/5 22:23
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import multiprocessing
import os
import pathlib
import re
import warnings
from abc import abstractmethod
from typing import Union, Callable, List, Any

import pandas as pd
from path import Path
from tqdm import tqdm


class _BasePathOut:
    """
    class for deal with the software and files in batch.

    The subclass should fill the follow properties and functions.

    self.necessary_files = []  # fill
    self.out_file = ""  # fill
    self.software = []  # fill
    self.inter_temp_files = []  # fill
    self.key_help = []  # fill


    def run(self, path: Path, files: List = None):
        ...  # fill

    def batch_after_treatment(self, paths, res_code):
        ...  # fill


    cls.__dos__ = "This is the doc of class."

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, log=True):
        self.n_jobs = n_jobs
        self.tq = tq
        self.log = log
        self.store_single = store_single
        self.necessary_files = []  # fill
        self.out_file = ""  # fill
        self.software = []  # fill
        self.inter_temp_files = []  # fill
        self.key_help = ""  # fill

    def __str__(self):
        return f"{self.__class__.__name__} <necessary file: {self.necessary_files}, out file: {self.out_file}>"

    def __repr__(self):
        return f"{self.__class__.__name__} <necessary file: {self.necessary_files}, out file: {self.out_file}>"

    @staticmethod
    def _to_path(path: Union[os.PathLike, Path, pathlib.Path, str]) -> Path:
        if isinstance(path, str):
            path = path.replace("\n", "")
        return Path(path).abspath() if not isinstance(path, Path) else path

    def convert(self, path: Union[os.PathLike, Path, pathlib.Path, str]):
        """
        convert raw data in path.

        Parameters
        ----------
        path:path
            path.

        Returns
        -------
        data:optional(pd.Dataframe)
            data table.
        """
        self.check_software()
        path = self._to_path(path)
        path_bool = self.check_path_and_file(path)
        if path_bool:
            return self.wrapper(path)
        else:
            raise FileNotFoundError

    def transform(self, paths: List[Union[os.PathLike, Path, pathlib.Path, str]]):
        """
        transform raw data in each path containing in paths.

        Parameters
        ----------
        paths:list of path
            paths

        Returns
        -------
        data:pd.Dataframe
            data table.
        """
        self.check_software()
        paths = [self._to_path(pi) for pi in paths]
        paths_bool = [self.check_path_and_file(pi) for pi in paths]
        if self.n_jobs == 1:
            res_code = [self.wrapper(pi) if pib is True else None for pi, pib in zip(paths, paths_bool)]
        else:
            # trues_index = np.where(np.array(paths_bool))[0]
            trues = [i for i, b in zip(paths, paths_bool) if b]
            res_code = self.parallelize_imap(self.wrapper, iterable=list(trues), n_jobs=self.n_jobs, tq=self.tq)

        return self.batch_after_treatment(paths, res_code)

    fit_transform = transform

    @staticmethod
    def parallelize_imap(func: Callable, iterable: List, n_jobs: int = 1, tq: bool = True):
        """
        Parallelize the function for iterable.
        """
        pool = multiprocessing.Pool(processes=n_jobs)
        if tq:
            result_list_tqdm = [result for result in tqdm(pool.imap(func=func, iterable=iterable),
                                                          total=len(iterable))]
            pool.close()
        else:
            result_list_tqdm = [result for result in pool.imap(func=func, iterable=iterable)]
            pool.close()

        return result_list_tqdm

    def check_path_and_file(self, path: Path):
        """1.Check the path exist, and necessary file."""
        if not path.exists():
            return False
        else:
            if path.isdir():
                if all([True for i in self.necessary_files if (path / i).exists()]):
                    return True
                else:
                    print(f"Loss necessary files for {path}.")
                    return False
            elif path.isfile():
                warnings.warn("If path is one file, we think the necessary_files is just this file, "
                              "the ``necessary_files`` would be empty,"
                              "and with no follow-up necessary file inspection.")
                self.necessary_files = []
                return True
            else:
                return False

    def check_software(self):
        """2.Check software exist (just for linux). """
        for si in self.software:
            res = os.popen(f"which {si}").readlines()
            if len(res) == 0:
                raise NotImplementedError(f"Can't find the {si}, please:\n"
                                          f"1. Add this software to your PATH for system,\n"
                                          f"   such as, add 'export PATH=$PATH:~/bin/chgsum.pl' to '~/.bashrc' ;\n"
                                          f"2. keep access permission by such as, 'chmod u+x ~/bin/chgsum.pl' .")

    def wrapper(self, path: Path):
        """For exception."""
        if path.isfile():
            files = [path, ]  # 单个文件
        else:
            files = [path / i for i in self.necessary_files]

        old = os.getcwd()
        os.chdir(path)
        try:
            result = self.run(path, files)
            if result is None:
                print("No return for:", path)
                if self.log:
                    print(self.log_txt)
            else:
                print("Ok for:", path)
            os.chdir(old)
            return result
        except BaseException as e:
            print(e)
            print("Error for:", path)
            if self.log:
                print(self.log_txt)
            os.chdir(old)
            return None

    @property
    def log_txt(self):
        return f"""
    ##### HELP START for Method <{self.__class__.__name__}> #######"
    1.Check the input paths:
    
        If paths is offered by '-f' for 'Batching', please make sure the file such as 'paths.temp' exits and not empty.
        Generated one file by 'findpath' command in mgetool package is suggest. use 'findpath' command 
        directly now, to get all sub-folder in current path. Or use 'findpath -h' for more help.
        
        If one path is offered by '-p' for 'Single Case' (default, and in WORKPATH), please make sure the necessary files 
        under the path exists, as following.
    
    2.Check the necessary files, and software:
    {self.necessary_files}, {self.software}
    3.Check the key parameter of Method:
    {self.key_help}
    4.Check the intermediate temporary files are generated in each path:
    {self.inter_temp_files}
    
    Add '-h' from more examples.
    ##### HELP END #######"""

    @abstractmethod
    def run(self, path: Path, files: List = None) -> Any:  # fill
        """3.Run with software and necessary file and get data for one sample.
        (1) Return result in code, or (2) Both return file to each path and in code."""

        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # Sample
        # vasprun = Vasprun(path/"vasprun.xml")
        # ef = vasprun.efermi
        # return {"efermi":ef}

        return {"File": path}

    @abstractmethod
    def batch_after_treatment(self, paths, res_code):  # fill
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        # 合并多个文件到一个csv.

        # Sample
        # data_all = {pi:ri for pi,ri in zip(paths,res_code)}
        # result = pd.DataFrame(data_all).T
        # result.to_csv(self.out_file)
        # print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))

        return pd.DataFrame.from_dict({"File": paths}).T

    @staticmethod
    def extract(data, *args, format_path: Callable = None, **kwargs):
        """
        The last process! Extract the message in data, and formed it to
        be one table or ML or plot.

        Parameters
        ----------
        data:pd.DateFrame
            to transform data.
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
            if "File" in data:  # File are repetitive
                return data
            else:
                data["File"] = data["Unnamed: 0"]  # File are sole
                del data["Unnamed: 0"]
                data = data.set_index("File")
                data.index = [format_path(ci) for ci in data.index]
        else:
            data.index = [format_path(ci) for ci in data.index]
        return data


class _BasePathOut2(_BasePathOut):
    """
    class for deal with the software and files in batch, for couple data.

    The subclass should fill the follow properties and functions.

    self.necessary_files = []  # fill
    self.out_file = ""  # fill
    self.software = []  # fill
    self.inter_temp_files = []  # fill
    self.key_help = []  # fill

    def run(self, path: Path, files: List = None):
        ...  # fill

    def batch_after_treatment(self, paths, res_code):
        ...  # fill

    cls.__dos__ = "This is the doc of class."

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, log=True):
        super().__init__(n_jobs=n_jobs, tq=tq, store_single=store_single, log=log)

    def convert(self, path2: List[Union[os.PathLike, Path, pathlib.Path, str]]):
        self.check_software()
        path2 = [self._to_path(path) for path in path2]
        path_bool = all([self.check_path_and_file(path) for path in path2])
        if path_bool:
            return self.wrapper(path2)
        else:
            raise FileNotFoundError

    def transform(self, paths: List[List[Union[os.PathLike, Path, pathlib.Path, str]]]):
        self.check_software()
        paths = [[self._to_path(pi) for pi in path] for path in paths]
        paths_bool = [all([self.check_path_and_file(pi) for pi in path]) for path in paths]
        if self.n_jobs == 1:
            res_code = [self.wrapper(pi) if pib is True else None for pi, pib in zip(paths, paths_bool)]
        else:
            # trues_index = np.where(np.array(paths_bool))[0]
            trues = [i for i, b in zip(paths, paths_bool) if b]
            res_code = self.parallelize_imap(self.wrapper, iterable=list(trues), n_jobs=self.n_jobs, tq=self.tq)

        return self.batch_after_treatment(paths, res_code)

    def wrapper(self, paths: List[Path]):
        """For exception."""

        try:
            result = self.run(paths)
            if result is None:
                print("No return for:", paths)
                if self.log:
                    print(self.log_txt)
            else:
                print("Ok for:", paths)
            return result
        except BaseException as e:
            print(e)
            print("Error for:", paths)
            if self.log:
                print(self.log_txt)
            return None

    @abstractmethod
    def run(self, paths: List[Path], files: List = None) -> Any:  # fill
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""

        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # Sample
        # vasprun = Vasprun(path/"vasprun.xml")
        # ef = vasprun.efermi
        # return {"efermi":ef}

        return {"File": paths}

    @abstractmethod
    def batch_after_treatment(self, paths: List[List[Path]], res_code):  # fill
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        # 合并多个文件到一个csv.

        # Sample
        # data_all = {pi:ri for pi,ri in zip(paths,res_code)}
        # result = pd.DataFrame(data_all).T
        # result.to_csv(self.out_file)
        # print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))

        return pd.DataFrame.from_dict({"File": paths}).T
