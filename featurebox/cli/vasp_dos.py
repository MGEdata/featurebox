# -*- coding: utf-8 -*-

# @Time  : 2022/7/27 14:05
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import os
from typing import List

import numpy as np
import pandas as pd
from path import Path
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.io.vasp import Poscar, Vasprun
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d

from featurebox.cli._basepathout import _BasePathOut


class Dosxyz:
    """
    Read DOSCAR and get pdos and band centor.

    """

    number_of_header_lines = 6

    def __init__(self, doscar="DOSCAR", poscar="CONTCAR", vasprun="vasprun.xml",
                 ispin=2, lmax=2, lorbit=11, spin_orbit_coupling=False, read_pdos=True, max=10, min=-10):
        """
        Create a Doscar object from a VASP DOSCAR file.

        Args:
            doscar (str): Filename of the VASP DOSCAR file to read.
            poscar (str): File POSCAR/CONTCAR
            vasprun (str): File vasprun.xml
            ispin (optional:int): ISPIN flag. Set to 1 for non-spin-polarised or 2 for spin-polarised calculations. (Default = 2.)
            lmax (optional:int): Maximum l angular momentum. (d=2, f=3). Default = 2.
            lorbit (optional:int): The VASP LORBIT flag. (Default=11).
            max (optional:int): max value
            min (optional:int): min value
            spin_orbit_coupling (optional:bool): Spin-orbit coupling (Default=False).
            read_pdos (optional:bool): Set to True to read the atom-projected density of states (Default=True).
        """
        self.doscar = doscar
        self.poscar = poscar
        self.vasprun = vasprun

        self.min = min
        self.max = max

        self.ispin = ispin
        self.lmax = lmax

        if self.lmax == 3:
            self.ls = ["s", "p", "d", "f"]
        else:
            self.ls = ["s", "p", "d"]

        self.spin_orbit_coupling = spin_orbit_coupling
        if self.spin_orbit_coupling:
            raise NotImplementedError('Spin-orbit coupling is not yet implemented')

        self.lorbit = lorbit
        self.read_header()

        start_to_read = self.number_of_header_lines

        df = pd.read_csv(self.doscar,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=['energy', 'up', 'down', 'integral_up', 'integral_down'],
                         index_col=False)

        df["energy"] = df["energy"] - self.efermi
        self.energy = df.energy.values
        self.energy_name = "E-E_fermi"

        df.drop('energy', axis=1)

        df.rename(columns={"energy": self.energy_name}, inplace=True)

        self.tdos = self.scale(df)

        if read_pdos:
            try:
                self.pdos_raw = self.read_projected_dos()
            except:
                self.pdos_raw = None

        self.structure = Poscar.from_file(poscar, check_for_POTCAR=False).structure

        self.atoms_list = [i.name for i in self.structure.species]

        assert self.number_of_atoms == len(self.atoms_list), "The POSCAR and DOSCAR are not matching."

        self.ntype_elements = [s for s in self.structure.symbol_set]

        self.atoms_group = [[n for n, j in enumerate(self.structure.species) if i == j.symbol] for i in
                            self.ntype_elements]

    def scale(self, data) -> pd.DataFrame:
        e = data.iloc[:, 0].values
        mark1 = e >= self.min
        mark2 = e <= self.max
        mark = mark1 * mark2
        data = data.iloc[mark, :]
        return data

    @property
    def number_of_channels(self):
        if self.lorbit == 11:
            return {2: 9, 3: 16}[self.lmax]
        else:
            raise NotImplementedError

    def read_header(self):
        self.header = []
        with open(self.doscar, 'r') as file_in:
            for i in range(self.number_of_header_lines):
                self.header.append(file_in.readline())
        self.process_header()

    def process_header(self):
        self.number_of_atoms = int(self.header[0].split()[0])
        self.number_of_data_points = int(self.header[5].split()[2])
        self.efermi = float(self.header[5].split()[3])

    @staticmethod
    def pdos_column_names(lmax, ispin):
        if lmax == 2:
            names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2']
            # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2' ]
        elif lmax == 3:
            names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2',
                     'f_y(3x2-y2)', 'f_xyz', 'f_yz2', 'f_z3', 'f_xz2', 'f_z(x2-y2)', 'f_x(x2-3y2)']
            # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2','f-3',
            #         'f-2', 'f-1', 'f0', 'f1', 'f2', 'f3']
        else:
            raise ValueError('lmax value not supported')

        if ispin == 2:
            all_names = []
            for n in names:
                all_names.extend(['{}_up'.format(n), '{}_down'.format(n)])
        else:
            all_names = names
        all_names.insert(0, 'energy')
        return all_names

    def read_atomic_dos_as_df(self, atom_number):  # currently assume spin-polarised, no-SO-coupling, no f-states
        assert atom_number > 0 & atom_number <= self.number_of_atoms
        start_to_read = self.number_of_header_lines + atom_number * (self.number_of_data_points + 1)
        df = pd.read_csv(self.doscar,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=self.pdos_column_names(lmax=self.lmax, ispin=self.ispin),
                         index_col=False)
        return df.drop('energy', axis=1)

    def read_projected_dos(self):
        """
        Read the projected density of states data into """
        pdos_list = []
        for i in range(self.number_of_atoms):
            df = self.read_atomic_dos_as_df(i + 1)
            pdos_list.append(df)

        return np.vstack([np.array(df) for df in pdos_list]).reshape((
            self.number_of_atoms, self.number_of_data_points, self.number_of_channels, self.ispin))

    def pdos_select(self, atoms=None, spin=None, l=None, m=None):
        """
        Returns a subset of the projected density of states array.
        """
        valid_m_values = {'s': [],
                          'p': ['x', 'y', 'z'],
                          'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2'],
                          'f': ['y(3x2-y2)', 'xyz', 'yz2', 'z3', 'xz2', 'z(x2-y2)', 'x(x2-3y2)']}
        if not atoms:
            atom_idx = list(range(self.number_of_atoms))
        else:
            atom_idx = atoms
        to_return = self.pdos_raw[atom_idx, :, :, :]
        if not spin:
            spin_idx = list(range(self.ispin))
        elif spin == 'up':
            spin_idx = [0]
        elif spin == 'down':
            spin_idx = [1]
        elif spin == 'both':
            spin_idx = [0, 1]
        else:
            raise ValueError
        to_return = to_return[:, :, :, spin_idx]

        if not l:
            channel_idx = list(range(self.number_of_channels))
        elif l == 's':
            channel_idx = [0]
        elif l == 'p':
            if not m:
                channel_idx = [1, 2, 3]
            else:
                channel_idx = [1 + i for i, v in enumerate(valid_m_values['p']) if v in m]
        elif l == 'd':
            if not m:
                channel_idx = [4, 5, 6, 7, 8]
            else:
                channel_idx = [4 + i for i, v in enumerate(valid_m_values['d']) if v in m]
        elif l == 'f':
            if not m:
                channel_idx = [9, 10, 11, 12, 13, 14, 15]
            else:
                channel_idx = [9 + i for i, v in enumerate(valid_m_values['f']) if v in m]
        else:
            raise ValueError

        return to_return[:, :, channel_idx, :]

    def pdos_sum(self, atoms=None, spin=None, l=None, m=None):
        return np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))

    def pdos_by_spdf_atom_num(self, spin=None, l=None):
        data = {self.energy_name: self.energy}

        if isinstance(l, str):
            ls = [l, ]
        elif isinstance(l, list):
            ls = l
        else:
            ls = self.ls

        for atoms in range(self.number_of_atoms):
            for l in ls:
                datai = {f"{self.atoms_list[atoms]}-{atoms}-{l}":
                             np.sum(self.pdos_select(atoms=(atoms,), spin=spin, l=l), axis=(0, 2, 3))}
                data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_atom_num(self, spin=None):
        data = {self.energy_name: self.energy}

        for atoms in range(self.number_of_atoms):
            datai = {f"{self.atoms_list[atoms]}-{atoms}":
                         np.sum(self.pdos_select(atoms=(atoms,), spin=spin), axis=(0, 2, 3))}
            data.update(datai)

        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_spdf_atom_type(self, spin=None, l=None):
        data = {self.energy_name: self.energy}
        if isinstance(l, str):
            ls = [l, ]
        elif isinstance(l, list):
            ls = l
        else:
            ls = self.ls

        for i, atoms in enumerate(self.atoms_group):
            for l in ls:
                datai = {f"{self.ntype_elements[i]}-{l}":
                             np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l), axis=(0, 2, 3))}
                data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_atom_type(self, spin=None):
        data = {self.energy_name: self.energy}
        for i, atoms in enumerate(self.atoms_group):
            datai = {f"{self.ntype_elements[i]}":
                         np.sum(self.pdos_select(atoms=atoms, spin=spin), axis=(0, 2, 3))}
            data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_spdf(self, atoms=None, spin=None, l=None):
        data = {self.energy_name: self.energy}
        if isinstance(l, str):
            ls = [l, ]
        elif isinstance(l, list):
            ls = l
        else:
            ls = self.ls

        for l in ls:
            datai = {l: np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l), axis=(0, 2, 3))}
            data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_spdf_xyz_atom_num(self, path=None):
        tab = {'p': ['x', 'y', 'z'], 'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2']}
        data_all = {}
        for atoms in range(self.number_of_atoms):
            for l in ["p", "d"]:

                m = tab[l]

                for mii in m:
                    datai = {
                        f"{path}-{self.energy_name}": self.energy,
                        f"{path}-{self.atoms_list[atoms]}-{atoms}-{l}-{mii}":
                            np.sum(self.pdos_select(atoms=(atoms,), l=l, m=mii), axis=(0, 2, 3))}
                    data_all.update(datai)

        return self.scale(pd.DataFrame.from_dict(data_all))

    def pdos_by_spdf_xyz_atom_type(self, path=None):
        tab = {'p': ['x', 'y', 'z'], 'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2']}
        data_all = {}
        for i, atoms in enumerate(self.atoms_group):
            for l in ["p", "d"]:
                m = tab[l]
                for mii in m:
                    datai = {
                        f"{path}-{self.energy_name}": self.energy,
                        f"{path}-{self.ntype_elements[i]}-{l}-{mii}":
                            np.sum(self.pdos_select(atoms=atoms, l=l, m=mii), axis=(0, 2, 3))}
                    data_all.update(datai)

        return self.scale(pd.DataFrame.from_dict(data_all))

    def calculate(self, orb="d", species: List[str] = None, atoms: List[int] = None, emax=2, emin=-10, m=None):
        """species (optional:list(str)): List of atomic species strings, e.g. [ 'Fe', 'Fe', 'O', 'O', 'O' ].
        Default=None."""

        if species is None and atoms is None:
            atoms = list(range(0, self.number_of_atoms))

        elif species is not None and atoms is None:
            atoms = [idx for idx, j in enumerate(self.atoms_list) if j in species]
        elif species is None and atoms is not None:
            pass
        else:
            print("Species, atoms are assigned at the same time is not recommended.")
            atoms = [idx for idx in atoms if self.atoms_list[idx] in species]

        assert len(atoms) > 0
        # calculation of d-band center
        # Set atoms for integration
        up = self.pdos_sum(atoms, spin='up', l=orb, m=m)
        down = self.pdos_sum(atoms, spin='down', l=orb, m=m)

        # Set integrating range
        energies = self.energy - self.efermi

        erange = (self.efermi + emin, self.efermi + emax)  # integral energy range
        emask = (energies <= erange[-1])

        # Calculating center of the orbital specified above in line 184
        x = energies[emask]
        y1 = up[emask]
        y2 = down[emask]
        dbc_up = simps(y1 * x, x) / simps(y1, x)
        dbc_down = simps(y2 * x, x) / simps(y2, x)
        dbc_all = [dbc_up, dbc_down]
        return dbc_all

    def dbc_part_atom_num(self, path=None):
        data_all = {}
        for ii in range(self.number_of_atoms):
            data = {"File": path, "Atom Number": ii}
            for k in ["p", "d"]:
                tab = {'p': ['x', 'y', 'z'], 'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2']}
                m = tab[k]

                for mii in m:
                    d_band_center_up_and_down = self.calculate(orb=k,
                                                               atoms=[ii, ],
                                                               m=mii
                                                               )
                    dbc_value = sum(d_band_center_up_and_down) / 2
                    data.update({f"{k}-{mii}": dbc_value})
            data_all.update({ii: data})
        return pd.DataFrame.from_dict(data_all).T

    def dbc_part_atom_type(self, path=None):
        data_all = {}
        for i, ii in enumerate(self.atoms_group):
            data = {"File": path, "Atom type": self.ntype_elements[i]}
            for k in ["p", "d"]:
                tab = {'p': ['x', 'y', 'z'], 'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2']}

                m = tab[k]

                for mii in m:
                    d_band_center_up_and_down = self.calculate(orb=k,
                                                               atoms=ii,
                                                               m=mii
                                                               )
                    dbc_value = sum(d_band_center_up_and_down) / 2
                    data.update({f"{k}-{mii}": dbc_value})
            data_all.update({self.ntype_elements[i]: data})
        return pd.DataFrame.from_dict(data_all).T


class DosxyzPathOut(_BasePathOut):
    """Get dos from paths and return csv file."""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, method="ele"):
        super(DosxyzPathOut, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["vasprun.xml", "DOSCAR", "CONTCAR"]
        self.out_file = "dos_xyz_py_all.csv"
        self.software = []
        self.method = method
        self.extract = None

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        dos = Dosxyz(doscar="DOSCAR", poscar="CONTCAR", vasprun="vasprun.xml", ispin=2, lmax=2,
                     lorbit=11, read_pdos=True, max=10, min=-10)

        if self.method == "ele":
            result_single = dos.pdos_by_spdf_xyz_atom_type(path)
        else:
            result_single = dos.pdos_by_spdf_xyz_atom_num(path)
        store_name = "dos_xyz_single.csv"

        if self.store_single:
            result_single.to_csv(store_name)
            print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

        return result_single.to_dict()

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = {}
        for res in res_code:
            if isinstance(res, dict):
                data_all.update(res)

        result = pd.DataFrame.from_dict(data_all)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


def get_ele_pdos(dos: CompleteDos = None, mark_orbital=None, mark_element=None, sigma=0.1, path=None):
    elements = [e.symbol for e in dos.structure.composition.elements]
    if mark_element is None:
        mark_element = elements
    mark_element_num = [elements.index(i) for i in mark_element]
    # specify energy range of plot

    # set up bs and dos plot

    dos_energies = [e - dos.efermi for e in dos.energies]

    if mark_orbital is None:
        mark_orbital = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]
    else:
        kv = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d, "f": OrbitalType.f}
        mark_orbital = [kv[i] for i in mark_orbital]

    spd_dos = [dos.get_element_spd_dos(_) for _ in dos.structure.composition]

    spd_dos_res = {f"{path}-PDOS": dos_energies,
                   f"{path}-total": dos.densities[Spin.up] + dos.densities[Spin.down]}

    for i in mark_element_num:
        spd_dosi = spd_dos[i]
        for ii in spd_dosi.keys():
            if ii in mark_orbital:
                spd_dos_res.update(
                    {f"{path}-{elements[i]}-{ii}-PDOS": spd_dosi[ii].densities[Spin.up] + spd_dosi[ii].densities[
                        Spin.down]})
    if sigma == 0:
        pass
    else:
        spd_dos_res = {k: gaussian_filter1d(v, sigma) for k, v in spd_dos_res.items()}

    return spd_dos_res


def get_atom_pdos(dos: CompleteDos = None, mark_orbital=None, mark_atom_numbers=None, sigma=0.1, path=None):
    elements = [e.symbol for e in dos.structure.species]
    if mark_orbital is None:
        ns = [i for i in range(dos.structure.num_sites)]
    elif isinstance(mark_atom_numbers, int):
        ns = [mark_atom_numbers, ]
    else:
        ns = mark_atom_numbers

    # set up bs and dos plot

    dos_energies = [e - dos.efermi for e in dos.energies]

    if mark_orbital is None:
        mark_orbital = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]
    else:
        kv = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d, "f": OrbitalType.f}
        mark_orbital = [kv[i] for i in mark_orbital]

    spd_dos = [dos.get_element_spd_dos(_) for _ in dos.structure.species]

    spd_dos_res = {f"{path}-PDOS": dos_energies,
                   f"{path}-total": dos.densities[Spin.up] + dos.densities[Spin.down]}

    for i in ns:
        spd_dosi = spd_dos[i]
        for ii in spd_dosi.keys():
            if ii in mark_orbital:
                spd_dos_res.update(
                    {f"{path}-{elements[i]}-{i}-{ii}-PDOS": spd_dosi[ii].densities[Spin.up] + spd_dosi[ii].densities[
                        Spin.down]})
    if sigma == 0:
        pass
    else:
        spd_dos_res = {k: gaussian_filter1d(v, sigma) for k, v in spd_dos_res.items()}

    return spd_dos_res


class DosPy(_BasePathOut):
    """Get d band center from paths and return csv file."""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, method="ele"):
        super(DosPy, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["vasprun.xml"]
        self.out_file = "dos_py_all.csv"
        self.software = []
        self.method = method
        self.extract = None

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        vasprun = Vasprun(path / "vasprun.xml", parse_potcar_file=False)
        dos_data = vasprun.complete_dos
        # set figure parameters, draw figure
        if self.method == "ele":
            spd_dos = get_ele_pdos(dos=dos_data, sigma=0, path=path)
        else:
            spd_dos = get_atom_pdos(dos=dos_data, sigma=0, path=path)

        result_single = pd.DataFrame.from_dict(spd_dos)

        if self.store_single:
            result_single.to_csv("dos_py_single.csv")
            print("'{}' are sored in '{}'".format("dos_py_single.csv", os.getcwd()))

        return spd_dos

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = {}
        for res in res_code:
            if isinstance(res, dict):
                data_all.update(res)

        result = pd.DataFrame.from_dict(data_all)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


class _CLICommand:
    """
    批量提取 DOS，保存到当前工作文件夹。 查看参数帮助使用 -h。

    Notes:

        1.1 前期准备
        INCAR, CONTCAR, DOSCAR

        1.2 INCAR准备
        INCAR文件参数要求：
        LORBIT=11
        NSW = 0
        IBRION = -1

        2.运行文件要求:
        vaspkit <= 1.2.1, for -j in (0,1)

    -j 参数说明：

        0       1
        [DosPy, DosxyzPathOut]

        0: 调用pymatgen运行。
        1: 调用此python代码运行。

    补充:

        在 featurebox 中运行，请使用 featurebox dos ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox dos -f /home/sdfa/paths.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox dos -p /home/parent_dir/***/sample_dir
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
        methods = [DosPy, DosxyzPathOut]
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
