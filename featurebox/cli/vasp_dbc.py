# -*- coding: utf-8 -*-
import functools
import os
import warnings
from typing import List, Callable

# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import numpy as np
import pandas as pd
from path import Path
from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.io.vasp import Poscar, Vasprun
from scipy.integrate import simps

from featurebox.cli._basepathout import _BasePathOut

warnings.filterwarnings("ignore", category=RuntimeWarning)


class _Dosxyz:
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
            ispin (optional:int): ISPIN flag. Set to 1 for non-spin-polarised or 2 for spin-polarised calculations.
            Default = 2.
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


class DBCxyzPathOut(_BasePathOut):
    """Get d band center by code and return csv file."""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, method="atom"):
        super(DBCxyzPathOut, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["DOSCAR", "CONTCAR"]
        self.out_file = "dbc_xyz_all.csv"
        self.software = []
        self.key_help = "Make sure the 'LORBIT=11' in vasp INCAR file."
        self.method = method

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        dbc = _Dosxyz("DOSCAR", poscar="CONTCAR")
        if self.method == "ele":
            result_single = dbc.dbc_part_atom_type(path)
        else:
            result_single = dbc.dbc_part_atom_num(path)

        store_name = "dbc_xyz_single.csv"

        if self.store_single:
            result_single.to_csv(store_name)
            print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

        return result_single

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

    @staticmethod
    def extract(data: pd.DataFrame, atoms, orbit=None, format_path: Callable = None):
        """atoms start from 0."""
        if isinstance(atoms, (list, tuple)):
            if orbit is None:
                orbit = ['p-y', 'p-z', 'p-x', 'd-xy', 'd-yz', 'd-z2-r2', 'd-xz', 'd-x2-y2']
            if "File" not in orbit:
                orbit.append("File")
            if "Unnamed: 0" in data:
                del data["Unnamed: 0"]
            res = []
            if format_path is not None:
                data["File"] = [format_path(ci) for ci in data["File"]]
            for v in atoms:
                sel = data[data["Atom Number"] == v + 1]
                sel = sel[orbit].set_index("File")
                n_name = [f"{i}-{v}" for i in sel.columns]
                sel.columns = n_name
                res.append(sel)
            return pd.concat(res, axis=1)
        else:
            raise NotImplementedError("'atoms' just accept list or tuple.")


def get_atom_pdos_center(dos: CompleteDos = None, mark_orbital=None, mark_atom_numbers=None):
    # elements = [e.symbol for e in dos.structure.species]
    if mark_orbital is None:
        ns = [i for i in range(dos.structure.num_sites)]
    elif isinstance(mark_atom_numbers, int):
        ns = [mark_atom_numbers, ]
    else:
        ns = mark_atom_numbers

    # specify energy range of plot

    if mark_orbital is None:
        mark_orbital = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]
    else:
        kv = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d, "f": OrbitalType.f}
        mark_orbital = [kv[i] for i in mark_orbital]

    spd_dos_center_all = {}

    for mi in ns:
        ci = dos.structure.sites[mi]
        for oi in mark_orbital:
            try:
                dbc = dos.get_band_center(band=oi, sites=[ci])
                dbw = dos.get_band_width(band=oi, sites=[ci])
                dbf = dos.get_band_filling(band=oi, sites=[ci])
                spd_dos_center_all.update({f"Atom-{mi}-{oi}-band_center": dbc})
                spd_dos_center_all.update({f"Atom-{mi}-{oi}-band_width": dbw})
                spd_dos_center_all.update({f"Atom-{mi}-{oi}-band_filling": dbf})
            except BaseException as e:
                print(e)
                pass

    return spd_dos_center_all


def get_ele_pdos_center(dos: CompleteDos = None, mark_orbital=None, mark_element=None):
    elements = [s for s in dos.structure.symbol_set]
    if mark_element is None:
        mark_element = elements
    mark_element_num = [[n for n, j in enumerate(dos.structure.species) if i == j.symbol] for i in mark_element]
    # specify energy range of plot

    if mark_orbital is None:
        mark_orbital = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]
    else:
        kv = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d, "f": OrbitalType.f}
        mark_orbital = [kv[i] for i in mark_orbital]

    spd_dos_center_all = {}

    for rank_mi, mi in enumerate(mark_element_num):
        ci = [dos.structure.sites[i] for i in mi]
        for oi in mark_orbital:
            try:
                dbc = dos.get_band_center(band=oi, sites=ci)
                dbw = dos.get_band_width(band=oi, sites=ci)
                dbf = dos.get_band_filling(band=oi, sites=ci)
                spd_dos_center_all.update({f"{elements[rank_mi]}-{oi}-band_center": dbc})
                spd_dos_center_all.update({f"{elements[rank_mi]}-{oi}-band_width": dbw})
                spd_dos_center_all.update({f"{elements[rank_mi]}-{oi}-band_filling": dbf})
            except:
                pass

    return spd_dos_center_all


class DBCPy(_BasePathOut):
    """Get d band center by pymatgen and return csv file.
    pymatgen>=2022.5.26
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, method="ele"):
        super(DBCPy, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["vasprun.xml"]
        self.out_file = "dbc_py_all.csv"
        self.software = []
        self.method = method
        if self.method == "atom":
            self.extract = self._extract_atom
        else:
            self.extract = self._extract_ele

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        vasprun = Vasprun(path / "vasprun.xml")
        com_dos = vasprun.complete_dos
        if self.method == "atom":
            dbc = get_atom_pdos_center(com_dos)
        elif self.method == "ele":
            dbc = get_ele_pdos_center(com_dos)
        else:
            raise NotImplementedError('method ="atom" or "ele".')

        result_single = pd.DataFrame.from_dict({path: dbc}).T

        if self.store_single:
            result_single.to_csv("dbc_py_single.csv")
            print("'{}' are sored in '{}'".format("dbc_py_single.csv", os.getcwd()))

        return result_single

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""

        result = pd.concat(res_code, axis=0)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result

    @staticmethod
    def _extract_ele(data: pd.DataFrame, ele_and_orbit=None, join_ele=None, join_ele_orbit=None, format_path=None):
        """

        >>> self.extract(res, ele_and_orbit=["C-p", "O-p", "Mo-d", "JTM-d"], join_ele=["Ag","Au",...], )

        Parameters
        ----------
        data:pd.Dateframe
            data.
        ele_and_orbit:
            atom_orbit.
        join_ele:list
            atom name.
        join_ele_orbit:
            orbit for join ele.
        format_path: Callable
            function.
        """
        import re

        @functools.lru_cache(4)
        def col_corresponding_index(sel_name, index_all):
            sel_index = []
            for i in sel_name:
                xx = [j for j in index_all if i in re.split(r" |-|/|\\", j)]
                assert len(xx) == 1, f"Find {len(xx)} index for {i}, can't decide which one. {xx}."
                sel_index.append(xx[0])
            return sel_index

        res = []
        if "Unnamed: 0" in data:
            data["File"] = data["Unnamed: 0"]
            del data["Unnamed: 0"]
            data = data.set_index("File")

        if format_path is not None:
            data.index = [format_path(ci) for ci in data.index]

        if isinstance(ele_and_orbit, (list, tuple)):
            for v in ele_and_orbit:
                sel = [i for i in data.columns if f"{v}" in i]
                sel = data[sel]
                res.append(sel)
        if isinstance(join_ele, (list, tuple)):
            if join_ele_orbit is None:
                join_ele_orbit = ["s", "p", "d"]
            builtin_name = [f"{i}-{j}" for i in join_ele_orbit for j in ["band_center", "band_filling", "band_width"]]

            for bi in builtin_name:
                names = [f"{i}-{bi}" for i in join_ele]
                sel_col = list(set(data.columns) & set(names))
                sel_col.sort()
                sel_name = [i.split("-")[0] for i in sel_col]
                sel_index = col_corresponding_index(tuple(sel_name), tuple(data.index))

                sel = {i: data.loc[i, c] for i, c in zip(sel_index, sel_col)}

                df = pd.DataFrame.from_dict({f"JTM-{bi}": sel})

                res.append(df)

        if len(res) == 0:
            raise NotImplementedError("'names' and  'join_names' are at least to be set one or more.")

        return pd.concat(res, axis=1)

    @staticmethod
    def _extract_atom(data: pd.DataFrame, atoms, orbit=None, format_path: Callable = None):
        """atoms start from 0."""
        print("The extract_atom function just accept table result from method='atom'.")
        if orbit is None:
            orbit = ["s", "p", "d"]
        if isinstance(atoms, (list, tuple)):
            res = []
            if "Unnamed: 0" in data:
                data["File"] = data["Unnamed: 0"]
                del data["Unnamed: 0"]
                data = data.set_index("File")

            if format_path is not None:
                data.index = [format_path(ci) for ci in data.index]

            for v in atoms:
                sel = [i for i in data.columns for k in orbit if f"-{v}-{k}-" in i]
                sel = data[sel]
                res.append(sel)
            return pd.concat(res, axis=1)
        else:
            raise NotImplementedError("'atoms' just accept list or tuple.")


class DBCStartZero(_BasePathOut):
    """Get d band center from paths and return csv file.
    VASPKIT Version: 1.2.1  or below.
    Download vaspkit from
    https://vaspkit.com/installation.html#download
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(DBCStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["INCAR", "DOSCAR", "CONTCAR"]
        self.out_file = "dbc_all.csv"
        self.software = ["vaspkit"]
        self.key_help = "Make sure the 'LORBIT=11' in vasp INCAR file."

    @staticmethod
    def read(d, store=False, store_name="dbc_single.csv"):
        """Run linux cmd and return result, make sure the vaspkit is installed."""
        result_name = d / "D_BAND_CENTER"
        with open(result_name, mode="r") as f:
            ress = f.readlines()

        n = 0
        res = []
        for i in ress:
            if i == "\n":
                break
            else:
                i = i.split(" ")
                i = [i for i in i if i != ""]
                i = [i[:-2] if "\n" in i else i for i in i]
                i.insert(0, str(n))
                res.append(i)
                n += 1

        res = np.array(res[1:])
        res = np.concatenate((np.full(res.shape[0], fill_value=str(d)).reshape(-1, 1), res), axis=1)
        result = pd.DataFrame(res,
                              columns=["File", "Atom Number", "Atom", "d-Band-Center (UP)", "d-Band-Center (DOWN)",
                                       "d-Band-Center (Average)"])

        if store:
            result.to_csv(store_name)
            print("'{}' are sored in '{}'".format(store_name, os.getcwd()))
        return result

    @staticmethod
    def extract(data: pd.DataFrame, atoms=None, ele=None, join_ele=None, format_path: Callable = None):
        """atoms start from 1.
        This atom number are different to the structure atom number.

        >>> self.extract(res, ele=["O1","C1",...], join_ele=[f'{i}1' if i !='Mo' else 'Mo18' for i in self.doping], )

        """
        ele = [] if join_ele is not None and ele is None else ele
        data = data.apply(pd.to_numeric, errors='ignore')
        if format_path is not None:
            data["File"] = [format_path(ci) for ci in data["File"]]
        res = []
        if isinstance(atoms, (list, tuple)):
            print("This atom numbers in 'atoms' parameter are different to the structure atom numbers,"
                  "rather than the rank of number in bader file 'D_BANF_CENTER' (strat from 1)."
                  "Check you file and make sure the number are correct."
                  "Check you file and make sure the number are correct.")
            print("We suggest use the 'names' parameter to extarct data such as: ['Mo1','Mo2'].")

            # index = [re.sub("\D", "", i) for i in data["Atom"].values]
            # data["Atom"] = [int(i) if i!="" else None for i in index]
            for v in atoms:
                sel = data[data["Atom Number"] == v]
                sel = sel[["File", "d-Band-Center (Average)"]].set_index("File")
                n_name = [f"{i}-{v}" for i in sel.columns]
                sel.columns = n_name
                res.append(sel)
            return pd.concat(res, axis=1)
        elif isinstance(ele, (list, tuple)):

            data["Atom"] = [i.replace(":", "") for i in data["Atom"].values]

            for v in ele:
                sel = data[data["Atom"] == v]
                sel = sel[["File", "d-Band-Center (Average)"]].set_index("File")
                n_name = [f"{i}-{v}" for i in sel.columns]
                sel.columns = n_name
                res.append(sel)

            js = []
            for je in join_ele:
                sel1 = data[data["Atom"] == je]
                sel1 = sel1[["File", "d-Band-Center (Average)"]].set_index("File")
                assert sel1.values.shape[0] <= 1, "join element must be sole."
                js.append(sel1)

            df = pd.concat(js, axis=0)
            n_name = [f"JTM-{i}" for i in df.columns]
            df.columns = n_name
            res.append(df)

            return pd.concat(res, axis=1)
        else:
            raise NotImplementedError("'atoms' just accept list or tuple.")

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # 1.外部软件
        cmds = ("vaspkit -task 503",)
        for i in cmds:
            os.system(i)

        return self.read(path, store=self.store_single, store_name="dbc_single.csv")

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


class DBCStartInter(DBCStartZero):
    """
    For some system can't run this DBCStartZero.

    Download vaspkit from
    https://vaspkit.com/installation.html#download

    1. Copy follow code to form one 'dbc.sh' file, and run 'sh dbc.sh':

    Notes::

        #!/bin/bash

        old_path=$(cd "$(dirname "$0")"; pwd)

        for i in $(cat paths.temp)

        do

        cd $i

        echo $(cd "$(dirname "$0")"; pwd)

        vaspkit -task 503

        cd $old_path

        done

    2. tar -zcvf data.tar.gz D_BAND_CENTER.

    3. Move to other system and run  'tar -zxvf data.tar.gz'.

    4. Run with this class.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(DBCStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["D_BAND_CANTER"]
        self.out_file = "dbc_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        return self.read(path, store=self.store_single, store_name="dbc_single.csv")


class DBCStartSingleResult(DBCStartZero):
    """Avoid Double Calculation. Just reproduce the 'results_all' from a 'result_single' files.
    keeping the 'result_single.csv' files exists.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(DBCStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["dbc_single.csv"]
        self.out_file = "dbc_all.csv"
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
    批量提取 band center，保存到当前工作文件夹。 查看参数帮助使用 -h。

    Notes:

        1.1 前期准备
        INCAR, CONTCAR, DOSCAR

        2.运行文件要求:
        vaspkit <= 1.2.1, for -j in (0,1,2)

    -j 参数说明：

        0              1              2                     3              4
        [DBCStartZero, DBCStartInter, DBCStartSingleResult, DBCxyzPathOut, DBCPy]

        0: 调用vaspkit软件运行。（首次启动）
        1: 调用vaspkit软件分析结果运行。（热启动）
        2: 调用单个cohp_single.csv运行。（热启动）
        3: 调用此python代码运行。
        4: 调用pymatgen运行。(pymatgen>=2022.5.26)

    补充:

        在 featurebox 中运行，请使用 featurebox dbc ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox dbc -f /home/sdfa/paths.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox dbc -p /home/parent_dir/***/sample_dir
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=4)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        methods = [DBCStartZero, DBCStartInter, DBCStartSingleResult, DBCxyzPathOut, DBCPy]
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
