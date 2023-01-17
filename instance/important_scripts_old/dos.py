# -*- coding: utf-8 -*-

# @Time  : 2022/6/3 22:42
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os.path
import pathlib

import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar, Vasprun


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


class Doscar:
    """
    Contains all the data in a VASP DOSCAR file, and methods for manipulating this.

    Examples
    -------------
    >>> doscar = Doscar("DOSCAR")


    """

    number_of_header_lines = 6

    def __init__(self, doscar="DOSCAR", poscar="POSCAR", vasprun="vasprun.xml", ispin=2, lmax=2,
                 lorbit=11, spin_orbit_coupling=False, read_pdos=True,max=8,min=-8):
        """
        Create a Doscar object from a VASP DOSCAR file.
        Args:
            doscar (str): Filename of the VASP DOSCAR file to read.
            ispin (optional:int): ISPIN flag. Set to 1 for non-spin-polarised or 2 for spin-polarised calculations. Default = 2.
            lmax (optional:int): Maximum l angular momentum. (d=2, f=3). Default = 2.
            lorbit (optional:int): The VASP LORBIT flag. (Default=11).
            spin_orbit_coupling (optional:bool): Spin-orbit coupling (Default=False).
            read_pdos (optional:bool): Set to True to read the atom-projected density of states (Default=True).
        """
        self.filename = doscar
        self.min=min
        self.max=max

        self.ispin = ispin
        self.lmax = lmax
        self.spin_orbit_coupling = spin_orbit_coupling
        if self.spin_orbit_coupling:
            raise NotImplementedError('Spin-orbit coupling is not yet implemented')
        self.lorbit = lorbit
        self.read_header()

        start_to_read = self.number_of_header_lines
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=['energy', 'up', 'down', 'integral_up', 'integral_down'],
                         index_col=False)

        if os.path.isfile(vasprun):
            vasprun = Vasprun(vasprun)
            self.efermi = vasprun.efermi
            df["energy"] = df["energy"]-self.efermi
            self.energy =  df.energy.values
            self.energy_name = "E-E_fermi"
        else:
            self.energy = df.energy.values
            self.energy_name = "E"

        df.drop('energy', axis=1)
        df.rename(columns={"energy":self.energy_name}, inplace=True)

        self.tdos = self.scale(df)

        if read_pdos:
            try:
                self.pdos_raw = self.read_projected_dos()
            except:
                self.pdos_raw=None
        # if species is set, should check that this is consistent with the number of entries in the
        # projected_dos dataset

        self.structure = Poscar.from_file(poscar, check_for_POTCAR=False).structure
        self.atoms_list = [i.name for i in self.structure.species]
        self.starts = [0, ]
        mark= self.atoms_list[0]
        for n,i in enumerate(self.atoms_list):
            if i==mark:
                pass
            else:
                self.starts.append(n)
            mark =i
        self.starts.append(len(self.atoms_list))


    @property
    def number_of_channels(self):
        if self.lorbit == 11:
            return {2: 9, 3: 16}[self.lmax]
        else:
            raise NotImplementedError

    def read_header(self):
        self.header = []
        with open(self.filename, 'r') as file_in:
            for i in range(self.number_of_header_lines):
                self.header.append(file_in.readline())
        self.process_header()

    def process_header(self):
        self.number_of_atoms = int(self.header[0].split()[0])
        self.number_of_data_points = int(self.header[5].split()[2])
        self.efermi = float(self.header[5].split()[3])

    def read_atomic_dos_as_df(self, atom_number):  # currently assume spin-polarised, no-SO-coupling, no f-states
        assert atom_number > 0 & atom_number <= self.number_of_atoms
        start_to_read = self.number_of_header_lines + atom_number * (self.number_of_data_points + 1)
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=pdos_column_names(lmax=self.lmax, ispin=self.ispin),
                         index_col=False)
        return df.drop('energy', axis=1)

    def scale(self,data):
        e = data.iloc[:,0].values
        mark1 = e>=self.min
        mark2 = e<=self.max
        mark = mark1*mark2
        data = data.iloc[mark,:]
        return data

    def read_projected_dos(self):
        """
        Read the projected density of states data into """
        pdos_list = []
        for i in range(self.number_of_atoms):
            df = self.read_atomic_dos_as_df(i + 1)
            pdos_list.append(df)

        return np.vstack([np.array(df) for df in pdos_list]).reshape(
            self.number_of_atoms, self.number_of_data_points, self.number_of_channels, self.ispin)

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

    def pdos_by_spdf_atom_num(self, spin=None, m=None):
        data = {self.energy_name:self.energy}
        if self.lmax==3:
            ls = ["s", "p", "d", "f"]
        else:
            ls= ["s", "p", "d"]

        atom_idx = list(range(self.number_of_atoms))
        for atoms in atom_idx:
            for l in ls:
                datai = {f"{atoms}-{l}": np.sum(self.pdos_select(atoms=(atoms,), spin=spin, l=l, m=m), axis=(0, 2, 3))}
                data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_spdf_atom_type(self, spin=None, m=None):
        data = {self.energy_name:self.energy}
        if self.lmax==3:
            ls = ["s", "p", "d", "f"]
        else:
            ls= ["s", "p", "d"]

        atom_num = len(self.starts)-1

        for ai in range(atom_num):
            atoms = list(range(self.starts[ai],self.starts[ai+1]))
            name_atoms = self.atoms_list[self.starts[ai]]
            for l in ls:
                datai = {f"{name_atoms}-{l}": np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))}
                data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_atom_type(self, spin=None, l=None, m=None):
        data = {self.energy_name:self.energy}

        atom_num = len(self.starts)-1

        for ai in range(atom_num):
            atoms = list(range(self.starts[ai],self.starts[ai+1]))
            name_atoms = self.atoms_list[self.starts[ai]]

            datai = {f"{name_atoms}-{l}": np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))}
            data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_atom_num(self, spin=None,l=None, m=None):
        data = {self.energy_name:self.energy}

        atom_num = len(self.starts)-1

        for ai in range(atom_num):
            atoms = list(range(self.starts[ai],self.starts[ai+1]))
            name_atoms = self.atoms_list[self.starts[ai]]

            datai = {f"{name_atoms}-{l}": np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))}
            data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

    def pdos_by_spdf(self, atoms=None, spin=None, m=None):
        data = {self.energy_name:self.energy}
        if self.lmax==3:
            ls = ["s", "p", "d", "f"]
        else:
            ls= ["s", "p", "d"]
        for l in ls:
            datai={l:np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))}
            data.update(datai)
        return self.scale(pd.DataFrame.from_dict(data))

