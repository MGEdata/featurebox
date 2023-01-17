from math import tan, pi
from typing import List
from warnings import warn

from featurebox.featurizers.base_feature import BaseFeature

try:
    import kpcpu
except ImportError:
    kpcpu = None

from numpy import pi, dot

import torch
from pymatgen.core import PeriodicSite, Structure

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.util.num import abs_cap
import itertools

import math
from math import cos
from math import sin

import numpy as np

from pymatgen.core.lattice import Lattice


def get_lengths(matrix):
    """Return The lengths (a, b, c) of the lattice."""
    return tuple(np.sqrt(np.sum(matrix ** 2, axis=1)).tolist())  # type: ignore


def get_angles(matrix, lengths=None):
    """Returns the angles (alpha, beta, gamma) of the lattice."""
    if lengths is not None:
        lengths = get_lengths(matrix)
    m = matrix
    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(dot(m[j], m[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / pi
    return tuple(angles.tolist())  # type: ignore


def reciprocal_lattice_angle(matrix):
    """Return the angle of reciprocal lattice."""
    new_matrix = np.linalg.inv(matrix).T * 2 * np.pi
    return get_angles(new_matrix)


ERROE_SPACE = np.array([-1, 0.0, 0.0, 0.0, 0.0, 0.0])


class SpacegroupAnalyzerUser(SpacegroupAnalyzer):
    """A copy from pymatgen. The change is for quick return the preprocessing:
    [type_number,a, b, c, alpha_prim, alpha_conv]

    The result of SpacegroupAnalyzerUser and input of HighSymPoints are matched.
    """

    def __init__(self, structure, symprec=0.01, angle_tolerance=5.0):
        """

        Args:
            structure: (Structure)
            symprec: (float)
            angle_tolerance:(float)
        """
        super().__init__(structure, symprec, angle_tolerance)
        self.lattice_type = self.get_lattice_type()
        self.crystal_system = self.get_crystal_system()
        self.spg_symbol = self.get_space_group_symbol()
        self.struct = self.get_refined_structure()

    def get_conventional_to_primitive_transformation_matrix(self, international_monoclinic=True):
        """
        Gives the transformation matrix to transform a conventional
        unit cell to a primitive cell according to certain standards
        the standards are defined in Setyawan, W., & Curtarolo, S. (2010).
        High-throughput electronic band structure calculations:
        Challenges and tools. Computational Materials Science,
        49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010

        Returns:
            Transformation matrix to go from conventional to primitive cell
        """
        if hasattr(self, "_conv"):
            conv = self._conv
        else:
            conv = self.get_conventional_standard_structure(
                international_monoclinic=international_monoclinic)

        lattice = self.lattice_type

        if "P" in self.spg_symbol or lattice == "hexagonal":
            return np.eye(3)

        if lattice == "rhombohedral":
            # check if the conventional representation is hexagonal or
            # rhombohedral
            lengths = conv.lattice.lengths
            if abs(lengths[0] - lengths[2]) < 0.0001:
                transf = np.eye
            else:
                transf = np.array([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]],
                                  dtype=np.float) / 3

        elif "I" in self.spg_symbol:
            transf = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                              dtype=np.float) / 2
        elif "F" in self.spg_symbol:
            transf = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                              dtype=np.float) / 2
        elif "C" in self.spg_symbol or "A" in self.spg_symbol:
            if self.crystal_system == "monoclinic":
                transf = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]],
                                  dtype=np.float) / 2
            else:
                transf = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]],
                                  dtype=np.float) / 2
        else:
            transf = np.eye(3)

        return transf

    def get_primitive_standard_structure(self, international_monoclinic=True):
        """
        Gives a structure with a primitive cell according to certain standards
        the standards are defined in Setyawan, W., & Curtarolo, S. (2010).
        High-throughput electronic band structure calculations:
        Challenges and tools. Computational Materials Science,
        49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010

        Returns:
            The structure in a primitive standardized cell
        """
        if hasattr(self, "_conv"):
            conv = self._conv
        else:
            conv = self.get_conventional_standard_structure(
                international_monoclinic=international_monoclinic)
        lattice = self.lattice_type

        if "P" in self.spg_symbol or lattice == "hexagonal":
            return conv

        transf = self.get_conventional_to_primitive_transformation_matrix(
            international_monoclinic=international_monoclinic)

        new_sites = []
        latt = Lattice(np.dot(transf, conv.lattice.matrix))
        for s in conv:
            new_s = PeriodicSite(
                s.specie, s.coords, latt,
                to_unit_cell=True, coords_are_cartesian=True,
                properties=s.properties)
            if not any(map(new_s.is_periodic_image, new_sites)):
                new_sites.append(new_s)

        if lattice == "rhombohedral":
            prim = Structure.from_sites(new_sites)
            lengths = prim.lattice.lengths
            angles = prim.lattice.angles
            a = lengths[0]
            alpha = math.pi * angles[0] / 180
            new_matrix = [
                [a * cos(alpha / 2), -a * sin(alpha / 2), 0],
                [a * cos(alpha / 2), a * sin(alpha / 2), 0],
                [a * cos(alpha) / cos(alpha / 2), 0,
                 a * math.sqrt(1 - (cos(alpha) ** 2 / (cos(alpha / 2) ** 2)))]]
            new_sites = []
            latt = Lattice(new_matrix)
            for s in prim:
                new_s = PeriodicSite(
                    s.specie, s.frac_coords, latt,
                    to_unit_cell=True, properties=s.properties)
                if not any(map(new_s.is_periodic_image, new_sites)):
                    new_sites.append(new_s)
            return Structure.from_sites(new_sites)

        return Structure.from_sites(new_sites)

    def get_conventional_standard_structure(self, international_monoclinic=True):
        """
        Gives a structure with a conventional cell according to certain
        standards. The standards are defined in Setyawan, W., & Curtarolo,
        S. (2010). High-throughput electronic band structure calculations:
        Challenges and tools. Computational Materials Science,
        49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010
        They basically enforce as much as possible
        norm(a1)<norm(a2)<norm(a3)

        Returns:
            The structure in a conventional standardized cell
        """
        tol = 1e-5
        struct = self.struct
        latt = self.struct.lattice
        latt_type = self.lattice_type
        sorted_lengths = sorted(latt.abc)
        sorted_dic = sorted([{'vec': latt.matrix[i],
                              'length': latt.abc[i],
                              'orig_idx': i} for i in [0, 1, 2]],
                            key=lambda k: k['length'])

        if latt_type in ("orthorhombic", "cubic"):
            # you want to keep the c axis where it is
            # to keep the C- settings
            transf = np.zeros(shape=(3, 3))
            if self.spg_symbol.startswith("C"):
                transf[2] = [0, 0, 1]
                a, b = sorted(latt.abc[:2])
                sorted_dic = sorted([{'vec': latt.matrix[i],
                                      'length': latt.abc[i],
                                      'orig_idx': i} for i in [0, 1]],
                                    key=lambda k: k['length'])
                for i in range(2):
                    transf[i][sorted_dic[i]['orig_idx']] = 1
                c = latt.abc[2]
            elif self.spg_symbol.startswith(
                    "A"):  # change to C-centering to match Setyawan/Curtarolo convention
                transf[2] = [1, 0, 0]
                a, b = sorted(latt.abc[1:])
                sorted_dic = sorted([{'vec': latt.matrix[i],
                                      'length': latt.abc[i],
                                      'orig_idx': i} for i in [1, 2]],
                                    key=lambda k: k['length'])
                for i in range(2):
                    transf[i][sorted_dic[i]['orig_idx']] = 1
                c = latt.abc[0]
            else:
                for i, d in enumerate(sorted_dic):
                    transf[i][d['orig_idx']] = 1
                a, b, c = sorted_lengths
            latt = Lattice.orthorhombic(a, b, c)

        elif latt_type == "tetragonal":
            # find the "a" vectors
            # it is basically the vector repeated two times
            transf = np.zeros(shape=(3, 3))
            a, b, c = sorted_lengths
            for i, d in enumerate(sorted_dic):
                transf[i][d['orig_idx']] = 1

            if abs(b - c) < tol < abs(a - c):
                a, c = c, a
                transf = np.dot([[0, 0, 1], [0, 1, 0], [1, 0, 0]], transf)
            latt = Lattice.tetragonal(a, c)
        elif latt_type in ("hexagonal", "rhombohedral"):
            # for the conventional cell representation,
            # we allways show the rhombohedral lattices as hexagonal

            # check first if we have the refined structure shows a rhombohedral
            # cell
            # if so, make a supercell
            a, b, c = latt.abc
            if np.all(np.abs([a - b, c - b, a - c]) < 0.001):
                struct.make_supercell(((1, -1, 0), (0, 1, -1), (1, 1, 1)))
                a, b, c = sorted(struct.lattice.abc)

            if abs(b - c) < 0.001:
                a, c = c, a
            new_matrix = [[a / 2, -a * math.sqrt(3) / 2, 0],
                          [a / 2, a * math.sqrt(3) / 2, 0],
                          [0, 0, c]]
            latt = Lattice(new_matrix)
            transf = np.eye(3, 3)

        elif latt_type == "monoclinic":
            # You want to keep the c axis where it is to keep the C- settings

            if self.get_space_group_operations().int_symbol.startswith("C"):
                transf = np.zeros(shape=(3, 3))
                transf[2] = [0, 0, 1]
                sorted_dic = sorted([{'vec': latt.matrix[i],
                                      'length': latt.abc[i],
                                      'orig_idx': i} for i in [0, 1]],
                                    key=lambda k: k['length'])
                a = sorted_dic[0]['length']
                b = sorted_dic[1]['length']
                c = latt.abc[2]
                new_matrix = None
                for t in itertools.permutations(list(range(2)), 2):
                    m = latt.matrix
                    latt2 = Lattice([m[t[0]], m[t[1]], m[2]])
                    lengths = latt2.lengths
                    angles = latt2.angles
                    if angles[0] > 90:
                        # if the angle is > 90 we invert a and b to get
                        # an angle < 90
                        a, b, c, alpha, beta, gamma = Lattice(
                            [-m[t[0]], -m[t[1]], m[2]]).parameters
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][2] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                        continue

                    if angles[0] < 90:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][2] = 1
                        a, b, c = lengths
                        alpha = math.pi * angles[0] / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]

                if new_matrix is None:
                    # this if is to treat the case
                    # where alpha==90 (but we still have a monoclinic sg
                    new_matrix = [[a, 0, 0],
                                  [0, b, 0],
                                  [0, 0, c]]
                    transf = np.zeros(shape=(3, 3))
                    transf[2] = [0, 0, 1]  # see issue #1929
                    for i, d in enumerate(sorted_dic):
                        transf[i][d['orig_idx']] = 1
            # if not C-setting
            else:
                # try all permutations of the axis
                # keep the ones with the non-90 angle=alpha
                # and b<c
                new_matrix = None
                for t in itertools.permutations(list(range(3)), 3):
                    m = latt.matrix
                    a, b, c, alpha, beta, gamma = Lattice(
                        [m[t[0]], m[t[1]], m[t[2]]]).parameters
                    if alpha > 90 and b < c:
                        a, b, c, alpha, beta, gamma = Lattice(
                            [-m[t[0]], -m[t[1]], m[t[2]]]).parameters
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][t[2]] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                        continue
                    if alpha < 90 and b < c:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][t[2]] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                if new_matrix is None:
                    # this if is to treat the case
                    # where alpha==90 (but we still have a monoclinic sg
                    new_matrix = [[sorted_lengths[0], 0, 0],
                                  [0, sorted_lengths[1], 0],
                                  [0, 0, sorted_lengths[2]]]
                    transf = np.zeros(shape=(3, 3))
                    for i, d in enumerate(sorted_dic):
                        transf[i][d['orig_idx']] = 1

            if international_monoclinic:
                pass

            latt = Lattice(new_matrix)

        elif latt_type == "triclinic":
            # we use a LLL Minkowski-like reduction for the triclinic cells
            struct = struct.get_reduced_structure("LLL")

            a, b, c = latt.lengths
            alpha, beta, gamma = [math.pi * i / 180 for i in latt.angles]
            new_matrix = None
            test_matrix = [[a, 0, 0],
                           [b * cos(gamma), b * sin(gamma), 0.0],
                           [c * cos(beta),
                            c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                          - cos(beta) ** 2
                                          + 2 * cos(alpha) * cos(beta)
                                          * cos(gamma)) / sin(gamma)]]

            def is_all_acute_or_obtuse(m):
                recp_angles = np.array(Lattice(m).reciprocal_lattice.angles)
                return np.all(recp_angles <= 90) or np.all(recp_angles > 90)

            if is_all_acute_or_obtuse(test_matrix):
                transf = np.eye(3)
                new_matrix = test_matrix

            test_matrix = [[-a, 0, 0],
                           [b * cos(gamma), b * sin(gamma), 0.0],
                           [-c * cos(beta),
                            -c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                           - cos(beta) ** 2
                                           + 2 * cos(alpha) * cos(beta)
                                           * cos(gamma)) / sin(gamma)]]

            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]]
                new_matrix = test_matrix

            test_matrix = [[-a, 0, 0],
                           [-b * cos(gamma), -b * sin(gamma), 0.0],
                           [c * cos(beta),
                            c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                          - cos(beta) ** 2
                                          + 2 * cos(alpha) * cos(beta)
                                          * cos(gamma)) / sin(gamma)]]

            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]]
                new_matrix = test_matrix

            test_matrix = [[a, 0, 0],
                           [-b * cos(gamma), -b * sin(gamma), 0.0],
                           [-c * cos(beta),
                            -c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                           - cos(beta) ** 2
                                           + 2 * cos(alpha) * cos(beta)
                                           * cos(gamma)) / sin(gamma)]]
            if is_all_acute_or_obtuse(test_matrix):
                transf = [[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]]
                new_matrix = test_matrix

            latt = Lattice(new_matrix)

        new_coords = np.dot(transf, np.transpose(struct.frac_coords)).T
        new_struct = Structure(latt, struct.species_and_occu, new_coords,
                               site_properties=struct.site_properties,
                               to_unit_cell=True)
        return new_struct.get_sorted_structure()

    def alpha_conv(self):
        if hasattr(self, "_conv"):
            pass
        else:
            self._conv = self.get_conventional_standard_structure(international_monoclinic=False)
        return self._conv.lattice.parameters[3]

    def alpha_prim(self):
        if hasattr(self, "_prim"):
            pass
        else:
            self._prim = self.get_primitive_standard_structure(international_monoclinic=False)
        return self._prim.lattice.parameters[3]

    def angle_rec_lattice(self):
        if hasattr(self, "_prim"):
            pass
        else:
            self._prim = self.get_primitive_standard_structure(international_monoclinic=False)

        return reciprocal_lattice_angle(self._prim.lattice.matrix)

    def abc_conv(self):
        if hasattr(self, "_conv"):
            pass
        else:
            self._conv = self.get_conventional_standard_structure(international_monoclinic=False)
        return self._conv.lattice.abc

    def message(self):
        """ return lattice_type, spg_symbol,
                    a, b, c, alpha_prim, alpha_conv,
                    alpha_rec_lattice,
                    beta_rec_lattice,
                    gamma_rec_lattice]"""
        try:
            a, b, c = self.abc_conv()
            alpha_conv = self.alpha_conv()
            alpha_prim = self.alpha_prim()
            alpha_rec_lattice, beta_rec_lattice, gamma_rec_lattice = self.abc_conv()

            return [self.lattice_type, self.spg_symbol,
                    a, b, c, alpha_prim, alpha_conv,
                    alpha_rec_lattice,
                    beta_rec_lattice,
                    gamma_rec_lattice]
        except TypeError:
            return []

    @staticmethod
    def decide(lattice_type, spg_symbol,
               a, b, c, alpha_prim, alpha_conv,
               alpha_rec_lattice,
               beta_rec_lattice,
               gamma_rec_lattice):
        """get type number by message"""
        kp_num = -1

        if lattice_type == "cubic":
            if "P" in spg_symbol:
                kp_num = 0
            elif "F" in spg_symbol:
                kp_num = 1
            elif "I" in spg_symbol:
                kp_num = 2
            else:
                warn("Unexpected value for spg_symbol: %s" % spg_symbol)

        elif lattice_type == "tetragonal":
            if "P" in spg_symbol:
                kp_num = 3
            elif "I" in spg_symbol:

                if c < a:
                    kp_num = 4
                else:
                    kp_num = 5
            else:
                warn("Unexpected value for spg_symbol: %s" % spg_symbol)

        elif lattice_type == "orthorhombic":

            if "P" in spg_symbol:
                kp_num = 6

            elif "F" in spg_symbol:
                if 1 / a ** 2 > 1 / b ** 2 + 1 / c ** 2:
                    kp_num = 7
                elif 1 / a ** 2 < 1 / b ** 2 + 1 / c ** 2:
                    kp_num = 8
                else:
                    kp_num = 9

            elif "I" in spg_symbol:
                kp_num = 10

            elif "C" in spg_symbol or "A" in spg_symbol:
                kp_num = 11
            else:
                warn("Unexpected value for spg_symbol: %s" % spg_symbol)

        elif lattice_type == "hexagonal":
            kp_num = 12

        elif lattice_type == "rhombohedral":

            if alpha_prim < 90:
                kp_num = 13
            else:
                kp_num = 14

        elif lattice_type == "monoclinic":

            alpha = alpha_conv

            if "P" in spg_symbol:
                kp_num = 15
            elif "C" in spg_symbol:
                kgamma = gamma_rec_lattice
                if kgamma > 90:
                    kp_num = 16
                if kgamma == 90:
                    kp_num = 17
                if kgamma < 90:
                    if b * cos(alpha * pi / 180) / c + b ** 2 * sin(alpha * pi / 180) ** 2 / a ** 2 < 1:
                        kp_num = 18
                    if b * cos(alpha * pi / 180) / c + b ** 2 * sin(alpha * pi / 180) ** 2 / a ** 2 == 1:
                        kp_num = 19
                    if b * cos(alpha * pi / 180) / c + b ** 2 * sin(alpha * pi / 180) ** 2 / a ** 2 > 1:
                        kp_num = 20
            else:
                warn("Unexpected value for spg_symbol: %s" % spg_symbol)

        elif lattice_type == "triclinic":
            kalpha = alpha_rec_lattice
            kbeta = beta_rec_lattice
            kgamma = gamma_rec_lattice
            if kalpha > 90 and kbeta > 90 and kgamma > 90:
                kp_num = 21
            if kalpha < 90 and kbeta < 90 and kgamma < 90:
                kp_num = 22
            if kalpha > 90 and kbeta > 90 and kgamma == 90:
                kp_num = 23
            if kalpha < 90 and kbeta < 90 and kgamma == 90:
                kp_num = 24

        else:
            warn("Unknown lattice type %s" % lattice_type)

        return kp_num

    def get_sgt(self):
        """return 7 member.
        [type_number,
        a,
        b,
        c,
        alpha_prim,
        alpha_conv]
        """
        message = self.message()
        if not message:
            return ERROE_SPACE
        else:
            return np.array([self.decide(*message), *message[2:7]])


class HighSymPoints:
    """Get High K point"""

    def __init__(self):
        self.kp20tem = [
                           np.array([0.0, 0.0, 0.0])] * 20

    def _get_all_kp(self, a, b, c, alpha_prim, alpha_conv, **kwargs):
        return [self.cubic,  # 0 /
                self.fcc,  # 1 /
                self.bcc,  # 2 /
                self.tet,  # 3 /
                self.bctet1(c, a),  # 4  conv.lattice.abc
                self.bctet2(c, a),  # 5  conv.lattice.abc
                self.orc,  # 6 /
                self.orcf1(a, b, c),  # 7
                self.orcf2(a, b, c),  # 8
                self.orcf3(a, b, c),  # 9
                self.orci(a, b, c),  # 10
                self.orcc(a, b, c),  # 11
                self.hex,  # 12 /
                self.rhl1(alpha_prim * pi / 180),  # 13  _prim.lattice.parameters
                self.rhl2(alpha_prim * pi / 180),  # 14 _prim.lattice.parameters
                self.mcl(b, c, alpha_conv * pi / 180),  # 15  _conv.lattice.parameters
                self.mclc1(a, b, c, alpha_conv * pi / 180),  # 16
                self.mclc2(a, b, c, alpha_conv * pi / 180),  # 17
                self.mclc3(a, b, c, alpha_conv * pi / 180),  # 18
                self.mclc4(a, b, c, alpha_conv * pi / 180),  # 19
                self.mclc5(a, b, c, alpha_conv * pi / 180),  # 20
                self.tria,  # 21 /
                self.trib,  # 22 /
                self.tria,  # 23 /
                self.trib,  # 24 /
                ]

    def kp(self, ty, a, b, c, alpha_prim, alpha_conv) -> List:
        ty = int(ty)
        if ty == 0:
            return self.cubic  # 0 /
        elif ty == 1:
            return self.fcc  # 1
        elif ty == 2:
            return self.bcc  # 2 /
        elif ty == 3:
            return self.tet  # 3 /
        elif ty == 4:
            return self.bctet1(c, a)
        elif ty == 5:
            return self.bctet2(c, a)  # 5  conv.lattice.abc
        elif ty == 6:
            return self.orc  # 6 /
        elif ty == 7:
            return self.orcf1(a, b, c)  # 7
        elif ty == 8:
            return self.orcf2(a, b, c)  # 8
        elif ty == 9:
            return self.orcf3(a, b, c)  # 9
        elif ty == 10:
            return self.orci(a, b, c)  # 10
        elif ty == 11:
            return self.orcc(a, b, c)  # 11
        elif ty == 12:
            return self.hex  # 12 /
        elif ty == 13:
            return self.rhl1(alpha_prim * pi / 180)  # 13  _prim.lattice.parameters

        elif ty == 14:
            return self.rhl2(alpha_prim * pi / 180)  # 14 _prim.lattice.parameters
        elif ty == 15:
            return self.mcl(b, c, alpha_conv * pi / 180)  # 15  _conv.lattice.parameters
        elif ty == 16:
            return self.mclc1(a, b, c, alpha_conv * pi / 180)  # 16
        elif ty == 17:
            return self.mclc2(a, b, c, alpha_conv * pi / 180)  # 17
        elif ty == 18:
            return self.mclc3(a, b, c, alpha_conv * pi / 180)  # 18
        elif ty == 19:
            return self.mclc4(a, b, c, alpha_conv * pi / 180)  # 19
        elif ty == 20:
            return self.mclc5(a, b, c, alpha_conv * pi / 180)  # 20
        elif ty == 21:
            return self.tria  # 21 /
        elif ty == 22:
            return self.trib  # 22 /
        elif ty == 23:
            return self.tria  # 23 /
        elif ty == 24:
            return self.trib  # 24 /
        # todo add 2d
        else:
            return []

    @property
    def cubic(self):
        """
        CUB Path
        """
        # self.name = "CUB"

        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]

        return kp

    @property
    def fcc(self):
        """
        FCC Path
        """
        # self.name = "FCC"

        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0]),
            np.array([0.5, 1.0 / 4.0, 3.0 / 4.0]),
            np.array([0.5, 0.0, 0.5]),
        ]
        return kp

    @property
    def bcc(self):
        """
        BCC Path
        """
        # self.name = "BCC"

        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, -0.5, 0.5]),
            np.array([0.25, 0.25, 0.25]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @property
    def tet(self):
        """
        TET Path
        """
        # self.name = "TET"
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @staticmethod
    def bctet1(c, a):
        """
        BCT1 Path
        """
        # self.name = "BCT1"
        eta = (1 + c ** 2 / a ** 2) / 4.0
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.5, 0.5, 0.5]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.25, 0.25, 0.25]),
            np.array([0.0, 0.0, 0.5]),
            np.array([eta, eta, -eta]),
            np.array([-eta, 1 - eta, eta]),
        ]

        return kp

    @staticmethod
    def bctet2(c, a):
        """
        BCT2 Path
        """
        # self.name = "BCT2"
        eta = (1 + a ** 2 / c ** 2) / 4.0
        zeta = a ** 2 / (2 * c ** 2)
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.25, 0.25, 0.25]),
            np.array([-eta, eta, eta]),
            np.array([eta, 1 - eta, -eta]),
            np.array([0.0, 0.0, 0.5]),
            np.array([-zeta, zeta, 0.5]),
            np.array([0.5, 0.5, -zeta]),
            np.array([0.5, 0.5, -0.5]),
        ]

        return kp

    @property
    def orc(self):
        """
        ORC Path
        """
        # self.name = "ORC"
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @staticmethod
    def orcf1(a, b, c):
        """
        ORFC1 Path
        """
        # self.name = "ORCF1"
        zeta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
        eta = (1 + a ** 2 / b ** 2 + a ** 2 / c ** 2) / 4

        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5 + zeta, zeta]),
            np.array([0.5, 0.5 - zeta, 1 - zeta]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1, 0.5, 0.5]),
            np.array([0.0, eta, eta]),
            np.array([1, 1 - eta, 1 - eta]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]

        return kp

    @staticmethod
    def orcf2(a, b, c):
        """
        ORFC2 Path
        """
        # self.name = "ORCF2"
        phi = (1 + c ** 2 / b ** 2 - c ** 2 / a ** 2) / 4
        eta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
        delta = (1 + b ** 2 / a ** 2 - b ** 2 / c ** 2) / 4
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5 - eta, 1 - eta]),
            np.array([0.5, 0.5 + eta, eta]),
            np.array([0.5 - delta, 0.5, 1 - delta]),
            np.array([0.5 + delta, 0.5, delta]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1 - phi, 0.5 - phi, 0.5]),
            np.array([phi, 0.5 + phi, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]

        return kp

    @staticmethod
    def orcf3(a, b, c):
        """
        ORFC3 Path
        """
        # self.name = "ORCF3"
        zeta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
        eta = (1 + a ** 2 / b ** 2 + a ** 2 / c ** 2) / 4
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5 + zeta, zeta]),
            np.array([0.5, 0.5 - zeta, 1 - zeta]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1, 0.5, 0.5]),
            np.array([0.0, eta, eta]),
            np.array([1, 1 - eta, 1 - eta]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]

        return kp

    @staticmethod
    def orci(a, b, c):
        """
        ORCI Path
        """
        # self.name = "ORCI"
        zeta = (1 + a ** 2 / c ** 2) / 4
        eta = (1 + b ** 2 / c ** 2) / 4
        delta = (b ** 2 - a ** 2) / (4 * c ** 2)
        mu = (a ** 2 + b ** 2) / (4 * c ** 2)
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-mu, mu, 0.5 - delta]),
            np.array([mu, -mu, 0.5 + delta]),
            np.array([0.5 - delta, 0.5 + delta, -mu]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.5]),
            np.array([0.25, 0.25, 0.25]),
            np.array([-zeta, zeta, zeta]),
            np.array([zeta, 1 - zeta, -zeta]),
            np.array([eta, -eta, eta]),
            np.array([1 - eta, eta, -eta]),
            np.array([0.5, 0.5, -0.5]),
        ]

        return kp

    @staticmethod
    def orcc(a, b, c):
        """
        ORCC Path
        """
        # self.name = "ORCC"
        zeta = (1 + a ** 2 / b ** 2) / 4
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([zeta, zeta, 0.5]),
            np.array([-zeta, 1 - zeta, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.0, 0.5, 0.0]),
            np.array([-0.5, 0.5, 0.5]),
            np.array([zeta, zeta, 0.0]),
            np.array([-zeta, 1 - zeta, 0.0]),
            np.array([-0.5, 0.5, 0]),
            np.array([0.0, 0.0, 0.5]),
        ]
        return kp

    @property
    def hex(self):
        """
        HEX Path
        """
        # self.name = "HEX"
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.5]),
            np.array([1.0 / 3.0, 1.0 / 3.0, 0.5]),
            np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.0, 0.0]),
        ]

        return kp

    @staticmethod
    def rhl1(alpha):
        """
        RHL1 Path
        """
        # self.name = "RHL1"
        eta = (1 + 4 * cos(alpha)) / (2 + 4 * cos(alpha))
        nu = 3.0 / 4.0 - eta / 2.0
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([eta, 0.5, 1.0 - eta]),
            np.array([1.0 / 2.0, 1.0 - eta, eta - 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.5]),
            np.array([eta, nu, nu]),
            np.array([1.0 - nu, 1.0 - nu, 1.0 - eta]),
            np.array([nu, nu, eta - 1.0]),
            np.array([1.0 - nu, nu, 0.0]),
            np.array([nu, 0.0, -nu]),
            np.array([0.5, 0.5, 0.5]),
        ]

        return kp

    @staticmethod
    def rhl2(alpha):
        """
        RHL2 Path
        """
        # self.name = "RHL2"
        eta = 1 / (2 * tan(alpha / 2.0) ** 2)
        nu = 3.0 / 4.0 - eta / 2.0
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, -0.5, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([1 - nu, -nu, 1 - nu]),
            np.array([nu, nu - 1.0, nu - 1.0]),
            np.array([eta, eta, eta]),
            np.array([1.0 - eta, -eta, -eta]),
            np.array([0.5, -0.5, 0.5]),
        ]
        return kp

    @staticmethod
    def mcl(b, c, beta):
        """
        MCL Path
        """
        # self.name = "MCL"
        eta = (1 - b * cos(beta) / c) / (2 * sin(beta) ** 2)
        nu = 0.5 - eta * c * cos(beta) / b
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, -0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.0, eta, 1.0 - nu]),
            np.array([0.0, 1.0 - eta, nu]),
            np.array([0.0, eta, -nu]),
            np.array([0.5, eta, 1.0 - nu]),
            np.array([0.5, 1 - eta, nu]),
            np.array([0.5, 1 - eta, nu]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, -0.5]),
            np.array([0.5, 0.0, 0.0]),
        ]

        return kp

    @staticmethod
    def mclc1(a, b, c, alpha):
        """
        MCLC1 Path
        """
        # self.name = "MCLC1"
        zeta = (2 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
        eta = 0.5 + 2 * zeta * c * cos(alpha) / b
        psi = 0.75 - a ** 2 / (4 * b ** 2 * sin(alpha) ** 2)
        phi = psi + (0.75 - psi) * b * cos(alpha) / c
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([1 - zeta, 1 - zeta, 1 - eta]),
            np.array([zeta, zeta, eta]),
            np.array([-zeta, -zeta, 1 - eta]),
            np.array([phi, 1 - phi, 0.5]),
            np.array([1 - phi, phi - 1, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([1 - psi, psi - 1, 0.0]),
            np.array([psi, 1 - psi, 0.0]),
            np.array([psi - 1, -psi, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([-0.5, -0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]
        return kp

    @staticmethod
    def mclc2(a, b, c, alpha):
        """
        MCLC2 Path
        """
        # self.name = "MCLC2"
        zeta = (2 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
        eta = 0.5 + 2 * zeta * c * cos(alpha) / b
        psi = 0.75 - a ** 2 / (4 * b ** 2 * sin(alpha) ** 2)
        phi = psi + (0.75 - psi) * b * cos(alpha) / c
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([1 - zeta, 1 - zeta, 1 - eta]),
            np.array([zeta, zeta, eta]),
            np.array([-zeta, -zeta, 1 - eta]),
            np.array([1 - zeta, -zeta, 1 - eta]),
            np.array([phi, 1 - phi, 0.5]),
            np.array([1 - phi, phi - 1, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([1 - psi, psi - 1, 0.0]),
            np.array([psi, 1 - psi, 0.0]),
            np.array([psi - 1, -psi, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([-0.5, -0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @staticmethod
    def mclc3(a, b, c, alpha):
        """
        MCLC3 Path
        """
        # self.name = "MCLC3"
        mu = (1 + b ** 2 / a ** 2) / 4.0
        delta = b * c * cos(alpha) / (2 * a ** 2)
        zeta = mu - 0.25 + (1 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
        eta = 0.5 + 2 * zeta * c * cos(alpha) / b
        phi = 1 + zeta - 2 * mu
        psi = eta - 2 * delta
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1 - phi, 1 - phi, 1 - psi]),
            np.array([phi, phi - 1, psi]),
            np.array([1 - phi, -phi, 1 - psi]),
            np.array([zeta, zeta, eta]),
            np.array([1 - zeta, -zeta, 1 - eta]),
            np.array([-zeta, -zeta, 1 - eta]),
            np.array([0.5, -0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([0.5, -0.5, 0.0]),
            np.array([mu, mu, delta]),
            np.array([1 - mu, -mu, -delta]),
            np.array([-mu, -mu, -delta]),
            np.array([mu, mu - 1, delta]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @staticmethod
    def mclc4(a, b, c, alpha):
        """
        MCLC4 Path
        """
        # self.name = "MCLC4"
        mu = (1 + b ** 2 / a ** 2) / 4.0
        delta = b * c * cos(alpha) / (2 * a ** 2)
        zeta = mu - 0.25 + (1 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
        eta = 0.5 + 2 * zeta * c * cos(alpha) / b
        phi = 1 + zeta - 2 * mu
        psi = eta - 2 * delta
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1 - phi, 1 - phi, 1 - psi]),
            np.array([phi, phi - 1, psi]),
            np.array([1 - phi, -phi, 1 - psi]),
            np.array([zeta, zeta, eta]),
            np.array([1 - zeta, -zeta, 1 - eta]),
            np.array([-zeta, -zeta, 1 - eta]),
            np.array([0.5, -0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([0.5, -0.5, 0.0]),
            np.array([mu, mu, delta]),
            np.array([1 - mu, -mu, -delta]),
            np.array([-mu, -mu, -delta]),
            np.array([mu, mu - 1, delta]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @staticmethod
    def mclc5(a, b, c, alpha):
        """
        MCLC5 Path
        """
        # self.name = "MCLC5"
        zeta = (b ** 2 / a ** 2 + (1 - b * cos(alpha) / c) / sin(alpha) ** 2) / 4
        eta = 0.5 + 2 * zeta * c * cos(alpha) / b
        mu = eta / 2 + b ** 2 / (4 * a ** 2) - b * c * cos(alpha) / (2 * a ** 2)
        nu = 2 * mu - zeta
        rho = 1 - zeta * a ** 2 / b ** 2
        omega = (4 * nu - 1 - b ** 2 * sin(alpha) ** 2 / a ** 2) * c / (2 * b * cos(alpha))
        delta = zeta * c * cos(alpha) / b + omega / 2 - 0.25
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([nu, nu, omega]),
            np.array([1 - nu, 1 - nu, 1 - omega]),
            np.array([nu, nu - 1, omega]),
            np.array([zeta, zeta, eta]),
            np.array([1 - zeta, -zeta, 1 - eta]),
            np.array([-zeta, -zeta, 1 - eta]),
            np.array([rho, 1 - rho, 0.5]),
            np.array([1 - rho, rho - 1, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([0.5, -0.5, 0.0]),
            np.array([mu, mu, delta]),
            np.array([1 - mu, -mu, -delta]),
            np.array([-mu, -mu, -delta]),
            np.array([mu, mu - 1, delta]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @property
    def tria(self):
        """
        TRI1a Path
        """
        # self.name = "TRI1a"
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]

        return kp

    @property
    def trib(self):
        """
        TRI1b Path
        """
        # self.name = "TRI1b"
        kp = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, -0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
            np.array([-0.5, -0.5, 0.5]),
            np.array([0.0, -0.5, 0.5]),
            np.array([0.0, -0.5, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.5]),
        ]

        return kp

    def kpuni(self, ty, a, b, c, alpha_prim, alpha_conv) -> np.ndarray:
        a = self.kp(ty, a, b, c, alpha_prim, alpha_conv)
        a.extend(self.kp20tem)
        return np.stack(a[:20], 0)

    def kpeuni(self, narrayi) -> np.ndarray:
        return self.kpuni(*narrayi)

    def kps(self, narray: np.ndarray) -> np.ndarray:

        return np.stack([self.kpeuni(narrayi) for narrayi in narray], 0)

    def kps_tensor(self, narray: torch.Tensor) -> np.ndarray:

        return np.stack([self.kpeuni(narrayi) for narrayi in narray.numpy()], 0)


class HighSymPointsCPP:
    """Quick for get k points by cpp method."""

    @staticmethod
    def kps(narray: np.ndarray) -> np.ndarray:
        """Get kps shape(n,20,3)"""
        k = np.array(kpcpu.kps(narray))
        return k.transpose(0, 2, 1)

    @staticmethod
    def kps_tensor(narray: torch.Tensor) -> np.ndarray:
        """get kps shape(n,20,3)"""
        k = kpcpu.kps(narray.numpy())
        return np.transpose(k, (0, 2, 1))

    @staticmethod
    def kpeuni(narrayi) -> np.ndarray:
        return kpcpu.kpeuni(narrayi)

    @staticmethod
    def kpuni(ty, a, b, c, alpha_prim, alpha_conv) -> np.ndarray:
        return kpcpu.kpuni(ty, a, b, c, alpha_prim, alpha_conv)


class HSPConverter(BaseFeature):
    """
    Get K points.
    """

    def __init__(self):
        super(HSPConverter, self).__init__()

        self.sg_converter = SpacegroupAnalyzerUser
        if kpcpu is None:
            self.hsp = HighSymPoints()
        else:
            self.hsp = HighSymPointsCPP()

    def convert(self, d: Structure) -> np.ndarray:
        """Get kps shape (n,20,3)"""
        space_group_analysis = self.sg_converter(d)
        spg_tp = space_group_analysis.get_sgt()
        return self.hsp.kps(spg_tp)
