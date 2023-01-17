"""This is one script for ... """
import os

import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster import Cohpcar, Icohplist

if __name__ == '__main__':
    os.chdir(r"D:\MoCMo-O-4")

    ds_by_number = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo",
                    "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", ]

    ###############################

    data_all = {}
    for i in ds_by_number:
        os.chdir(f"{i}/pure_static")

        if os.path.isfile("ICOHPLIST.lobster"):
            icohplist = Icohplist(are_coops=False, are_cobis=False, filename="ICOHPLIST.lobster")

            data = np.array([
                icohplist.icohplist["1"]["length"],
                -(icohplist.icohplist["1"]["icohp"][Spin.up]),
                -(icohplist.icohplist["1"]["icohp"][Spin.down])
            ]).ravel()
            data_all.update({f"{i}": data})

            os.chdir("../..")

        else:
            print(f"no data for {i}")
            data_all.update({f"{i}": None})
            os.chdir("../..")

    da = pd.DataFrame.from_dict(data_all).T
    da.to_csv("-ichop_point.csv")
    ##############

    data_all = {}
    for i in ds_by_number:
        os.chdir(f"{i}/pure_static")

        if os.path.isfile("COHPCAR.lobster"):
            cohpcar = Cohpcar(are_coops=False, are_cobis=False, filename="COHPCAR.lobster")

            # import matplotlib.pyplot as plt
            # plt.plot(cohpcar.energies,cohpcar.cohp_data["average"]["COHP"][Spin.up])
            # plt.show()

            data = np.vstack((cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            # data = np.vstack((cohpcar.energies, cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            data_all.update({f"{i}": data})

            os.chdir("../..")

        else:
            print(f"no data for {i}")
            data_all.update({f"{i}": None})
            os.chdir("../..")

    da = pd.DataFrame.from_dict(data_all)
    da.to_csv("chop_point.csv")
