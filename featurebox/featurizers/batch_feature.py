from typing import List

from featurebox.featurizers.atom.mapper import AtomPymatgenPropMap
from featurebox.featurizers.base_feature import BaseFeature
from featurebox.featurizers.state.state_mapper import StructurePymatgenPropMap
from featurebox.featurizers.state.statistics import WeightedAverage


class BatchFeature:
    """Script for generate batch_data, could be copied and user-defined."""

    def __init__(self, data_type: str = "compositions", user_convert: BaseFeature = None, n_jobs: int = 1,
                 on_errors: str = 'raise', return_type: str = 'df', batch_calculate: bool = False,
                 batch_size: int = 30):
        """

        Parameters
        ----------
        data_type:str
            Predefined name ["elements", "compositions", "structures"]
        user_convert:BatchFeature
            which contain `convert` method.
        """
        prop_name = ["density",
                     "volume",
                     "ntypesp",

                     "_lattice.lengths",
                     "_lattice.angles",

                     "composition.average_electroneg",
                     "composition.num_atoms",
                     "composition.total_electrons",
                     "composition.weight",
                     ]
        self.structure_c = StructurePymatgenPropMap(prop_name=prop_name,
                                                    batch_size=batch_size,
                                                    on_errors=on_errors,
                                                    batch_calculate=batch_calculate,
                                                    n_jobs=n_jobs,
                                                    return_type=return_type)
        self.structure_c.set_feature_labels(prop_name)

        func_map = [
            "atomic_radius",
            "atomic_mass",
            "Z",
            "X",
            "number",
            "max_oxidation_state",
            "min_oxidation_state",
            "row",
            "group",
            "atomic_radius_calculated",

            "van_der_waals_radius",
            "mendeleev_no",
            "molar_volume",
            "boiling_point",
            "melting_point",
            "critical_temperature",
            "density_of_solid",
            "average_ionic_radius",
            "average_cationic_radius",
            "average_anionic_radius",
        ]
        appm = AtomPymatgenPropMap(prop_name=func_map, n_jobs=1, search_tp="name")
        self.composition_c = WeightedAverage(appm, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.composition_c.set_feature_labels([f"Mean-{i}" for i in func_map])

        self.element_c = AtomPymatgenPropMap(prop_name=func_map, n_jobs=n_jobs, search_tp="name", on_errors=on_errors)

        if user_convert is not None:
            self.data_type = "user-defined"
            self.convert_method = user_convert
        else:
            self.data_type = data_type
            self.maps = {
                "elements": self.element_c,
                "compositions": self.composition_c,
                "structures": self.structure_c,
                # "electron":self.electron_c,
                # "dos":self.dos_dict,
                # "band_gap":self.band_gap_dict,
            }
            if self.data_type not in self.maps:
                raise TypeError("Undefined name for data_type, if use your self convert, please set 'user_convert'.")
            self.convert_method = self.maps[self.data_type]

    def convert(self, d):
        return self.convert_method.convert(d)

    def transform(self, entries: List):
        return self.convert_method.transform(entries)

    def fit_transform(self, entries: List):
        return self.transform(entries)

    @property
    def feature_labels(self):
        return self.convert_method.feature_labels

    def set_feature_labels(self, values: List[str]):
        self.convert_method.set_feature_labels(values)
