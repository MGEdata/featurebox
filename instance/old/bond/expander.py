from pathlib import Path

import numpy as np

from featurebox.featurizers.base_feature import BaseFeature

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


# ###############bond##########################################

class BondGaussianConverter(BaseFeature):
    """
    For bond distance.
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = None, width=0.5, ndim=1):
        """

        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis

        """
        if centers is None:
            centers = np.linspace(0, 5, 100)
        self.centers = centers
        self.width = width
        self.ndim = ndim
        super(BondGaussianConverter, self).__init__()

    def _convert(self, d: np.ndarray) -> np.ndarray:
        """
        expand distance vector d with given parameters

        Args:
            d: (1d array) distance array

        Returns:
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)

        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)
