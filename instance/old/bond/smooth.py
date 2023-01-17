from math import cos, pi
from pathlib import Path

import numba
import numpy as np

from featurebox.featurizers.base_feature import BaseFeature

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


# ###############bond##########################################

class Smooth(BaseFeature):
    """
    smooth.
    """

    def __init__(self, r_c=5, r_cs=3):
        """
        Args:
            r_cs: float, strat of decrease cutoff radius.
            r_c: (float) cutoff radius.
        """
        super(Smooth, self).__init__()
        self.r_c = r_c
        self.r_cs = r_cs
        self.ndim = 2

    def convert(self, d: np.ndarray) -> np.ndarray:
        """
        d: np.ndarray, shape (N, fill_size, atom_fea_len).
            ``nbr_fea`` neighbor features for each center_index.
        """
        assert d.ndim == 2 or d.ndim == 3, "Just accept 2D or 1D array,d.shape={}".format(d.shape)
        d = np.array(d)
        d = d.astype(np.float64)

        return smooth_func(d, self.r_c, self.r_cs)


@numba.jit("float64[:](float64[:],float64,float64)")
def _smooth_func(rs, r_c, r_cs):
    ks = []
    for r in rs:
        if r <= r_cs:
            ks.append(1 / r)
        elif r_cs < r <= r_c:
            ks.append(1 / r * (0.5 * cos((r - r_cs) / (r_c - r_cs) * pi) + 0.5))
        else:
            ks.append(0.0)
    return np.array(ks)


def smooth_func(d, r_c, r_cs):
    if d.ndim == 3:
        r = (d[:, :, 0] ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2) ** 0.5
        k = np.array([_smooth_func(ri, r_c, r_cs) for ri in r])
        k = k[..., np.newaxis]
        r = r[..., np.newaxis]

    elif d.ndim == 2:
        r = (d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2) ** 0.5
        k = _smooth_func(r, r_c, r_cs)
        k = k[..., np.newaxis]
        r = r[..., np.newaxis]
    else:
        r = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        if r <= r_cs:
            k = 1 / r
        elif r_cs < r <= r_c:
            k = 1 / r * (0.5 * cos((r - r_cs) / (r_c - r_cs) * pi) + 0.5)
        else:
            k = 0.0

    return k * d / r
