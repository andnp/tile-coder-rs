from .tile_coder import get_tc_indices

import numba
import numpy as np
import numpy.typing as npt

from typing import Optional
from PyFixedReps.TileCoder import TileCoderConfig
from PyFixedReps.BaseRepresentation import BaseRepresentation


class TileCoder(BaseRepresentation):
    def __init__(self, config: TileCoderConfig, rng: Optional[np.random.RandomState] = None):
        self.rng = rng
        self._c = c = config

        self._input_ranges = None
        if c.input_ranges is not None:
            self._input_ranges = np.array(c.input_ranges)

        self._tiling_offsets = np.array([ self._build_offset(ntl) for ntl in range(c.tilings) ])
        self._total_tiles: int = c.tilings * c.tiles ** c.dims

    # construct tiling offsets
    # defaults to evenly spaced tilings
    def _build_offset(self, n: int):
        if self._c.offset == 'random':
            assert self.rng is not None
            return self.rng.uniform(0, 1, size=self._c.dims)

        if self._c.offset == 'cascade':
            tile_length = 1.0 / self._c.tiles
            return np.ones(self._c.dims) * n * (tile_length / self._c.tilings)

        if self._c.offset == 'even':
            tile_length = 1.0 / self._c.tiles
            i = n - (self._c.tilings / 2)
            return np.ones(self._c.dims) * i * (tile_length / self._c.tilings)

        raise Exception('Unknown offset type')

    def get_indices(self, pos: npt.ArrayLike):
        pos_ = np.asarray(pos, dtype=np.float_)
        if self._input_ranges is not None:
            pos_ = minMaxScaling(pos_, self._input_ranges[:, 0], self._input_ranges[:, 1])

        return get_tc_indices(self._c.dims, self._c.tiles, self._c.tilings, self._tiling_offsets, pos_)

    def features(self):
        return int(self._total_tiles * self._c.actions)

    def encode(self, s: npt.ArrayLike):
        indices = self.get_indices(s)
        vec = np.zeros(self.features())

        v = 1.
        if self._c.scale_output:
            v = 1. / self._c.tilings

        vec[indices] = v
        return vec

@numba.njit(cache=True, fastmath=True, nogil=True)
def minMaxScaling(x: np.ndarray, mi: np.ndarray, ma: np.ndarray):
    return (x - mi) / (ma - mi)
