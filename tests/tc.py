from tile_coder_rs import get_tc_indices

import numpy as np
import numpy.typing as npt

from typing import Iterable, List, Optional, Sequence, Tuple
from PyFixedReps.TileCoder import TileCoderConfig
from PyFixedReps.BaseRepresentation import BaseRepresentation


class TileCoder(BaseRepresentation):
    def __init__(self, config: TileCoderConfig, rng: Optional[np.random.RandomState] = None):
        self.rng = rng
        self._c = c = config

        ranges = [None] * c.dims
        if c.input_ranges is not None:
            assert len(c.input_ranges) == c.dims
            ranges = c.input_ranges

        self._tiles = _normalize_tiles(c.tiles, c.dims)
        self._input_ranges = _normalize_scalars(ranges)
        self._tiling_offsets = np.array([ self._build_offset(ntl) for ntl in range(c.tilings) ], dtype=np.float64)
        self._total_tiles = int(c.tilings * self._tiles.prod())

    # construct tiling offsets
    # defaults to evenly spaced tilings
    def _build_offset(self, n: int):
        if self._c.offset == 'random':
            assert self.rng is not None
            return self.rng.uniform(0, 1, size=self._c.dims)

        if self._c.offset == 'cascade':
            tile_length = 1.0 / self._tiles
            return np.ones(self._c.dims) * n * (tile_length / self._c.tilings)

        if self._c.offset == 'even':
            tile_length = 1.0 / self._tiles
            i = n - (self._c.tilings / 2)
            return np.ones(self._c.dims) * i * (tile_length / self._c.tilings)

        raise Exception('Unknown offset type')

    def get_indices(self, pos: npt.ArrayLike):
        pos_ = np.asarray(pos, dtype=np.float64)
        return get_tc_indices(self._c.dims, self._tiles, self._c.tilings, self._input_ranges, self._tiling_offsets, pos_)

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

def _normalize_tiles(tiles: int | Iterable[int], dims: int) -> np.ndarray:
    if isinstance(tiles, int):
        tiles = [tiles for _ in range(dims)]

    return np.fromiter(tiles, dtype=np.uint32, count=dims)

def _normalize_scalars(sc: Sequence[Optional[Tuple[float, float]]]):
    out: List[Tuple[float, float]] = []
    for r in sc:
        if r is None:
            out.append((0., 1.))

        else:
            out.append(r)

    return np.array(out, dtype=np.float64)
