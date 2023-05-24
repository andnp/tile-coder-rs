import numpy as np
from typing import Sequence

def get_tc_indices(
    dims: int,
    tiles: Sequence[int],
    tilings: int,
    bounds: np.ndarray,
    offsets: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray: ...
