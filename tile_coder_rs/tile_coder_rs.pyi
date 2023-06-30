import numpy as np
from typing import List

def get_tc_indices(
    dims: int,
    tiles: np.ndarray,
    tilings: int,
    bounds: np.ndarray,
    offsets: np.ndarray,
    bound_strats: List[str],
    pos: np.ndarray,
) -> np.ndarray: ...
