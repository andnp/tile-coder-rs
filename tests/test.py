import numpy as np

from PyFixedReps.TileCoder import TileCoder, TileCoderConfig
from tc import TileCoder as RustTileCoder

config = TileCoderConfig(
    tiles=8,
    tilings=64,
    dims=6,
    input_ranges=[
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 2),
    ]
)
arr = np.arange(6) / 6

# tc1 = TileCoder(config)
# print(tc1.get_indices(arr))

tc2 = RustTileCoder(config)
print(tc2.features())
print(tc2.get_indices(arr))
