import timeit



shared_setup = """
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
"""

setup1 = shared_setup + """
tc1 = TileCoder(config)
tc1.get_indices(arr)
"""

setup2 = shared_setup + """
tc2 = RustTileCoder(config)
tc2.get_indices(arr)
"""

test_code1 = """
tc1.get_indices(arr)
"""

test_code2 = """
tc2.get_indices(arr)
"""

v = timeit.timeit(test_code1, setup1)
print('Original tc:', v)

v = timeit.timeit(test_code2, setup2)
print('Rust tc:', v)
