"""
- Permutohedral lattice implementation in NP, channel-last as well.
- `np.float32` and `np.int32` are default float and integer types, respectively.
"""

import numpy as np
from cpp import permutohedral_factory

class Permutohedral():
    def __init__(self, N, d) -> None:
        self.N, self.M, self.d = N, 0, d

        self.blur_neighborsT_flat = np.zeros(((d + 1) * N * 2, ), dtype=np.int32)
        self.os_flat = np.zeros((N * (d + 1), ), dtype=np.int32)
        self.ws_flat = np.zeros((N * (d + 1), ), dtype=np.float32)

    def init(self, features_flat):
        features_1d = features_flat.reshape((-1, ))
        self.M = permutohedral_factory.init(features_1d, self.N, self.d, self.os_flat, self.ws_flat, self.blur_neighborsT_flat)

        # Get M blur neighbor pairs
        self.blur_neighborsT_flat = self.blur_neighborsT_flat[:self.M * (self.d + 1) * 2]

    def compute(self, inp_flat, reverse=False, out_flat=None):
        """
        Compute.
        """  
        N, value_size = inp_flat.shape[:2]
        inp_1d = inp_flat.reshape((-1, ))
        out_1d = out_flat.reshape((-1, ))

        permutohedral_factory.compute(inp_1d, self.N, value_size, self.M, self.d, self.os_flat, self.ws_flat, self.blur_neighborsT_flat, reverse, out_1d)

        out_flat = out_1d.reshape((N, value_size))
