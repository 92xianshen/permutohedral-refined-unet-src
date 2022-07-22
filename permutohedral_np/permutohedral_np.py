"""
- Permutohedral lattice implementation in NP, channel-last as well.
- `np.float32` and `np.int32` are default float and integer types, respectively.
"""

import numpy as np
from collections import defaultdict

class Permutohedral():
    def __init__(self, N, d) -> None:
        self.N, self.M, self.d = N, 0, d

        canonical = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
        for i in range(d + 1):
            canonical[i, : d + 1 - i] = i
            canonical[i, d + 1 - i :] = i - (d + 1)
        self.canonical = canonical  # (d + 1, d + 1)

        E = np.vstack(
            [
                np.ones((d,), dtype=np.float32),
                np.diag(-np.arange(d, dtype=np.float32) - 2)
                + np.triu(np.ones((d, d), dtype=np.float32)),
            ]
        )  # (d + 1, d)
        self.E = E.astype(np.float32)  # (d + 1, d)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * np.float32(d + 1)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (
            1.0 / np.sqrt((np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev
        )  # (d, )
        self.scale_factor = np.float32(scale_factor)  # (d, )

        diff_valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32)) # (d + 1, d + 1)
        self.diff_valid = diff_valid # (d + 1, d + 1)

        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        self.alpha = np.float32(1.0 / (1.0 + np.power(2.0, -d))) 

        d_mat = np.ones((d + 1,), dtype=np.int32) * d  # (d + 1, )
        d_mat = np.diag(d_mat)  # (d + 1, d + 1)
        diagone = np.diag(np.ones(d + 1, dtype=np.int32))  # (d + 1, d + 1)
        self.d_mat = d_mat  # (d + 1, d + 1)
        self.diagone = diagone  # (d + 1, d + 1)

        self.blur_neighbors = None
        self.os = None
        self.ws = None

    def init(self, features):
        # Compute the simplex each feature lies in
        # !!! Shape of feature (N, d)
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = features * self.scale_factor[np.newaxis, ...]  # (N, d)
        elevated = np.matmul(cf, self.E.T)  # (N, d + 1)

        # Find the closest 0-colored simplex through rounding
        down_factor = np.float32(1.0 / (self.d + 1))
        up_factor = np.float32(self.d + 1)
        v = down_factor * elevated  # (N, d + 1)
        up = np.ceil(v) * up_factor  # (N, d + 1)
        down = np.floor(v) * up_factor  # [N, d + 1]
        rem0 = np.where(up - elevated < elevated - down, up, down).astype(np.float32)  # (N, d + 1)
        _sum = np.int32(rem0.sum(axis=-1) * down_factor) # (N, )

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        diff = elevated - rem0  # (N, d + 1)
        diff_i = diff[..., np.newaxis]  # (N, d + 1, 1)
        diff_j = diff[..., np.newaxis, :]  # (N, 1, d + 1)
        rank = ((diff_i < diff_j) * self.diff_valid[np.newaxis, ...]).sum(axis=-1).astype(np.int32)  # (N, d + 1)
        rank += ((diff_i >= diff_j) * self.diff_valid[np.newaxis, ...]).sum(axis=-2)  # (N, d + 1)

        # If the point doesn't lie on the plane (sum != 0) bring it back
        rank += _sum[..., np.newaxis]  # (N, d + 1)
        ls_zero = rank < 0  # (N, d + 1)
        gt_d = rank > self.d  # (N, d + 1)
        rank[ls_zero] += (self.d + 1)
        rem0[ls_zero] += (self.d + 1)
        rank[gt_d] -= (self.d + 1)
        rem0[gt_d] -= (self.d + 1)

        # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        barycentrics = np.zeros((self.N * (self.d + 2), ), dtype=np.float32) # (N * (d + 2), )
        vs = ((elevated - rem0) * down_factor).reshape(-1) # (N * (d + 1), )
        idx = ((self.d - rank) + np.arange(self.N)[..., np.newaxis] * (self.d + 2)).reshape(-1) # (N * (d + 1), )
        idx1 = ((self.d - rank + 1) + np.arange(self.N)[..., np.newaxis] * (self.d + 2)).reshape(-1) # (N * (d + 1), )
        barycentrics[idx] += vs # (N * (d + 2), )
        barycentrics[idx1] -= vs # (N * (d + 2), )
        barycentrics = barycentrics.reshape((self.N, self.d + 2)) # (N, d + 2)
        barycentrics[..., 0] += (1. + barycentrics[..., self.d + 1]) # (N, d + 2)

        # Compute all vertices and their offset
        canonicalT = self.canonical.T # (d + 1, d + 1)
        canonical_ext = canonicalT[rank] # (N, d + 1, d + 1)
        canonical_ext = np.transpose(canonical_ext, axes=(0, 2, 1)) # (N, d + 1, d + 1)

        keys = np.int32(rem0[..., np.newaxis, :self.d] + canonical_ext[..., :self.d]) # (N, d + 1, d)

        # Keys in string format.
        keys = keys.reshape((-1, self.d)) # (N * (d + 1), d)
        hkeys, offsets = np.unique(keys, return_inverse=True, axis=0) # (M, d), (N * (d + 1), )
        self.M = hkeys.shape[0] # Get M

        hash_table = defaultdict(lambda:-1)
        for i in range(self.M):
            k, v = hkeys[i], i
            sk = ','.join(k.astype(np.str))
            hash_table[sk] = v

        # Find the neighbors of each lattice point
        # Get the number of vertices in the lattice
        # Create the neighborhood structure
        # For each of d+1 axes,
        n1s = np.repeat(hkeys[:, np.newaxis, :], repeats=self.d + 1, axis=1) - 1 # (M, d + 1, d)
        n2s = np.repeat(hkeys[:, np.newaxis, :], repeats=self.d + 1, axis=1) + 1 # (M, d + 1, d)
        n1s += (self.d_mat[np.newaxis, ..., :self.d] + self.diagone[np.newaxis, ..., :self.d]) # (M, d + 1, d)
        n2s -= (self.d_mat[np.newaxis, ..., :self.d] + self.diagone[np.newaxis, ..., :self.d]) # (M, d + 1, d)
        n1s = n1s.reshape((-1, self.d)) # (M * (d + 1), d)
        n2s = n2s.reshape((-1, self.d)) # (M * (d + 1), d)

        self.blur_neighbors = np.zeros((self.M * (self.d + 1), 2), dtype=np.int32) # (M * (d + 1), 2)

        for i in np.arange(n1s.shape[0]):
            n1, n2 = n1s[i], n2s[i]
            sn1 = ','.join(n1.astype(np.str))
            sn2 = ','.join(n2.astype(np.str))
            self.blur_neighbors[i, 0] = hash_table[sn1]
            self.blur_neighbors[i, 1] = hash_table[sn2]

        # Shift all values by 1 such that -1 -> 0 (used for blurring)
        self.os = offsets.reshape(-1) + 1 # (N * (d + 1), )
        self.ws = barycentrics[..., self.d + 1].reshape(-1) # (N * (d + 1), )
        self.blur_neighbors = self.blur_neighbors.reshape((self.M, self.d + 1, 2)) + 1 # (M, d + 1, 2)


    def compute(self, inp, reverse=False):
        """
        Compute.
        """
        value_size = inp.shape[1]
        values = np.zeros((self.M + 2, value_size), dtype=np.float32)

        # ->> Splat
        for v in value_size:
            ch = np.repeat(inp[..., v:v + 1], repeats=self.d + 1, axis=-1) # (N, d + 1)
            ch = ch.reshape(-1) # (N * (d + 1), )
            values[..., v] = np.bincount(self.os, weights=ch * self.ws, minlength=self.M + 2)

        # ->> Blur
        j_range = np.arange(self.d, -1, -1) if reverse else np.arange(self.d + 1)
        
        for j in j_range:
            n1s = self.blur_neighbors[..., j, 0]  # [M, ]
            n2s = self.blur_neighbors[..., j, 1]  # [M, ]
            n1_vals = values[n1s]  # [M, value_size]
            n2_vals = values[n2s]  # [M, value_size]

            values[1:self.M + 1] += 0.5 * (n1_vals + n2_vals)

        # ->> Slice
        out = self.ws[..., np.newaxis] * values[self.os] * self.alpha # (N * (d + 1), value_size)
        out = out.reshape((self.N, self.d + 1, value_size)) # (N, d + 1, value_size)
        out = out.sum(axis=1) # (N, value_size)

        return out