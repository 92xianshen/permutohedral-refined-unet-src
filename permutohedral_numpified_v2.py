''' Renumpify 
'''

import numpy as np

class HashTable:
    pass

class Permutohedral:
    def __init__(self, N, d):
        self.N_, self.M_, self.d_ = N, 0, d
        self.blur_neighbors_ = None
        self.hash_table = HashTable(d, N * (d + 1))
        self.offset_ = np.zeros( (N, (d + 1)), dtype=np.int32 )
        self.rank_ = np.zeros( (N, (d + 1)), dtype=np.int32 )
        self.barycentric_ = np.zeros( (N, (d + 1)), dtype=np.float32 )

    def init(self, feature):
        # feature: (d, N), channel first

        # Compute the lattice coordinates for each feature [there is going to be a lot of magic here
        # pass

        # Allocate the class memory
        # pass

        # Allocate the local memory
        scale_factor = np.zeros((self.d_, ), dtype=np.float32)
        # elevated = np.zeros((self.d_ + 1, ), dtype=np.float32)
        # rem0 = np.zeros((self.d_ + 1, ), dtype=np.float32)
        barycentric = np.zeros((self.d_ + 2, ), dtype=np.float32)
        rank = np.zeros((self.d_ + 1, self.N_), dtype=np.short)
        canonical = np.zeros( ((self.d_ + 1) * (self.d_ + 1), ), dtype=np.short)
        key = np.zeros((self.d_ + 1, ), dtype=np.short)

        # Compute the canonical simplex
        # ->> Numpify
        canonical = canonical.reshape((self.d_ + 1, self.d_ + 1))
        for i in range(self.d_ + 1):
            canonical[i, :self.d_ + 1 - i] = i
            canonical[i, self.d_ + 1 - i:] = i - (self.d_ + 1)
        canonical = canonical.reshape(-1)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2. / 3.) * (self.d_ + 1)
        # Compute the diagonal part of E (p.5 in [Adams et al 2021])
        scale_factor[:] = 1. / np.sqrt( (np.arange(self.d_) + 2) * (np.arange(self.d_) + 1) )  * inv_std_dev

        # Compute the simplex each feature lies in
        # ->> Numpify 
        # Shape of feature (N, d)
        
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        # ->> Numpify
        cf = feature * scale_factor[:, np.newaxis] # (d, N)
        E = np.vstack([
            np.ones((self.d_, ), dtype=np.float32), 
            np.diag(-np.arange(self.d_, dtype=np.float32) - 2) + np.triu(np.ones((self.d_, self.d_), dtype=np.float32))]) # (d + 1, d)
        elevated = np.matmul(E, cf) # (d, N)

        # Find the closest 0-colored simplex through rounding
        # ->> Numpify
        down_factor = 1. / (self.d_ + 1)
        up_factor = self.d_ + 1
        _sum = np.zeros((self.N_, ), dtype=np.int32)
        v = down_factor * elevated # (d, N)
        up = np.ceil(v) * up_factor # (d, N)
        down = np.floor(v) * up_factor # (d, N)
        rem0 = np.where(up - elevated < elevated - down, up, down).astype(np.float32) # (d, N)
        _sum[:] = int(rem0.sum(axis=0) * down_factor) # (N, )

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        # ->> Numpify
        rank.fill(0)
        ds = elevated - rem0 # (d, N)
        for i in range(self.d_):
            di = ds[i:i + 1] # (1, N)
            dj = ds[list(range(i + 1, self.d_ + 1))] # (i + 1:d + 1, N)
            rank[:]
            di < dj
        # ->> ABORTED
        
        # If the point doesn't lie on the plane (sum != 0) bring it back
        # ->> Numpify
        rank += _sum
        ls_zero = rank < 0
        gt_d = rank > self.d_
        rank[ls_zero] += self.d_ + 1
        rem0[ls_zero] += self.d_ + 1
        rank[gt_d] -= self.d_ + 1
        rem0[gt_d] -= self.d_ + 1

        # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        # ->> Numpify
        barycentric.fill(0)
        v = (elevated - rem0) * down_factor
        barycentric[self.d_ - rank] += v
        barycentric[self.d_ - rank + 1] -= v
        # Wrap around
        barycentric[0] += 1. + barycentric[self.d_ + 1]

        # Compute all vertices and their offset
        # ->> Numpify
        key = rem0 + canonical[remainder * (self.d_ + 1) + rank[:self.d_]]
        self.offset_[:, remainder] = self.hash_table.find(key, True)
        self.rank_[:, remainder] = rank[remainder]
        self.barycentric_[:, ]


