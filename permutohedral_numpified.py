''' 
This is a python implementation of permutohedral lattice in Dense CRF
->> 2021.10.08 Removing loops
->> 2022.05.27 Numpify `init()`, create `np_init()`
->> 2022.05.31 Remove comments
->> 2022.05.31 Try to remove hash table
'''

import numpy as np


class HashTable:

    def __init__(self, key_size, n_elements):
        self.key_size_ = key_size
        self.filled_ = 0
        self.capacity_ = 2 * n_elements
        self.keys_ = np.zeros(((self.capacity_ // 2 + 10) * self.key_size_, ),
                              dtype=np.short)
        self.table_ = np.ones((2 * n_elements, ), dtype=np.int32) * -1

    def grow(self):
        # Create the new memory and copy the values in
        print('Hashtable grows...')
        old_capacity = self.capacity_
        self.capacity_ *= 2
        old_keys = np.zeros(((old_capacity + 10) * self.key_size_, ),
                            dtype=np.short)
        old_keys[:(old_capacity // 2 + 10) * self.key_size_] = self.keys_
        old_table = np.ones((self.capacity_, ), dtype=np.int32) * -1

        # Swap the memory
        self.table_, old_table = old_table, self.table_
        self.keys_, old_keys = old_keys, self.keys_

        # Reinsert each element
        for i in range(old_capacity):
            if old_table[i] >= 0:
                e = old_table[i]
                h = self.hash(self.get_key(e)) % self.capacity_
                while self.table_[h] >= 0:
                    if h < self.capacity_ - 1:
                        h = h + 1
                    else:
                        h = 0
                self.table_[h] = e

    def hash(self, k):
        # r = np.int64(0)
        r = np.uint(0)
        for i in range(self.key_size_):
            r += k[i]
            r *= 1664525
        return r

    def size(self):
        return self.filled_

    def reset(self):
        self.filled_ = 0
        self.table_.fill(-1)

    def find(self, k, create=False):
        if self.capacity_ <= 2 * self.filled_:
            self.grow()
        # Get the hash value
        h = self.hash(k) % self.capacity_
        # Find the element with the right key, using linear probing
        while True:
            e = self.table_[h]
            if e == -1:
                if create:
                    # Insert a new key and return the new id
                    self.keys_[self.filled_ *
                               self.key_size_:self.filled_ * self.key_size_ +
                               self.key_size_] = k[:self.key_size_]
                    self.table_[h] = self.filled_
                    self.filled_ += 1
                    return self.table_[h]
                else:
                    return -1
            # Check if the current key is The One
            good = np.all(self.keys_[e * self.key_size_:e * self.key_size_ +
                                     self.key_size_] == k[:self.key_size_])
            if good:
                return e
            # Continue searching
            h += 1
            if h == self.capacity_:
                h = 0

    def get_key(self, i):
        return self.keys_[i * self.key_size_:]


class Permutohedral:

    def __init__(self, N, d):
        self.N_, self.M_, self.d_ = N, 0, d
        self.blur_neighbors_ = None
        self.hash_table = HashTable(d, N * (d + 1))
        # ->> Numpified
        self.offset_ = np.zeros((N, (d + 1)), dtype=np.int32)
        self.rank_ = np.zeros((N, (d + 1)), dtype=np.int32)
        self.barycentric_ = np.zeros((N, (d + 1)), dtype=np.float32)

    def init(self, feature):
        # ->> Numpification test
        # ->> Note that the shape of `feature` is (N, d), channel-last

        # Compute the lattice coordinates for each feature [there is going to be a lot of magic here
        # pass

        # Allocate the class memory
        # pass

        # ->> 2022.05.27 Allocate the local memory
        scale_factor = np.zeros((self.d_, ), dtype=np.float32)
        elevated = np.zeros((self.N_, self.d_ + 1), dtype=np.float32)
        rem0 = np.zeros((self.N_, self.d_ + 1), dtype=np.float32)
        barycentric = np.zeros((self.N_, self.d_ + 2), dtype=np.float32)
        rank = np.zeros((self.N_, self.d_ + 1), dtype=np.short)
        canonical = np.zeros((self.d_ + 1, self.d_ + 1), dtype=np.short)
        key = np.zeros((self.N_, self.d_ + 1, self.d_ + 1), dtype=np.short)
        _sum = np.zeros((self.N_, ), dtype=np.int32)

        # Compute the canonical simplex, (d + 1, d + 1)
        for i in range(self.d_ + 1):
            canonical[i, :self.d_ + 1 - i] = i
            canonical[i, self.d_ + 1 - i:] = i - (self.d_ + 1)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2. / 3.) * (self.d_ + 1)
        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor[:] = 1. / np.sqrt(
            (np.arange(self.d_) + 2) *
            (np.arange(self.d_) + 1)) * inv_std_dev  # (d, )

        # Compute the simplex each feature lies in
        # !!! Shape of feature (N, d)
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = feature * scale_factor[np.newaxis, ...]  # (N, d)
        E = np.vstack([
            np.ones((self.d_, ), dtype=np.float32),
            np.diag(-np.arange(self.d_, dtype=np.float32) - 2) +
            np.triu(np.ones((self.d_, self.d_), dtype=np.float32))
        ])  # (d + 1, d)
        elevated[:] = np.matmul(cf, E.T)  # (N, d + 1)

        # Find the closest 0-colored simplex through rounding
        # ->> Numpify
        down_factor = 1. / (self.d_ + 1)
        up_factor = self.d_ + 1
        v = down_factor * elevated  # (N, d + 1)
        up = np.ceil(v) * up_factor  # (N, d + 1)
        down = np.floor(v) * up_factor  # (N, d + 1)
        rem0[:] = np.where(up - elevated < elevated - down, up,
                           down).astype(np.float32)  # (N, d + 1)
        _sum[:] = (rem0.sum(axis=1) * down_factor).astype(np.int32)  # (N, )

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        rank.fill(0)  # (N, d + 1)
        ds = elevated - rem0  # (N, d + 1)
        valid = 1 - np.tril(np.ones(
            (self.d_ + 1, self.d_ + 1), dtype=np.short))
        di = ds[..., np.newaxis]  # (N, d + 1, 1)
        dj = ds[..., np.newaxis, :]  # (N, 1, d + 1)
        rank += ((di < dj) * valid[np.newaxis, ...]).sum(axis=2)  # (N, d + 1)
        rank += ((di >= dj) * valid[np.newaxis, ...]).sum(axis=1)  # (N, d + 1)

        # If the point doesn't lie on the plane (sum != 0) bring it back
        rank += _sum[..., np.newaxis]  # (N, d + 1)
        ls_zero = rank < 0
        gt_d = rank > self.d_
        rank[ls_zero] += self.d_ + 1
        rem0[ls_zero] += self.d_ + 1
        rank[gt_d] -= self.d_ + 1
        rem0[gt_d] -= self.d_ + 1

        # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        barycentric.fill(0)  # (N, d + 2)
        vs = (elevated - rem0) * down_factor  # (N, d + 1)
        barycentric = barycentric.reshape(-1)  # (N x (d + 2), )
        idx = (self.d_ -
               rank) + np.arange(self.N_)[..., np.newaxis] * (self.d_ + 2)
        idx1 = (self.d_ - rank +
                1) + np.arange(self.N_)[..., np.newaxis] * (self.d_ + 2)
        barycentric[idx] += vs
        barycentric[idx1] -= vs
        barycentric = barycentric.reshape((self.N_, self.d_ + 2))
        barycentric[..., 0] += 1. + barycentric[..., self.d_ + 1]

        # Compute all vertices and their offset
        # ->> Numpification test
        canonicalT = np.transpose(canonical, (1, 0))  # (d + 1,  d + 1)
        canonical_ext = canonicalT[rank]  # (N, d + 1, d + 1)
        canonical_ext = np.transpose(canonical_ext,
                                     (0, 2, 1))  # (N, d + 1, d + 1)
        key[..., :self.d_] = rem0[..., np.newaxis, :self.d_] + canonical_ext[
            ..., :self.d_]

        for k in range(self.N_):
            for remainder in range(self.d_ + 1):
                self.offset_[k, remainder] = self.hash_table.find(
                    key[k, remainder], True)
        self.rank_[:] = rank  # (N, d + 1)
        self.barycentric_[:] = barycentric[..., :self.d_ + 1]  # (N, d + 1)

        del scale_factor, elevated, rem0, barycentric, rank, canonical, key, _sum

        # Find the neighbors of each lattice point

        # Get the number of vertices in the lattice
        self.M_ = self.hash_table.size()

        # Create the neighborhood structure
        self.blur_neighbors_ = np.zeros((self.d_ + 1, self.M_, 2),
                                        dtype=np.int32)

        # For each of d+1 axes,
        # ->> Numpification test
        n1s = np.zeros((self.M_, self.d_ + 1, self.d_ + 1),
                       dtype=np.short)  # (M, d + 1, d + 1)
        n2s = np.zeros((self.M_, self.d_ + 1, self.d_ + 1),
                       dtype=np.short)  # (M, d + 1, d + 1)
        hash_keys = self.hash_table.keys_[:self.hash_table.key_size_ *
                                          self.hash_table.filled_]
        hash_keys = hash_keys.reshape(
            (self.hash_table.filled_, self.hash_table.key_size_))  # (M, d)
        overflow_keys = np.concatenate([hash_keys[1:, 0], [0]])  # (M, )
        n1s[..., :self.d_] = hash_keys[:, np.newaxis, :] - 1
        n2s[..., :self.d_] = hash_keys[:, np.newaxis, :] + 1
        ds = np.zeros((self.d_ + 1, ), dtype=np.short)
        ds.fill(self.d_)  # (d + 1, )
        ds = np.diag(ds)  # (d + 1, d + 1)
        diagone = np.diag(np.ones(self.d_ + 1, dtype=np.short))
        n1s += ds[np.newaxis, ...] + diagone
        n2s -= ds[np.newaxis, ...] + diagone
        n1s[:, self.d_, self.d_] = overflow_keys + self.d_
        n2s[:, self.d_, self.d_] = overflow_keys - self.d_

        # ->> Loop exchanged
        for i in range(self.M_):
            for j in range(self.d_ + 1):
                self.blur_neighbors_[j, i, 0] = self.hash_table.find(n1s[i, j])
                self.blur_neighbors_[j, i, 1] = self.hash_table.find(n2s[i, j])

        del n1s, n2s, ds, diagone

    def seq_compute(self, out, inp, value_size, reverse):
        '''
        Compute sequentially

        Args:
            inp: (size, value_size)
            value_size: value size
            reverse: indicating the blurring order
        
        Returns:
            out: (size, value_size)
        '''

        # **************************
        # * 2022-05-26: Numpifying *
        # **************************
        # Shift all values by 1 such that -1 -> 0 (used for blurring)
        values = np.zeros(((self.M_ + 2), value_size), dtype=np.float32)
        new_values = np.zeros(((self.M_ + 2), value_size), dtype=np.float32)

        # ->> Splat
        os = self.offset_.reshape(-1) + 1  # (N x (d + 1), )
        ws = self.barycentric_.reshape(-1)  # (N x (d + 1), )

        for vs in range(value_size):
            inp_flat = np.broadcast_to(inp[:, vs][..., np.newaxis],
                                       (self.N_, self.d_ + 1))
            inp_flat = inp_flat.reshape(-1)
            values[:, vs] = np.bincount(os,
                                        weights=inp_flat * ws,
                                        minlength=self.M_ + 2)

        # ->> Blur
        j_range = range(self.d_, -1, -1) if reverse else range(self.d_ + 1)
        for j in j_range:
            old_vals = values[1:self.M_ + 1]
            new_vals = new_values[1:self.M_ + 1]
            n1s = self.blur_neighbors_[j, :self.M_, 0] + 1
            n2s = self.blur_neighbors_[j, :self.M_, 1] + 1
            n1_vals = values[n1s]
            n2_vals = values[n2s]

            new_vals[:] = old_vals + .5 * (n1_vals + n2_vals)

            values, new_values = new_values, values

        # ->> Slice
        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        alpha = 1. / (1 + np.power(2., -self.d_))

        out *= 0
        out[:] = (ws[..., np.newaxis] * values[os] * alpha).reshape(
            (self.N_, self.d_ + 1, value_size)).sum(axis=1)  # (N, vs)

        del values, new_values

    def compute(self, inp, reverse=False):
        size, ch = inp.shape
        out = np.zeros_like(inp)
        self.seq_compute(out, inp, ch, reverse)
        out = out.reshape((size, ch))
        return out