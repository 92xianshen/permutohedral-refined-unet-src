"""
This is a tf implementation of permutohedral lattice.
"""

import numpy as np
import tensorflow as tf


class HashTable:
    """
    Implement in TensorFlow.
    Keep loop structures until get a replacement.
    """

    def __init__(self, key_size: np.uint64, n_elements: np.uint64) -> None:
        self.key_size_ = tf.constant(key_size, dtype=tf.uint64)
        self.filled_ = tf.constant(0, dtype=tf.uint64)
        self.capacity_ = tf.constant(2 * n_elements, dtype=tf.uint64)
        self.keys_ = tf.Variable(
            tf.zeros(
                [
                    (self.capacity_ // 2 + 10) * self.key_size_,
                ],
                dtype=tf.int16,
            ),
            trainable=False,
        )
        self.table_ = tf.Variable(
            tf.ones(
                [
                    2 * n_elements,
                ],
                dtype=tf.int32,
            ) * -1,
            trainable=False,
        )

    def grow(self):
        # Create the new memory and copy the values in
        tf.print("Hashtable grows...")
        old_capacity = self.capacity_
        self.capacity_ *= 2
        old_keys = tf.Variable(
            tf.zeros(
                [
                    (old_capacity + 10) * self.key_size_,
                ],
                dtype=tf.int16,
            ),
            trainable=False,
        )
        old_keys[:(old_capacity // 2 + 10) * self.key_size_].assign(self.keys_)
        old_table = tf.Variable(
            tf.ones(
                [
                    self.capacity_,
                ],
                dtype=tf.int32,
            ) * -1,
            trainable=False,
        )

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
                self.table_[h].assign(e)

        del old_table, old_keys

    def hash(self, k):
        r = tf.Variable(0, dtype=tf.uint64, trainable=False)
        for i in range(self.key_size_):
            r.assign_add(k[i])
            r.assign(r * 1664525)
        return tf.constant(r, dtype=tf.uint64)

    def size(self):
        return self.filled_

    def reset(self):
        self.filled_ = tf.constant(0, dtype=tf.uint64)
        self.table_.assign(tf.ones_like(self.table_) * -1)

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
                               self.key_size_].assign(k[:self.key_size_])
                    self.table_[h].assign(self.filled_)
                    self.filled_ += 1
                    return self.table_[h]
                else:
                    return -1
            # Check if the current key is The One
            good = tf.reduce_all(
                self.keys_[e * self.key_size_:e * self.key_size_ +
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

    def __init__(self, N: np.int32, d: np.int32) -> None:
        self.N_, self.M_, self.d_ = (
            tf.constant(N, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(d, dtype=tf.int32),
        )

        canonical = np.zeros((d + 1, d + 1), dtype=np.short)
        for i in range(d + 1):
            canonical[i, :d + 1 - i] = i
            canonical[i, d + 1 - i:] = i - (d + 1)
        self.canonical = tf.constant(canonical, dtype=tf.int16)

        E = np.vstack([
            np.ones((d, ), dtype=np.float32),
            np.diag(-np.arange(d, dtype=np.float32) - 2) +
            np.triu(np.ones((d, d), dtype=np.float32)),
        ])  # (d + 1, d)
        self.E = tf.constant(E, dtype=tf.float32)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * (d + 1)
        self.inv_std_dev = tf.constant(inv_std_dev, dtype=tf.float32)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (1.0 / np.sqrt(
            (np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev)  # (d, )
        self.scale_factor = tf.constant(scale_factor, dtype=tf.float32)

        valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.short))
        self.valid = tf.constant(valid, dtype=tf.int16)

        ds = np.ones((d + 1, ), dtype=np.short) * d
        ds = np.diag(ds)  # (d + 1, d + 1)
        diagone = np.diag(
            np.ones(d + 1, ),
            dtype=np.short,
        ) # (d + 1, d + 1)
        self.ds = tf.constant(ds, dtype=tf.int16)
        self.diagone = tf.constant(diagone, dtype=tf.int16)

        self.blur_neighbors_, self.offset_, self.rank_, self.barycentric_ = (
            None,
            None,
            None,
            None,
        )
        self.hash_table = HashTable(self.d_, self.N_ * (self.d_ + 1))

    def init(self, feature):
        # Compute the simplex each feature lies in
        # !!! Shape of feature [N, d]
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = feature * self.scale_factor[tf.newaxis, ...]  # [N, d]
        elevated = tf.matmul(cf, tf.transpose(self.E))  # [N, d + 1]

        # Find the closest 0-colored simplex through rounding
        down_factor = tf.constant(1.0 / (self.d_ + 1), dtype=tf.float32)
        up_factor = tf.constant(self.d_ + 1, dtype=tf.float32)
        v = down_factor * elevated  # [N, d + 1]
        up = tf.math.ceil(v) * up_factor  # [N, d + 1]
        down = tf.math.floor(v) * up_factor  # [N, d + 1]
        rem0 = tf.cast(tf.where(up - elevated < elevated - down, up, down),
                       dtype=tf.float32)  # [N, d + 1]
        _sum = tf.cast(tf.reduce_sum(rem0, axis=1) * down_factor,
                       dtype=tf.int32)  # [N, ]

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        rank = tf.zeros(shape=[self.N_, self.d_ + 1],
                        dtype=tf.int16)  # [N, d + 1]
        ds = elevated - rem0  # [N, d + 1]
        di = ds[..., tf.newaxis]  # [N, d + 1, 1]
        dj = ds[..., tf.newaxis, :]  # [N, 1, d + 1]
        rank += tf.reduce_sum((di < dj) * self.valid[tf.newaxis, ...],
                              axis=2)  # [N, d + 1]
        rank += tf.reduce_sum((di >= dj) * self.valid[tf.newaxis, ...],
                              axis=1)  # [N, d + 1]

        # If the point doesn't lie on the plane (sum != 0) bring it back
        rank += _sum[..., tf.newaxis]  # (N, d + 1)
        ls_zero = rank < 0
        gt_d = rank > self.d_
        rank = tf.where(ls_zero, rank + self.d_ + 1, rank)
        rem0 = tf.where(ls_zero, rem0 + self.d_ + 1, rem0)
        rank = tf.where(gt_d, rank - self.d_ - 1, rank)
        rem0 = tf.where(gt_d, rem0 - self.d_ - 1, rem0)

        # # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        # barycentric = tf.zeros(
        #     shape=[self.N_, self.d_ + 2], dtype=tf.float32
        # )  # [N, d + 2]
        # vs = (elevated - rem0) * down_factor  # [N, d + 1]
        # barycentric = tf.reshape(barycentric, shape=[-1, ])  # [N x (d + 2), ]
        # idx = (self.d_ - rank) + tf.range(self.N_)[..., tf.newaxis] * (self.d_ + 2)
        # idx1 = (self.d_ - rank + 1) + tf.range(self.N_)[..., tf.newaxis] * (self.d_ + 2)
        # barycentric = tf.gather(barycentric, idx) + vs
        # barycentric = tf.gather(barycentric, idx1) - vs
        # barycentric = tf.reshape(barycentric, shape=[self.N_, self.d_ + 2])
        # barycentric[..., 0] += 1.0 + barycentric[..., self.d_ + 1]
        barycentric = tf.zeros(shape=[
            self.N_ * (self.d_ + 2),
        ],
                               dtype=tf.float32)  # [N x (d + 2), ]
        vs = tf.reshape((elevated - rem0) * down_factor, shape=[
            -1,
        ])  # [N x (d + 1), ]
        idx = tf.reshape((self.d_ - rank) +
                         tf.range(self.N_)[..., tf.newaxis] * (self.d_ + 2),
                         shape=[
                             -1,
                         ])  # [N x (d + 1), ]
        idx1 = tf.reshape((self.d_ - rank + 1) +
                          tf.range(self.N_)[..., tf.newaxis] * (self.d_ + 2),
                          shape=[
                              -1,
                          ])  # [N x (d + 1), ]
        barycentric = barycentric + tf.math.bincount(
            idx,
            weights=vs,
            minlength=self.N_ * (self.d_ + 2),
            maxlength=self.N_ * (self.d_ + 2))  # [N x (d + 2), ]
        barycentric = barycentric - tf.math.bincount(
            idx1,
            weights=vs,
            minlength=self.N_ * (self.d_ + 2),
            maxlength=self.N_ * (self.d_ + 2))  # [N x (d + 2), ]
        barycentric = tf.reshape(barycentric,
                                 shape=[self.N_, (self.d_ + 2)])  # [N, d + 2]
        barycentric0 = barycentric[..., 0] + 1. + barycentric[..., self.d_ +
                                                              1]  # [N, ]
        barycentric = tf.concat(
            [barycentric0[..., tf.newaxis], barycentric[..., 1:]],
            axis=-1)  # [N, d + 2]

        # Compute all vertices and their offset
        canonicalT = tf.transpose(self.canonical, perm=[1,
                                                        0])  # (d + 1, d + 1)
        canonical_ext = tf.gather(canonicalT, rank)  # [N, d + 1, d + 1]
        canonical_ext = tf.transpose(canonical_ext,
                                     perm=[0, 2, 1])  # [N, d + 1, d + 1]
        key = (rem0[..., tf.newaxis, :self.d_] + canonical_ext[..., :self.d_]
               )  # [N, d + 1, d]
        key = tf.concat([key, tf.zeros([self.N_, self.d_ + 1, 1])],
                        axis=-1)  # [N, d + 1, d + 1]

        offset = tf.Variable(tf.zeros([self.N_, self.d_ + 1], dtype=tf.int32),
                             trainable=False)
        for k in range(self.N_):
            for remainder in range(self.d_ + 1):
                offset[k, remainder].assign(
                    self.hash_table.find(key[k, remainder], True))
        self.offset_ = tf.constant(offset)
        self.rank_ = rank  # [N, d + 1]
        self.barycentric_ = barycentric[..., :self.d_ + 1]  # [N, d + 1]

        # Find the neighbors of each lattice point
        # Get the number of vertices in the lattice
        self.M_ = self.hash_table.size()

        # Create the neighborhood structure
        blur_neighbors = tf.Variable(tf.zeros([self.d_ + 1, self.M_, 2],
                                              dtype=tf.int32),
                                     trainable=False)

        # For each of d+1 axes,
        n1s = tf.Variable(
            tf.zeros([self.M_, self.d_ + 1, self.d_ + 1], dtype=tf.int16),
            trainable=False,
        )  # [M, d + 1, d + 1]
        n2s = tf.Variable(
            tf.zeros([self.M_, self.d_ + 1, self.d_ + 1], dtype=tf.int16),
            trainable=False,
        )  # [M, d + 1, d + 1]
        hash_keys = self.hash_table.keys_[:self.hash_table.key_size_ *
                                          self.hash_table.filled_]
        hash_keys = tf.reshape(
            hash_keys,
            shape=[self.hash_table.filled_,
                   self.hash_table.key_size_])  # [M, d]
        overflow_keys = tf.concat([hash_keys[1:, 0], [0]], axis=0)  # [M, ]
        n1s[..., :self.d_].assign(hash_keys[:, tf.newaxis, :] - 1)
        n2s[..., :self.d_].assign(hash_keys[:, tf.newaxis, :] + 1)
        n1s.assign_add(self.ds[tf.newaxis, ...] + self.diagone[tf.newaxis, ...])
        n2s.assign_sub(self.ds[tf.newaxis, ...] + self.diagone[tf.newaxis, ...])
        n1s[:, self.d_, self.d_].assign(overflow_keys + self.d_)
        n2s[:, self.d_, self.d_].assign(overflow_keys - self.d_)

        for i in range(self.M_):
            for j in range(self.d_ + 1):
                blur_neighbors[j, i, 0].assign(self.hash_table.find(n1s[i, j]))
                blur_neighbors[j, i, 1].assign(self.hash_table.find(n2s[i, j]))

        self.blur_neighbors_ = tf.constant(blur_neighbors)

    def seq_compute(self, inp, value_size, reverse):
        """
        Compute sequentially.

        Args:
            inp: [size, value_size], channel-last.
            value_size: value size.
            reverse: indicating the blur order.

        Returns:
            out: [size, value_size]
        """

        # **************************
        # * 2022-05-26: Numpifying *
        # **************************
        # Shift all values by 1 such that -1 -> 0 (used for blurring)
        # values, new_values = None, None

        # ->> Splat
        os = (tf.reshape(
            self.offset_,
            [
                -1,
            ],
        ) + 1)  # [N X (d + 1), ]
        ws = tf.reshape(
            self.barycentric_,
            [
                -1,
            ],
        )  # [N x (d + 1), ]

        inpT = tf.transpose(inp, perm=[1, 0])  # transpose to channel-first. [value_size, N]

        def splat_channelwise(ch):
            ch_ext = tf.tile(ch[..., tf.newaxis],
                              [1, self.d_ + 1])  # [N, (d + 1)]
            ch_flat = tf.reshape(
                ch_ext,
                [
                    -1,
                ],
            )  # [N x (d + 1), ]
            val_ch = tf.math.bincount(
                os,
                weights=ch_flat * ws,
                minlength=self.M_ + 2,
                maxlength=self.M_ + 2,
                dtype=tf.float32,
            )
            return val_ch

        valuesT = tf.vectorized_map(splat_channelwise, inpT)
        values = tf.transpose(valuesT, perm=[1, 0])
        new_values = tf.zeros([self.M_ + 2, value_size], dtype=tf.float32)

        # ->> Blur
        j_range = tf.range(self.d_, -1, -1) if reverse else tf.range(self.d_ +
                                                                     1)
        for j in j_range:
            old_vals = values[1:self.M_ + 1]
            # new_vals = new_values[1:self.M_ + 1]
            n1s = self.blur_neighbors_[j, :self.M_, 0] + 1
            n2s = self.blur_neighbors_[j, :self.M_, 1] + 1
            n1_vals = tf.gather(values, n1s)
            n2_vals = tf.gather(values, n2s)

            new_vals = old_vals + 0.5 * (n1_vals + n2_vals)
            new_values = tf.concat(new_values[0:1],
                                   new_vals,
                                   new_values[self.M_ + 1:],
                                   axis=0)

            values, new_values = new_values, values

        # ->> Slice
        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        alpha = 1.0 / (1 + tf.pow(2.0, -self.d_))

        out = ws[..., tf.newaxis] * tf.gather(values, os) * alpha
        out = tf.reshape(out, shape=[self.N_, self.d_ + 1, value_size])
        out = tf.reduce_sum(out, axis=1)

        return out

    def compute(self, inp, reverse=False):
        size, ch = tf.shape(inp)
        out = self.seq_compute(inp, ch, reverse)
        return out
