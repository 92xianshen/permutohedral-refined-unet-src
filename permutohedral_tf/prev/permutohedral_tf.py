"""
This is a tf implementation of permutohedral lattice.
"""

import numpy as np
import tensorflow as tf

enable_float64 = 0
tf_float = tf.float64 if enable_float64 else tf.float32

class Permutohedral:

    def __init__(self, N: np.int32, d: np.int32) -> None:
        self.N_, self.M_, self.d_ = (
            tf.constant(N, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(d, dtype=tf.int32),
        )

        canonical = np.zeros((d + 1, d + 1), dtype=np.int32)
        for i in range(d + 1):
            canonical[i, :d + 1 - i] = i
            canonical[i, d + 1 - i:] = i - (d + 1)
        self.canonical = tf.constant(canonical, dtype=tf.int32)

        E = np.vstack([
            np.ones((d, ), dtype=np.float32),
            np.diag(-np.arange(d, dtype=np.float32) - 2) +
            np.triu(np.ones((d, d), dtype=np.float32)),
        ])  # (d + 1, d)
        self.E = tf.constant(E, dtype=tf_float)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * (d + 1)
        self.inv_std_dev = tf.constant(inv_std_dev, dtype=tf_float)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (1.0 / np.sqrt(
            (np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev)  # (d, )
        self.scale_factor = tf.constant(scale_factor, dtype=tf_float)

        valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32))
        self.valid = tf.constant(valid, dtype=tf.int32)

        ds = np.ones((d + 1, ), dtype=np.short) * d
        ds = np.diag(ds)  # (d + 1, d + 1)
        diagone = np.diag(np.ones(d + 1, dtype=np.short))  # (d + 1, d + 1)
        self.ds = tf.constant(ds, dtype=tf.int32)
        self.diagone = tf.constant(diagone, dtype=tf.int32)

        self.blur_neighbors_, self.offset_, self.rank_, self.barycentric_ = (
            None,
            None,
            None,
            None,
        )
        # self.hash_table = HashTable(self.d_, self.N_ * (self.d_ + 1))

    def init(self, feature):
        # Compute the simplex each feature lies in
        # !!! Shape of feature [N, d]
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = feature * self.scale_factor[tf.newaxis, ...]  # [N, d]
        elevated = tf.matmul(cf, tf.transpose(self.E))  # [N, d + 1]

        # Find the closest 0-colored simplex through rounding
        down_factor = tf.constant(1.0 / tf.cast(self.d_ + 1, dtype=tf_float),
                                  dtype=tf_float)
        up_factor = tf.constant(tf.cast(self.d_ + 1, dtype=tf_float),
                                dtype=tf_float)
        v = down_factor * elevated  # [N, d + 1]
        up = tf.math.ceil(v) * up_factor  # [N, d + 1]
        down = tf.math.floor(v) * up_factor  # [N, d + 1]
        rem0 = tf.cast(tf.where(up - elevated < elevated - down, up, down),
                       dtype=tf_float)  # [N, d + 1]
        _sum = tf.cast(tf.reduce_sum(rem0, axis=1) * down_factor,
                       dtype=tf.int32)  # [N, ]

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        rank = tf.zeros(shape=[self.N_, self.d_ + 1],
                        dtype=tf.int32)  # [N, d + 1]
        ds = elevated - rem0  # [N, d + 1]
        di = ds[..., tf.newaxis]  # [N, d + 1, 1]
        dj = ds[..., tf.newaxis, :]  # [N, 1, d + 1]
        di_lt_dj = tf.where(di < dj, 1, 0)
        di_geq_dj = tf.where(di >= dj, 1, 0)
        rank = rank + tf.reduce_sum(di_lt_dj * self.valid[tf.newaxis, ...],
                                    axis=2)  # [N, d + 1]
        rank = rank + tf.reduce_sum(di_geq_dj * self.valid[tf.newaxis, ...],
                                    axis=1)  # [N, d + 1]

        # If the point doesn't lie on the plane (sum != 0) bring it back
        rank = rank + _sum[..., tf.newaxis]  # (N, d + 1)
        ls_zero = rank < 0
        gt_d = rank > self.d_
        rank = tf.where(ls_zero, rank + self.d_ + 1, rank)
        rem0 = tf.where(ls_zero, rem0 + tf.cast(self.d_ + 1, dtype=tf_float),
                        rem0)
        rank = tf.where(gt_d, rank - (self.d_ + 1), rank)
        rem0 = tf.where(gt_d, rem0 - tf.cast(self.d_ + 1, dtype=tf_float),
                        rem0)

        # # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        barycentric = tf.zeros(shape=[
            self.N_ * (self.d_ + 2),
        ],
                               dtype=tf_float)  # [N x (d + 2), ]
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
        canonical_ext = tf.gather(canonicalT,
                                  tf.cast(rank,
                                          dtype=tf.int32))  # [N, d + 1, d + 1]
        canonical_ext = tf.transpose(canonical_ext,
                                     perm=[0, 2, 1])  # [N, d + 1, d + 1]
        key = (tf.cast(rem0[..., tf.newaxis, :self.d_], dtype=tf.int32) +
               canonical_ext[..., :self.d_])  # [N, d + 1, d]
        key = tf.concat(
            [key, tf.zeros([self.N_, self.d_ + 1, 1], dtype=tf.int32)],
            axis=-1)  # [N, d + 1, d + 1]

        # ->> 2022.07.01 Hash.
        # Keys in string format.
        flat_key = tf.reshape(key, shape=[-1,
                                          self.d_ + 1])  # [N x (d + 1), d + 1]
        hash_keys, _ = tf.raw_ops.UniqueV2(x=flat_key, axis=[
            0,
        ])  # [M, d + 1]
        skey = tf.strings.reduce_join(tf.strings.as_string(flat_key),
                                      axis=-1,
                                      separator=',')  # [N x (d + 1), ]
        uniq_skey = tf.strings.reduce_join(tf.strings.as_string(hash_keys),
                                           axis=-1,
                                           separator=',')  # [M, ]
        n_skey = tf.shape(hash_keys)[0]  # Get M
        self.hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(uniq_skey, tf.range(n_skey)),
            default_value=-1)
        offset = self.hash_table.lookup(skey)  # [N x (d + 1), ]
        self.offset_ = tf.constant(offset, dtype=tf.int32)
        self.rank_ = rank  # [N, d + 1]
        self.barycentric_ = barycentric[..., :self.d_ + 1]  # [N, d + 1]

        # Find the neighbors of each lattice point
        # Get the number of vertices in the lattice
        # Create the neighborhood structure
        # For each of d+1 axes,
        self.M_ = n_skey
        hkeys = hash_keys[..., :self.d_]  # [M, d]
        overflow_keys = tf.concat([hash_keys[1:, 0], [0]], axis=0)  # [M, ]
        n1s = tf.tile(hkeys[:, tf.newaxis, :],
                      [1, self.d_ + 1, 1]) - 1  # [M, d + 1, d]
        n2s = tf.tile(hkeys[:, tf.newaxis, :],
                      [1, self.d_ + 1, 1]) + 1  # [M, d + 1, d]
        n1s = tf.concat(
            [n1s, tf.zeros([self.M_, self.d_ + 1, 1], dtype=tf.int32)],
            axis=-1)  # [M, d + 1, d + 1]
        n2s = tf.concat(
            [n2s, tf.zeros([self.M_, self.d_ + 1, 1], dtype=tf.int32)],
            axis=-1)  # [M, d + 1, d + 1]
        n1s = n1s + self.ds[tf.newaxis, ...] + self.diagone[tf.newaxis, ...]
        n2s = n2s + self.ds[tf.newaxis, ...] + self.diagone[tf.newaxis, ...]
        n1s = tf.concat([
            n1s[..., :self.d_],
            tf.tile(overflow_keys[..., tf.newaxis, tf.newaxis] + self.d_,
                    [1, self.d_ + 1, 1])
        ],
                        axis=-1)  # [M, d + 1, d + 1]
        n2s = tf.concat([
            n2s[..., :self.d_],
            tf.tile(overflow_keys[..., tf.newaxis, tf.newaxis] - self.d_,
                    [1, self.d_ + 1, 1])
        ],
                        axis=-1)  # [M, d + 1, d + 1]
        n1s_str = tf.strings.reduce_join(tf.strings.as_string(n1s),
                                         axis=-1,
                                         separator=',')  # [M, d + 1]
        n2s_str = tf.strings.reduce_join(tf.strings.as_string(n2s),
                                         axis=-1,
                                         separator=',')  # [M, d + 1]
        blur_neighbors0 = self.hash_table.lookup(n1s_str)  # [M, d + 1]
        blur_neighbors1 = self.hash_table.lookup(n2s_str)  # [M, d + 1]
        blur_neighbors = tf.stack([blur_neighbors0, blur_neighbors1],
                                  axis=-1)  # [M, d + 1, 2]
        self.blur_neighbors_ = tf.transpose(blur_neighbors,
                                            perm=[1, 0, 2])  # [d + 1, M, 2]

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

        inpT = tf.transpose(
            inp, perm=[1, 0])  # transpose to channel-first. [value_size, N]

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
                dtype=tf_float,
            )
            return val_ch

        valuesT = tf.vectorized_map(splat_channelwise, inpT)
        values = tf.transpose(valuesT, perm=[1, 0])
        new_values = tf.zeros([self.M_ + 2, value_size], dtype=tf_float)

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
            new_values = tf.concat(
                [new_values[0:1], new_vals, new_values[self.M_ + 1:]], axis=0)

            values, new_values = new_values, values

        # ->> Slice
        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        alpha = 1.0 / (1 + tf.pow(2.0, tf.cast(-self.d_, dtype=tf.float32)))

        out = ws[..., tf.newaxis] * tf.gather(values, os) * alpha
        out = tf.reshape(out, shape=[self.N_, self.d_ + 1, value_size])
        out = tf.reduce_sum(out, axis=1)

        return out

    def compute(self, inp, reverse=False):
        size, ch = tf.shape(inp)
        out = self.seq_compute(inp, ch, reverse)
        return out
