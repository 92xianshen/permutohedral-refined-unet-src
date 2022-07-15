'''
- TF implementation of permutohedral lattice, channel-last as well.
- `tf.float32` and `tf.int32` as default float and integer types, respectively.
- Refine the source code
- Update of v3: refine the key size, d + 1 ->> d
'''

import numpy as np
import tensorflow as tf


class Permutohedral(tf.Module):

    def __init__(self, N, d) -> None:
        super().__init__()
        self.N_, self.M_, self.d_ = N, 0, d

        canonical = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
        for i in range(d + 1):
            canonical[i, :d + 1 - i] = i
            canonical[i, d + 1 - i:] = i - (d + 1)
        self.canonical = tf.constant(canonical,
                                     dtype=tf.int32)  # [d + 1, d + 1]

        E = np.vstack([
            np.ones((d, ), dtype=np.float32),
            np.diag(-np.arange(d, dtype=np.float32) - 2) +
            np.triu(np.ones((d, d), dtype=np.float32)),
        ])  # (d + 1, d)
        self.E = tf.constant(E, dtype=tf.float32)  # [d + 1, d]

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * np.float32(d + 1)
        self.inv_std_dev = tf.constant(inv_std_dev, dtype=tf.float32)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (1.0 / np.sqrt(
            (np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev)  # (d, )
        self.scale_factor = tf.constant(scale_factor,
                                        dtype=tf.float32)  # [d, ]

        valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32))
        self.valid = tf.constant(valid, dtype=tf.int32)

        ds = np.ones((d + 1, ), dtype=np.short) * d  # (d + 1, )
        ds = np.diag(ds)  # (d + 1, d + 1)
        diagone = np.diag(np.ones(d + 1, dtype=np.short))  # (d + 1, d + 1)
        self.ds = tf.constant(ds, dtype=tf.int32)  # [d + 1, ]
        self.diagone = tf.constant(diagone, dtype=tf.int32)  # [d + 1, d + 1]

        self.hash_table = None
        self.blur_neighbors_ = None
        self.offsets_ = None
        self.ranks_ = None
        self.barycentrics_ = None

    def init(self, features):
        # Compute the simplex each feature lies in
        # !!! Shape of feature [N, d]
        # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = features * self.scale_factor[tf.newaxis, ...]  # [N, d]
        elevated = tf.matmul(cf, tf.transpose(self.E, perm=[1,
                                                            0]))  # [N, d + 1]

        # Find the closest 0-colored simplex through rounding
        down_factor = 1.0 / tf.cast(self.d_ + 1, dtype=tf.float32)
        up_factor = tf.cast(self.d_ + 1, dtype=tf.float32)
        v = down_factor * elevated  # [N, d + 1]
        up = tf.math.ceil(v) * up_factor  # [N, d + 1]
        down = tf.math.floor(v) * up_factor  # [N, d + 1]
        rem0 = tf.cast(tf.where(up - elevated < elevated - down, up, down),
                       dtype=tf.float32)  # [N, d + 1]
        _sum = tf.cast(tf.reduce_sum(rem0, axis=1) * down_factor,
                       dtype=tf.int32)  # [N, ]

        # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        rank = tf.zeros(shape=[self.N_, self.d_ + 1],
                        dtype=tf.int32)  # [N, d + 1]
        ds = elevated - rem0  # [N, d + 1]
        di = ds[..., tf.newaxis]  # [N, d + 1, 1]
        dj = ds[..., tf.newaxis, :]  # [N, 1, d + 1]
        di_lt_dj = tf.where(di < dj, 1, 0)  # [N, d + 1, d + 1]
        di_geq_dj = tf.where(di >= dj, 1, 0)  # [N, d + 1, d + 1]
        rank = rank + tf.reduce_sum(di_lt_dj * self.valid[tf.newaxis, ...],
                                    axis=2)  # [N, d + 1]
        rank = rank + tf.reduce_sum(di_geq_dj * self.valid[tf.newaxis, ...],
                                    axis=1)  # [N, d + 1]

        # If the point doesn't lie on the plane (sum != 0) bring it back
        rank = rank + _sum[..., tf.newaxis]  # [N, d + 1]
        ls_zero = rank < 0  # [N, d + 1]
        gt_d = rank > self.d_  # [N, d + 1]
        rank = tf.where(ls_zero, rank + self.d_ + 1, rank)
        rem0 = tf.where(ls_zero, rem0 + tf.cast(self.d_ + 1, dtype=tf.float32),
                        rem0)
        rank = tf.where(gt_d, rank - (self.d_ + 1), rank)
        rem0 = tf.where(gt_d, rem0 - tf.cast(self.d_ + 1, dtype=tf.float32),
                        rem0)

        # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
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
        barycentric = tf.tensor_scatter_nd_add(tensor=barycentric,
                                               indices=idx[..., tf.newaxis],
                                               updates=vs)  # [N x (d + 2), ]
        barycentric = tf.tensor_scatter_nd_sub(tensor=barycentric,
                                               indices=idx1[..., tf.newaxis],
                                               updates=vs)  # [N x (d + 2), ]
        barycentric = tf.reshape(barycentric,
                                 shape=[self.N_, (self.d_ + 2)])  # [N, d + 2]
        idx0 = tf.stack(
            [tf.range(self.N_),
             tf.zeros([
                 self.N_,
             ], dtype=tf.int32)],
            axis=-1)  # [N, 2]
        barycentric = tf.tensor_scatter_nd_add(
            tensor=barycentric,
            indices=idx0,
            updates=(1. + barycentric[..., self.d_ + 1]))  # [N, d + 2]

        # Compute all vertices and their offset
        canonicalT = tf.transpose(self.canonical, perm=[1,
                                                        0])  # [d + 1, d + 1]
        canonical_ext = tf.gather(params=canonicalT,
                                  indices=rank)  # [N, d + 1, d + 1]
        canonical_ext = tf.transpose(canonical_ext,
                                     perm=[0, 2, 1])  # [N, d + 1, d + 1]

        keys = (tf.cast(rem0[..., tf.newaxis, :self.d_], dtype=tf.int32) +
                canonical_ext[..., :self.d_])  # [N, d + 1, d]

        # Keys in string format.
        keys_flat = tf.reshape(keys, shape=[-1, self.d_])  # [N x (d + 1), d]
        hkeys, _ = tf.raw_ops.UniqueV2(x=keys_flat, axis=[
            0,
        ])  # [M, d]
        skeys = tf.strings.reduce_join(tf.strings.as_string(keys_flat),
                                       axis=-1,
                                       separator=',')  # [N x (d + 1), ]
        skeys_uniq = tf.strings.reduce_join(tf.strings.as_string(hkeys),
                                            axis=-1,
                                            separator=',')  # [M, ]
        self.M_ = tf.shape(hkeys)[0]  # Get M
        self.hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                skeys_uniq, tf.range(self.M_, dtype=tf.int32)),
            default_value=-1)
        offset = self.hash_table.lookup(skeys)  # [N x (d + 1), ]

        self.offsets_ = offset  # [N x (d + 1), ]
        self.ranks_ = rank  # [N, d + 1]
        self.barycentrics_ = barycentric[..., :self.d_ + 1]  # [N, d + 1]

        # Find the neighbors of each lattice point
        # Get the number of vertices in the lattice
        # Create the neighborhood structure
        # For each of d+1 axes,
        hkeys_neighbors = hkeys[..., :self.d_]  # [M, d]
        n1s = tf.tile(hkeys_neighbors[:, tf.newaxis, :],
                      [1, self.d_ + 1, 1]) - 1  # [M, d + 1, d]
        n2s = tf.tile(hkeys_neighbors[:, tf.newaxis, :],
                      [1, self.d_ + 1, 1]) + 1  # [M, d + 1, d]
        n1s = n1s + self.ds[tf.newaxis, ..., :self.d_] + self.diagone[
            tf.newaxis, ..., :self.d_]  # [M, d + 1, d]
        n2s = n2s - self.ds[tf.newaxis, ..., :self.d_] - self.diagone[
            tf.newaxis, ..., :self.d_]  # [M, d + 1, d]

        sn1s = tf.strings.reduce_join(tf.strings.as_string(n1s),
                                      axis=-1,
                                      separator=',')  # [M, d + 1]
        sn2s = tf.strings.reduce_join(tf.strings.as_string(n2s),
                                      axis=-1,
                                      separator=',')  # [M, d + 1]

        blur_neighbors0 = self.hash_table.lookup(sn1s)  # [M, d + 1]
        blur_neighbors1 = self.hash_table.lookup(sn2s)  # [M, d + 1]
        blur_neighbors = tf.stack([blur_neighbors0, blur_neighbors1],
                                  axis=-1)  # [M, d + 1, 2]

        self.blur_neighbors_ = blur_neighbors  # [M, d + 1, 2]

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
        os = tf.reshape(self.offsets_, shape=[
            -1,
        ]) + 1  # [N X (d + 1), ]
        ws = tf.reshape(self.barycentrics_, shape=[
            -1,
        ])  # [N x (d + 1), ]

        inpT = tf.transpose(
            inp, perm=[1, 0])  # transpose to channel-first. [value_size, N]

        def splat_channelwise(ch):
            ch_ext = tf.tile(ch[..., tf.newaxis],
                             [1, self.d_ + 1])  # [N, (d + 1)]
            ch_flat = tf.reshape(ch_ext, shape=[
                -1,
            ])  # [N x (d + 1), ]
            val_ch = tf.math.bincount(
                os,
                weights=ch_flat * ws,
                minlength=self.M_ + 2,
                maxlength=self.M_ + 2,
                dtype=tf.float32,
            )
            return val_ch

        valuesT = tf.vectorized_map(splat_channelwise,
                                    inpT)  # [value_size, M + 2]
        values = tf.transpose(valuesT, perm=[1, 0])  # [M + 2, value_size]

        # ->> Blur
        j_range = tf.range(self.d_, -1, -1) if reverse else tf.range(self.d_ +
                                                                     1)
        for j in j_range:
            n1s = self.blur_neighbors_[:self.M_, j, 0] + 1  # [M, ]
            n2s = self.blur_neighbors_[:self.M_, j, 1] + 1  # [M, ]
            n1_vals = tf.gather(values, n1s)  # [M, value_size]
            n2_vals = tf.gather(values, n2s)  # [M, value_size]

            idx_nv = tf.range(1, self.M_ + 1)  # [M, ]
            values = tf.tensor_scatter_nd_add(
                tensor=values,
                indices=idx_nv[..., tf.newaxis],
                updates=0.5 * (n1_vals + n2_vals))

        # ->> Slice
        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        alpha = 1.0 / (1.0 + tf.pow(2.0, -tf.cast(self.d_, dtype=tf.float32)))

        out = ws[..., tf.newaxis] * tf.gather(values, os) * alpha
        out = tf.reshape(out, shape=[self.N_, self.d_ + 1, value_size])
        out = tf.reduce_sum(out, axis=1)

        return out

    # @tf.function
    def compute(self, inp, reverse=False):
        size, n_ch = tf.shape(inp)[0], tf.shape(inp)[1]
        out = self.seq_compute(inp, n_ch, reverse)
        return out
