"""
Permutohedral Refined UNet
"""

import numpy as np
import tensorflow as tf

from unet import UNet


class PermutohedralRefinedUNet(tf.keras.Model):
    def __init__(
        self,
        pretrained_unet,
        height,
        width,
        n_classes,
        d_bifeats=5,
        d_spfeats=2,
        theta_alpha=80.0,
        theta_beta=0.0625,
        theta_gamma=3.0,
        bilateral_compat=10.0,
        spatial_compat=3.0,
        gt_prob=0.7,
        n_iterations=10,
    ):
        super().__init__()

        # UNet
        self.unet = UNet()

        if pretrained_unet:
            checkpoint = tf.train.Checkpoint(model=self.unet)
            checkpoint.restore(
                tf.train.latest_checkpoint(pretrained_unet)
            ).expect_partial()
            print(
                "Checkpoint restored, at {}".format(
                    tf.train.latest_checkpoint(pretrained_unet)
                )
            )

        # Permutohedral CRF
        self.n_classes = n_classes
        self.n_feats, self.d_bifeats, self.d_spfeats = height * width, d_bifeats, d_spfeats
        # self.M_bifeats, self.M_spfeats = 0, 0
        self.theta_alpha, self.theta_beta, self.theta_gamma = (
            theta_alpha,
            theta_beta,
            theta_gamma,
        )
        self.bilateral_compat, self.spatial_compat = bilateral_compat, spatial_compat
        self.gt_prob = gt_prob
        self.n_iterations = n_iterations

        def create_factors(d):
            canonical = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
            for i in range(d + 1):
                canonical[i, : d + 1 - i] = i
                canonical[i, d + 1 - i :] = i - (d + 1)
            canonical = tf.constant(canonical, dtype=tf.int32)  # [d + 1, d + 1]

            E = np.vstack(
                [
                    np.ones((d,), dtype=np.float32),
                    np.diag(-np.arange(d, dtype=np.float32) - 2)
                    + np.triu(np.ones((d, d), dtype=np.float32)),
                ]
            )  # (d + 1, d)
            E = tf.constant(E, dtype=tf.float32)  # [d + 1, d]

            # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
            inv_std_dev = np.sqrt(2.0 / 3.0) * np.float32(d + 1)

            # Compute the diagonal part of E (p.5 in [Adams et al 2010])
            scale_factor = (
                1.0 / np.sqrt((np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev
            )  # (d, )
            scale_factor = tf.constant(scale_factor, dtype=tf.float32)  # [d, ]

            diff_valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32))
            diff_valid = tf.constant(diff_valid, dtype=tf.int32)

            # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
            alpha = 1.0 / (1.0 + tf.pow(2.0, -tf.cast(d, dtype=tf.float32)))

            d_mat = np.ones((d + 1,), dtype=np.short) * d  # (d + 1, )
            d_mat = np.diag(d_mat)  # (d + 1, d + 1)
            diagone = np.diag(np.ones(d + 1, dtype=np.short))  # (d + 1, d + 1)
            d_mat = tf.constant(d_mat, dtype=tf.int32)  # [d + 1, d + 1]
            diagone = tf.constant(diagone, dtype=tf.int32)  # [d + 1, d + 1]

            return canonical, E, scale_factor, diff_valid, alpha, d_mat, diagone

        (
            self.canonical_bi,
            self.E_bi,
            self.scale_factor_bi,
            self.diff_valid_bi,
            self.alpha_bi,
            self.d_mat_bi,
            self.diagone_bi,
        ) = create_factors(d_bifeats)
        (
            self.canonical_sp,
            self.E_sp,
            self.scale_factor_sp,
            self.diff_valid_sp,
            self.alpha_sp,
            self.d_mat_sp,
            self.diagone_sp,
        ) = create_factors(d_spfeats)

        # Potts compatibility
        self.compatibility_matrix = -1 * tf.eye(n_classes, n_classes, dtype=tf.float32)

        # xs, ys
        ys, xs = tf.meshgrid(
            tf.range(height), tf.range(width), indexing="ij"
        )  # [h, w] and [h, w]
        self.ys, self.xs = tf.cast(ys, dtype=tf.float32), tf.cast(xs, dtype=tf.float32)

        spatial_feats = tf.stack([xs, ys], axis=-1) / theta_gamma  # [h, w, 2]
        self.spatial_feats = tf.reshape(
            spatial_feats, shape=[self.n_feats, d_spfeats]
        )  # [h x w, 2]

    def call(self, x):
        def init(features, bilateral=True):
            N = self.n_feats
            if bilateral:
                canonical, E, scale_factor, diff_valid, d_mat, diagone = (
                    self.canonical_bi,
                    self.E_bi,
                    self.scale_factor_bi,
                    self.diff_valid_bi,
                    self.d_mat_bi,
                    self.diagone_bi,
                )
                d = self.d_bifeats
            else:
                canonical, E, scale_factor, diff_valid, d_mat, diagone = (
                    self.canonical_sp,
                    self.E_sp,
                    self.scale_factor_sp,
                    self.diff_valid_sp,
                    self.d_mat_sp,
                    self.diagone_sp,
                )
                d = self.d_spfeats

            # Compute the simplex each feature lies in
            # !!! Shape of feature [N, d]
            # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
            cf = features * scale_factor[tf.newaxis, ...]  # [N, d]
            elevated = tf.matmul(cf, tf.transpose(E, perm=[1, 0]))  # [N, d + 1]

            # Find the closest 0-colored simplex through rounding
            down_factor = 1.0 / tf.cast(d + 1, dtype=tf.float32)
            up_factor = tf.cast(d + 1, dtype=tf.float32)
            v = down_factor * elevated  # [N, d + 1]
            up = tf.math.ceil(v) * up_factor  # [N, d + 1]
            down = tf.math.floor(v) * up_factor  # [N, d + 1]
            rem0 = tf.cast(
                tf.where(up - elevated < elevated - down, up, down), dtype=tf.float32
            )  # [N, d + 1]
            _sum = tf.cast(
                tf.reduce_sum(rem0, axis=1) * down_factor, dtype=tf.int32
            )  # [N, ]

            # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
            rank = tf.zeros(shape=[N, d + 1], dtype=tf.int32)  # [N, d + 1]
            diff = elevated - rem0  # [N, d + 1]
            diff_i = diff[..., tf.newaxis]  # [N, d + 1, 1]
            diff_j = diff[..., tf.newaxis, :]  # [N, 1, d + 1]
            di_lt_dj = tf.where(diff_i < diff_j, 1, 0)  # [N, d + 1, d + 1]
            di_geq_dj = tf.where(diff_i >= diff_j, 1, 0)  # [N, d + 1, d + 1]
            rank = rank + tf.reduce_sum(
                di_lt_dj * diff_valid[tf.newaxis, ...], axis=2
            )  # [N, d + 1]
            rank = rank + tf.reduce_sum(
                di_geq_dj * diff_valid[tf.newaxis, ...], axis=1
            )  # [N, d + 1]

            # If the point doesn't lie on the plane (sum != 0) bring it back
            rank = rank + _sum[..., tf.newaxis]  # [N, d + 1]
            ls_zero = rank < 0  # [N, d + 1]
            gt_d = rank > d  # [N, d + 1]
            rank = tf.where(ls_zero, rank + d + 1, rank)
            rem0 = tf.where(ls_zero, rem0 + tf.cast(d + 1, dtype=tf.float32), rem0)
            rank = tf.where(gt_d, rank - (d + 1), rank)
            rem0 = tf.where(gt_d, rem0 - tf.cast(d + 1, dtype=tf.float32), rem0)

            # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
            barycentric = tf.zeros(
                [
                    N * (d + 2),
                ],
                dtype=tf.float32,
            )  # [N x (d + 2), ]
            vs = tf.reshape(
                (elevated - rem0) * down_factor,
                shape=[
                    -1,
                ],
            )  # [N x (d + 1), ]
            idx = tf.reshape(
                (d - rank) + tf.range(N)[..., tf.newaxis] * (d + 2),
                shape=[
                    -1,
                ],
            )  # [N x (d + 1), ]
            idx1 = tf.reshape(
                (d - rank + 1) + tf.range(N)[..., tf.newaxis] * (d + 2),
                shape=[
                    -1,
                ],
            )  # [N x (d + 1), ]
            barycentric = tf.tensor_scatter_nd_add(
                tensor=barycentric, indices=idx[..., tf.newaxis], updates=vs
            )  # [N x (d + 2), ]
            barycentric = tf.tensor_scatter_nd_sub(
                tensor=barycentric, indices=idx1[..., tf.newaxis], updates=vs
            )  # [N x (d + 2), ]
            barycentric = tf.reshape(barycentric, shape=[N, (d + 2)])  # [N, d + 2]
            idx0 = tf.stack(
                [
                    tf.range(N),
                    tf.zeros(
                        [
                            N,
                        ],
                        dtype=tf.int32,
                    ),
                ],
                axis=-1,
            )  # [N, 2]
            barycentric = tf.tensor_scatter_nd_add(
                tensor=barycentric,
                indices=idx0,
                updates=(1.0 + barycentric[..., d + 1]),
            )  # [N, d + 2]

            # Compute all vertices and their offset
            canonicalT = tf.transpose(canonical, perm=[1, 0])  # [d + 1, d + 1]
            canonical_ext = tf.gather(
                params=canonicalT, indices=rank
            )  # [N, d + 1, d + 1]
            canonical_ext = tf.transpose(
                canonical_ext, perm=[0, 2, 1]
            )  # [N, d + 1, d + 1]

            keys = (
                tf.cast(rem0[..., tf.newaxis, :d], dtype=tf.int32)
                + canonical_ext[..., :d]
            )  # [N, d + 1, d]

            # Keys in string format.
            keys = tf.reshape(keys, shape=[-1, d])  # [N x (d + 1), d]
            hkeys, _ = tf.raw_ops.UniqueV2(
                x=keys,
                axis=[
                    0,
                ],
            )  # [M, d]
            skeys = tf.strings.reduce_join(
                tf.strings.as_string(keys), axis=-1, separator=","
            )  # [N x (d + 1), ]
            skeys_uniq = tf.strings.reduce_join(
                tf.strings.as_string(hkeys), axis=-1, separator=","
            )  # [M, ]

            M = tf.shape(hkeys)[0]  # Get M
            hash_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    skeys_uniq, tf.range(M, dtype=tf.int32)
                ),
                default_value=-1,
            )
            offset = hash_table.lookup(skeys)  # [N x (d + 1), ]

            # Find the neighbors of each lattice point
            # Get the number of vertices in the lattice
            # Create the neighborhood structure
            # For each of d+1 axes,
            hkeys_neighbors = hkeys[..., :d]  # [M, d]
            n1s = (
                tf.tile(hkeys_neighbors[:, tf.newaxis, :], [1, d + 1, 1]) - 1
            )  # [M, d + 1, d]
            n2s = (
                tf.tile(hkeys_neighbors[:, tf.newaxis, :], [1, d + 1, 1]) + 1
            )  # [M, d + 1, d]
            n1s = (
                n1s + d_mat[tf.newaxis, ..., :d] + diagone[tf.newaxis, ..., :d]
            )  # [M, d + 1, d]
            n2s = (
                n2s - d_mat[tf.newaxis, ..., :d] - diagone[tf.newaxis, ..., :d]
            )  # [M, d + 1, d]

            sn1s = tf.strings.reduce_join(
                tf.strings.as_string(n1s), axis=-1, separator=","
            )  # [M, d + 1]
            sn2s = tf.strings.reduce_join(
                tf.strings.as_string(n2s), axis=-1, separator=","
            )  # [M, d + 1]

            blur_neighbors0 = hash_table.lookup(sn1s)  # [M, d + 1]
            blur_neighbors1 = hash_table.lookup(sn2s)  # [M, d + 1]
            blur_neighbors = tf.stack(
                [blur_neighbors0, blur_neighbors1], axis=-1
            )  # [M, d + 1, 2]

            # Reshape and shift for `compute`
            offset = (
                tf.reshape(
                    offset,
                    shape=[
                        -1,
                    ],
                )
                + 1
            )  # [N X (d + 1), ]
            barycentric = tf.reshape(
                barycentric[..., : d + 1],
                shape=[
                    -1,
                ],
            )  # [N x (d + 1), ]
            blur_neighbors = blur_neighbors + 1

            return M, offset, barycentric, blur_neighbors

        def compute(inp, M, os, ws, blur_neighbors, reverse=False, bilateral=True):
            N = self.n_feats
            if bilateral:
                d = self.d_bifeats
                alpha = self.alpha_bi
            else:
                d = self.d_spfeats
                alpha = self.alpha_sp

            value_size = tf.shape(inp)[1]

            inpT = tf.transpose(
                inp, perm=[1, 0]
            )  # transpose to channel-first. [value_size, N]

            def splat_channelwise(ch):
                ch_ext = tf.tile(ch[..., tf.newaxis], [1, d + 1])  # [N, (d + 1)]
                ch_flat = tf.reshape(
                    ch_ext,
                    shape=[
                        -1,
                    ],
                )  # [N x (d + 1), ]
                val_ch = tf.math.bincount(
                    os,
                    weights=ch_flat * ws,
                    minlength=M + 2,
                    maxlength=M + 2,
                    dtype=tf.float32,
                )
                return val_ch

            valuesT = tf.vectorized_map(splat_channelwise, inpT)  # [value_size, M + 2]
            values = tf.transpose(valuesT, perm=[1, 0])  # [M + 2, value_size]

            # ->> Blur
            j_range = tf.range(d, -1, -1) if reverse else tf.range(d + 1)
            idx_nv = tf.range(1, M + 1)  # [M, ]
            for j in j_range:
                n1s = blur_neighbors[:M, j, 0]  # [M, ]
                n2s = blur_neighbors[:M, j, 1]  # [M, ]
                n1_vals = tf.gather(params=values, indices=n1s)  # [M, value_size]
                n2_vals = tf.gather(params=values, indices=n2s)  # [M, value_size]

                values = tf.tensor_scatter_nd_add(
                    tensor=values,
                    indices=idx_nv[..., tf.newaxis],
                    updates=0.5 * (n1_vals + n2_vals),
                )

            # ->> Slice

            out = ws[..., tf.newaxis] * tf.gather(params=values, indices=os) * alpha
            out = tf.reshape(out, shape=[N, d + 1, value_size])
            out = tf.reduce_sum(out, axis=1)

            return out

        # ->> x.rank should be three.
        image = x[..., 4:1:-1] # x.shape [H, W, N_classes], image.shape [H, W, 3]

        # Only forward
        logit = self.unet(x[tf.newaxis, ...])[0] # [H, W, N_classes]
        prob = tf.nn.softmax(logit, name='logit2prob') # [H, W, N_classes]
        unary = -tf.math.log(prob * self.gt_prob, name='prob2unary') # [H, W, N_classes]
        unary_shape = tf.shape(unary)

        bilateral_feats = tf.concat(
            [
                tf.stack([self.xs, self.ys], axis=-1) / self.theta_alpha,
                image / self.theta_beta,
            ],
            axis=-1,
        )  # [h, w, d_bifeats]
        bilateral_feats = tf.reshape(
            bilateral_feats, shape=[self.n_feats, self.d_bifeats]
        )  # [n_feats, d_bifeats]

        M_bi, os_bi, ws_bi, blur_neighbors_bi = init(bilateral_feats, bilateral=True)
        M_sp, os_sp, ws_sp, blur_neighbors_sp = init(
            self.spatial_feats, bilateral=False
        )

        all_ones = tf.ones([self.n_feats, 1], dtype=tf.float32)

        # Compute symmetric weight
        bilateral_norm_vals = compute(
            all_ones,
            M=M_bi,
            os=os_bi,
            ws=ws_bi,
            blur_neighbors=blur_neighbors_bi,
            reverse=False,
            bilateral=True,
        )  # [n_feats, n_classes]
        spatial_norm_vals = compute(
            all_ones,
            M=M_sp,
            os=os_sp,
            ws=ws_sp,
            blur_neighbors=blur_neighbors_sp,
            reverse=False,
            bilateral=False,
        )  # [n_feats, n_classes]

        bilateral_norm_vals = 1.0 / (bilateral_norm_vals**0.5 + 1e-20)
        spatial_norm_vals = 1.0 / (spatial_norm_vals**0.5 + 1e-20)

        # Initialize Q
        unary = tf.reshape(unary, shape=[-1, self.n_classes])
        Q = tf.nn.softmax(-unary)

        for i in range(self.n_iterations):
            tmp1 = -unary

            # Symmetric normalization and bilateral message passing
            bilateral_out = compute(
                Q * bilateral_norm_vals,
                M=M_bi,
                os=os_bi,
                ws=ws_bi,
                blur_neighbors=blur_neighbors_bi,
                reverse=False,
                bilateral=True,
            )  # [n_feats, n_classes]
            bilateral_out *= bilateral_norm_vals  # [n_feats, n_classes]

            # Symmetric normalization and spatial message passing
            spatial_out = compute(
                Q * spatial_norm_vals,
                M=M_sp,
                os=os_sp,
                ws=ws_sp,
                blur_neighbors=blur_neighbors_sp,
                reverse=False,
                bilateral=False,
            )  # [n_feats, n_classes]
            spatial_out *= spatial_norm_vals  # [n_feats, n_classes]

            # Message passing
            message_passing = (
                self.spatial_compat * spatial_out
                + self.bilateral_compat * bilateral_out
            )  # [n_feats, n_classes]

            # Compatibility transform
            pairwise = tf.matmul(
                message_passing, self.compatibility_matrix
            )  # [n_feats, n_classes]

            # Local update
            tmp1 -= pairwise

            # Normalize
            Q = tf.nn.softmax(tmp1)

        return tf.reshape(Q, shape=unary_shape)


    