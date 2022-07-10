from random import randrange
import numpy as np
import tensorflow as tf

N = 2
d = 5
features = np.arange(N * d).reshape((N, d)) / (N * d)
features = features.astype(np.float32)
print(features)

canonical = np.zeros(((d + 1) * (d + 1)), dtype=np.int32)
for i in range(d + 1):
    for j in range(d - i + 1):
        canonical[i * (d + 1) + j] = i
    for j in range(d - i + 1, d + 1):
        canonical[i * (d + 1) + j] = i - (d + 1)

canonical_tf = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
for i in range(d + 1):
    canonical_tf[i, :d + 1 - i] = i
    canonical_tf[i, d + 1 - i:] = i - (d + 1)

# if np.allclose(canonical, canonical_tf.flatten()):
#     print('good')
# else:
#     print('wrong')

elevated = np.zeros((d + 1), dtype=np.float32)
scale_factor = np.zeros((d, ), dtype=np.float32)
rem0 = np.zeros((d + 1, ), dtype=np.float32)
rank = np.zeros((d + 1, ), dtype=np.int32)
barycentric = np.zeros((d + 2, ), dtype=np.float32)
key = np.zeros((d + 1, ), dtype=np.int32)
offset_ = np.zeros(((d + 1) * N, ), dtype=np.int32)

table_keys = np.zeros((N + 1, d + 1, d), dtype=np.int32)

inv_std_dev = np.sqrt(2. / 3.) * (d + 1)
for i in range(d):
    scale_factor[i] = 1. / np.sqrt((i + 2) * (i + 1)) * inv_std_dev

inv_std_dev_tf = (np.sqrt(2. / 3.) * (d + 1)).astype(np.float32)
scale_factor_tf = (1.0 / np.sqrt((np.arange(d) + 2) * (np.arange(d) + 1)) *
                   inv_std_dev).astype(np.float32)  # (d, )
print(np.allclose(inv_std_dev, inv_std_dev_tf))
print(np.allclose(scale_factor, scale_factor_tf))
valid_tf = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32))

down_factor = 1. / (d + 1)
up_factor = d + 1

E = np.vstack([
    np.ones((d, ), dtype=np.float32),
    np.diag(-np.arange(d, dtype=np.float32) - 2) +
    np.triu(np.ones((d, d), dtype=np.float32)),
])  # (d + 1, d)

cf_tf = features * scale_factor_tf[tf.newaxis, ...]
elevated_tf = tf.matmul(cf_tf, tf.transpose(E, perm=[1, 0]))
v_tf = down_factor * elevated_tf  # [N, d + 1]
up_tf = tf.math.ceil(v_tf) * up_factor  # [N, d + 1]
down_tf = tf.math.floor(v_tf) * up_factor  # [N, d + 1]
rem0_tf = tf.cast(tf.where(up_tf - elevated_tf < elevated_tf - down_tf, up_tf,
                           down_tf),
                  dtype=tf.float32)  # [N, d + 1]
_sum_tf = tf.cast(tf.reduce_sum(rem0_tf, axis=1) * down_factor,
                  dtype=tf.int32)  # [N, ]

rank_tf = tf.zeros(shape=[N, d + 1], dtype=tf.int32)  # [N, d + 1]
ds_tf = elevated_tf - rem0_tf  # [N, d + 1]
di_tf = ds_tf[..., tf.newaxis]  # [N, d + 1, 1]
dj_tf = ds_tf[..., tf.newaxis, :]  # [N, 1, d + 1]
di_lt_dj_tf = tf.where(di_tf < dj_tf, 1, 0)  # [N, d + 1, d + 1]
di_geq_dj_tf = tf.where(di_tf >= dj_tf, 1, 0)  # [N, d + 1, d + 1]
rank_tf = rank_tf + tf.reduce_sum(di_lt_dj_tf * valid_tf[tf.newaxis, ...],
                                  axis=2)  # [N, d + 1]
rank_tf = rank_tf + tf.reduce_sum(di_geq_dj_tf * valid_tf[tf.newaxis, ...],
                                  axis=1)  # [N, d + 1]

rank_tf = rank_tf + _sum_tf[..., tf.newaxis]  # [N, d + 1]
ls_zero_tf = rank_tf < 0  # [N, d + 1]
gt_d_tf = rank_tf > d  # [N, d + 1]
rank_tf = tf.where(ls_zero_tf, rank_tf + d + 1, rank_tf)
rem0_tf = tf.where(ls_zero_tf, rem0_tf + d + 1, rem0_tf)
rank_tf = tf.where(gt_d_tf, rank_tf - (d + 1), rank_tf)
rem0_tf = tf.where(gt_d_tf, rem0_tf - (d + 1), rem0_tf)

barycentric_tf = tf.zeros(shape=[
    N * (d + 2),
], dtype=tf.float32)  # [N x (d + 2), ]
vs_tf = tf.reshape((elevated_tf - rem0_tf) * down_factor, shape=[
    -1,
])  # [N x (d + 1), ]
idx_tf = tf.reshape((d - rank_tf) + tf.range(N)[..., tf.newaxis] * (d + 2),
                    shape=[
                        -1,
                    ])  # [N x (d + 1), ]
idx1_tf = tf.reshape(
    (d - rank_tf + 1) + tf.range(N)[..., tf.newaxis] * (d + 2), shape=[
        -1,
    ])  # [N x (d + 1), ]
barycentric_tf = tf.tensor_scatter_nd_add(tensor=barycentric_tf,
                                          indices=idx_tf[..., tf.newaxis],
                                          updates=vs_tf)  # [N x (d + 2), ]
barycentric_tf = tf.tensor_scatter_nd_sub(tensor=barycentric_tf,
                                          indices=idx1_tf[..., tf.newaxis],
                                          updates=vs_tf)  # [N x (d + 2), ]
barycentric_tf = tf.reshape(barycentric_tf, shape=[N, (d + 2)])  # [N, d + 2]
idx0_tf = tf.stack([tf.range(N), tf.zeros([
    N,
], dtype=tf.int32)], axis=-1)  # [N, 2]
barycentric_tf = tf.tensor_scatter_nd_add(
    tensor=barycentric_tf,
    indices=idx0_tf,
    updates=(1. + barycentric_tf[..., d + 1]))  # [N, d + 2]

canonicalT_tf = tf.transpose(canonical_tf, perm=[1, 0])  # [d + 1, d + 1]
canonical_ext_tf = tf.gather(params=canonicalT_tf,
                             indices=rank_tf)  # [N, d + 1, d + 1]
canonical_ext_tf = tf.transpose(canonical_ext_tf,
                                perm=[0, 2, 1])  # [N, d + 1, d + 1]

keys_tf = (tf.cast(rem0_tf[..., tf.newaxis, :d], dtype=tf.int32) +
           canonical_ext_tf[..., :d])  # [N, d + 1, d]
# keys_tf = tf.concat([keys_tf, tf.zeros([N, d + 1, 1], dtype=tf.int32)],
#                     axis=-1)  # [N, d + 1, d + 1]

hkeys_neighbors_tf = tf.reshape(keys_tf, shape=[-1, d])[..., :d]
n1s_tf = tf.tile(hkeys_neighbors_tf[:, tf.newaxis, :], [1, d + 1, 1]) - 1
n2s_tf = tf.tile(hkeys_neighbors_tf[:, tf.newaxis, :], [1, d + 1, 1]) + 1
ds2_tf = np.ones((d + 1, ), dtype=np.short) * d  # (d + 1, )
ds2_tf = np.diag(ds2_tf)  # (d + 1, d + 1)
diagone_tf = np.diag(np.ones(d + 1, dtype=np.short))  # (d + 1, d + 1)
n1s_tf = n1s_tf + ds2_tf[tf.newaxis, ..., :d] + diagone_tf[tf.newaxis, ..., :d]
n2s_tf = n2s_tf - ds2_tf[tf.newaxis, ..., :d] - diagone_tf[tf.newaxis, ..., :d]

for k in range(N):
    f = features[k]
    sm = 0
    for j in range(d, 0, -1):
        cf = f[j - 1] * scale_factor[j - 1]
        elevated[j] = sm - j * cf
        sm += cf
    elevated[0] = sm

    print('elevated', k, np.allclose(elevated, elevated_tf[k].numpy()))

    sum_ = 0
    for i in range(0, d + 1):
        v = down_factor * elevated[i]
        up = np.ceil(v) * up_factor
        down = np.floor(v) * up_factor
        if (up - elevated[i] < elevated[i] - down):
            rd2 = up.astype(np.int32)
        else:
            rd2 = down.astype(np.int32)
        rem0[i] = rd2
        sum_ += rd2 * down_factor

    print('rem0', k, np.allclose(rem0, rem0_tf[k].numpy()))

    for i in range(d + 1):
        rank[i] = 0
    for i in range(d):
        di = elevated[i] - rem0[i]
        for j in range(i + 1, d + 1):
            if di < elevated[j] - rem0[j]:
                rank[i] += 1
            else:
                rank[j] += 1

    print('rank', k, np.allclose(rank, rank_tf[k].numpy()))

    for i in range(d + 1):
        rank[i] += sum_
        if rank[i] < 0:
            rank[i] += d + 1
            rem0[i] += d + 1
        if rank[i] > d:
            rank[i] -= d + 1
            rem0[i] -= d + 1

    print('rank', k, np.allclose(rank, rank_tf[k].numpy()))
    print('rem0', k, np.allclose(rem0, rem0_tf[k].numpy()))

    for i in range(d + 2):
        barycentric[i] = 0
    for i in range(d + 1):
        v = (elevated[i] - rem0[i]) * down_factor
        barycentric[d - rank[i]] += v
        barycentric[d - rank[i] + 1] -= v

    barycentric[0] += 1. + barycentric[d + 1]

    print('barycentric', k, np.allclose(barycentric, barycentric_tf[k]))

    for remainder in range(d + 1):
        for i in range(d):
            key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]]

        print('key', k, remainder, np.allclose(key[:-1], keys_tf[k, remainder]))

        table_keys[k, remainder] = key[:-1]

M = N * (d + 1)
table_keys = np.reshape(table_keys, (-1, ))
n1 = np.zeros((d + 1, ), dtype=np.int32)
n2 = np.zeros((d + 1, ), dtype=np.int32)

for j in range(d + 1):
    for i in range(M):
        key = table_keys[i * d:]
        n1[:d] = key[:d] - 1
        n2[:d] = key[:d] + 1

        n1[j] = key[j] + d
        n2[j] = key[j] - d

        print('n1', j, i, np.allclose(n1[:d], n1s_tf[i, j]))
        print('n2', j, i, np.allclose(n2[:d], n2s_tf[i, j]))


    