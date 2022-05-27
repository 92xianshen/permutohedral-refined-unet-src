import numpy as np
from PIL import Image
from permutohedral_numpified import Permutohedral
# from tf_impl import Permutohedral
import matplotlib.pyplot as plt
import cv2

im = Image.open('../data/lena.small.jpg')
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

features = np.zeros((h, w, 5), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) * invSpatialStdev
color_feat = im * invColorStdev
features[..., :2] = spatial_feat
features[..., 2:] = color_feat
features = features.reshape((-1, 5))

N, d = features.shape[0], features.shape[1]
lattice = Permutohedral(N, d)
lattice.init(features)
lattice2 = Permutohedral(N, d)
lattice2.numpified_init(features)
print('Lattice initialized.')

if np.allclose(lattice.offset_, lattice2.offset_):
    print('`offset` all close')
else:
    print('`offset` error')
if np.allclose(lattice.rank_, lattice2.rank_):
    print('`rank` all close')
else:
    print('`rank` error')
if np.allclose(lattice.barycentric_, lattice2.barycentric_):
    print('`barycentric` all close')
else:
    print('`barycentric` error')
if np.allclose(lattice.blur_neighbors_, lattice2.blur_neighbors_):
    print('`blur_neighbors` all close')


all_ones = np.ones((N, 1), dtype=np.float32)
all_ones = lattice.compute(all_ones)
all_ones = all_ones.reshape((h, w, 1))

src = im.reshape((-1, n_channels))
dst = lattice.compute(src)
dst = dst.reshape((h, w, n_channels))
dst = dst / all_ones
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

all_ones2 = np.ones((N, 1), dtype=np.float32)
all_ones2 = lattice2.compute(all_ones2)
all_ones2 = all_ones2.reshape((h, w, 1))

src2 = im.reshape((-1, n_channels))
dst2 = lattice2.compute(src2)
dst2 = dst2.reshape((h, w, n_channels))
dst2 = dst2 / all_ones2
dst2 = (dst2 - dst2.min()) / (dst2.max() - dst2.min() + 1e-5)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_filtered', dst[..., ::-1])
cv2.imshow('im_filtered2', dst2[..., ::-1])
cv2.waitKey()

# im_filtered = np.zeros_like(im)
# for ch in range(n_channels):
#     imch = im[..., ch:ch + 1].transpose((2, 0, 1)).reshape((1, -1))
#     imch_filtered = lattice.compute(imch)
#     imch_filtered = imch_filtered.reshape((1, h, w))[0]
#     imch_filtered = imch_filtered / all_ones
#     imch_filtered = (imch_filtered - imch_filtered.min()) / (imch_filtered.max() - imch_filtered.min())
#     im_filtered[..., ch] = imch_filtered

# cv2.imshow('im', im[..., ::-1])
# cv2.imshow('im_filtered', im_filtered[..., ::-1])
# cv2.waitKey()

# im_add = im.transpose((2, 0, 1)).reshape((n_channels, -1))
# im_add = np.vstack([im_add, np.ones((1, h * w), dtype=im.dtype)])
# print(im_add.shape)

# im_filtered = lattice.compute(im_add)
# im_filtered = (im_filtered[:3] / im_filtered[-1:])
# print(im_filtered.max(), im_filtered.min())
# im_filtered = im_filtered.reshape((n_channels, h, w)).transpose((1, 2, 0))
# plt.imshow(im_filtered / im_filtered.max())
# plt.show()

# # all_ones = np.ones((1, N), dtype=np.float32)
# # all_ones = lattice.compute(all_ones)
# # all_ones = all_ones.reshape((1, h, w))[0]

# # plt.imshow(all_ones / all_ones.max(), cmap='gray')
# # plt.show()