import numpy as np
from PIL import Image
from permutohedral_wloop import Permutohedral as PermutohedralLoop
from permutohedral_np import Permutohedral as PermutohedralNP
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

lattice_loop = PermutohedralLoop(N, d)
lattice_loop.init(features)
print('Lattice w/ loop initialized.')

lattice_np = PermutohedralNP(N, d)
lattice_np.init(features)
print('Lattice in NumPy initialized.')

all_ones = np.ones((N, 1), dtype=np.float32)
src = im.reshape((-1, n_channels))

norm = lattice_loop.compute(all_ones)
norm = norm.reshape((h, w, 1))

dst_loop = lattice_loop.compute(src)
dst_loop = dst_loop.reshape((h, w, n_channels))
dst_loop = dst_loop / norm
dst_loop = (dst_loop - dst_loop.min()) / (dst_loop.max() - dst_loop.min() + 1e-5)

dst_np = lattice_np.compute(src)
dst_np = dst_np.reshape((h, w, n_channels))
dst_np = dst_np / norm
dst_np = (dst_np - dst_np.min()) / (dst_np.max() - dst_np.min() + 1e-5)

print(np.max(np.abs(dst_loop - dst_np) / (np.maximum(1e-8, np.abs(dst_loop) + np.abs(dst_np)))))

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_loop', dst_loop[..., ::-1])
cv2.imshow('im_np', dst_np[..., ::-1])
cv2.waitKey()
