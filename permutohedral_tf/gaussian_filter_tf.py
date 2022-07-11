from pyexpat import features
import numpy as np
from PIL import Image
from permutohedral_tf_v3 import Permutohedral as PremutohedralTF
import matplotlib.pyplot as plt
import cv2

im = Image.open('../data/lena.jpg')
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

features = np.zeros((h, w, 2), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) * invSpatialStdev
features[..., :2] = spatial_feat
features = features.reshape((-1, 2))

N, d = features.shape[0], features.shape[1]

lattice_tf = PremutohedralTF(N, d)
lattice_tf.init(features)

all_ones = np.ones((N, 1), dtype=np.float32)
norms = lattice_tf.compute(all_ones)
norms = norms.numpy()
norms = norms.reshape((h, w, 1))

src = im.reshape((-1, n_channels))
dst = lattice_tf.compute(src.astype(np.float32))
dst = dst.numpy()
dst = dst.reshape((h, w, n_channels))
dst = dst / norms
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('dst', dst[..., ::-1])
cv2.waitKey()