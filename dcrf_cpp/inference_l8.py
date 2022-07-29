# -*- coding: utf-8 -*-

import numpy as np
from permutohedral_np_cpp import Permutohedral as PermutohedralNP
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from scipy.special import softmax
# import tensorflow as tf

'''
CRF inference
`image` and `unary` are channel-last
'''

def create_unary_from_pred(pred, n_classes, gt_prob):
    pred_oh = np.eye(n_classes)[pred]
    U = np.zeros_like(pred_oh, dtype=np.float32)
    U[pred_oh < 0.01] = -np.log((1. - gt_prob) / (n_classes - 1))
    U[pred_oh >= 0.01] = -np.log(gt_prob)
    return U

def _diagonal_compatibility(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)

def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)

# unary first
# @tf.function
def inference(unary, image, spatial_filter, bilateral_filter, height, width, num_classes, theta_alpha, theta_beta, theta_gamma, spatial_compat, bilateral_compat, num_iterations):
    n_feats = height * width
    d_bifeats = 5
    d_spfeats = 2
    unary_shape = unary.shape

    compatibility_matrix = _potts_compatibility((num_classes, num_classes)) # [n_classes, n_classes]

    # Create spatial and bilateral features
    spatial_coords = np.mgrid[:height, :width][::-1].transpose((1, 2, 0))
    
    bilateral_feats = np.zeros((height, width, d_bifeats), dtype=np.float32)
    bilateral_feats[..., :2] = spatial_coords / theta_alpha
    bilateral_feats[..., 2:] = image / theta_beta
    bilateral_feats = bilateral_feats.reshape((-1, d_bifeats))
    
    spatial_feats = np.zeros((height, width, d_spfeats), dtype=np.float32)
    spatial_feats[:] = spatial_coords / theta_gamma
    spatial_feats = spatial_feats.reshape((-1, d_spfeats))

    bilateral_filter.init(bilateral_feats)
    spatial_filter.init(spatial_feats)
    print('Lattice initialized.')

    all_ones = np.ones([n_feats, 1], dtype=np.float32) # [n_feats, 1]

    # Compute symmetric weight
    spatial_norm_vals = np.zeros_like(all_ones)
    spatial_filter.compute(all_ones, False, spatial_norm_vals) # [n_feats, 1]
    spatial_norm_vals[:] = 1. / (spatial_norm_vals ** .5 + 1e-20)

    bilateral_norm_vals = np.zeros_like(all_ones)
    bilateral_filter.compute(all_ones, False, bilateral_norm_vals) # [n_feats, 1]
    bilateral_norm_vals[:] = 1. / (bilateral_norm_vals ** .5 + 1e-20)

    # Initialize Q
    unary = unary.reshape([n_feats, num_classes]) # [n_feats, n_classes]
    Q = softmax(-unary, axis=-1) # [n_feats, n_classes]

    spatial_out = np.zeros_like(Q)
    bilateral_out = np.zeros_like(Q)

    for i in range(num_iterations):
        tmp1 = -unary # [n_feats, n_classes]

        # Symmetric normalization and spatial message passing
        spatial_filter.compute(Q * spatial_norm_vals, False, spatial_out) # [n_feats, n_classes]
        print('Spatial computation done.')
        spatial_out *= spatial_norm_vals # [n_feats, n_classes]

        # Symmetric normalization and bilateral message passing
        bilateral_filter.compute(Q * bilateral_norm_vals, False, bilateral_out) # [n_feats, n_classes]
        print('Bilateral computation done.')
        bilateral_out *= bilateral_norm_vals # [n_feats, n_classes]

        # Message passing
        message_passing = spatial_compat * spatial_out + bilateral_compat * bilateral_out # [n_feats, n_classes]

        # Compatibility transform
        pairwise = np.matmul(message_passing, compatibility_matrix) # [n_feats, n_classes]

        # Local update
        tmp1 -= pairwise # [n_feats, n_classes]

        # Normalize
        Q = softmax(tmp1, axis=-1) # [n_feats, n_classes]

    return Q.reshape(unary_shape)


if __name__ == "__main__":
    image = Image.open('../../data/false/LC08_L1TP_113026_20160412_20170326_01_T1_sr_bands.png').convert('RGB')
    image = np.asarray(image)
    pred = np.load('../../data/unet/LC08_L1TP_113026_20160412_20170326_01_T1_pred.npz')['arr_0']
    # img = cv2.imread('../../data/examples/im1.png')
    # anno_rgb = cv2.imread('../../data/examples/anno1.png').astype(np.uint32)
    # anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    # colors, labels = np.unique(anno_lbl, return_inverse=True)

    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    #     print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
    #     print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
    #     colors = colors[1:]
    
    # colorize = np.empty((len(colors), 3), np.uint8)
    # colorize[:,0] = (colors & 0x0000FF)
    # colorize[:,1] = (colors & 0x00FF00) >> 8
    # colorize[:,2] = (colors & 0xFF0000) >> 16

    # n_labels = len(set(labels.flat)) - int(HAS_UNK)
    # print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    n_classes = 4
    unary = create_unary_from_pred(pred, n_classes, .7)
    image = (image / 255.).astype(np.float32)

    # unary = unary_from_labels(labels, n_labels, 0.7, HAS_UNK)
    height, width = image.shape[:2]
    print(unary.shape, image.shape, height, width)
    print(unary.dtype)
    # unary = unary.reshape(n_labels, height, width)
    # unary = np.transpose(unary, (1, 2, 0))
    # print(unary.shape, img.shape)

    # Sizes and dims of spatial and color features
    d_spatial, d_image = 2, image.shape[-1]
    n_feats = height * width
    d_bifeats = d_spatial + d_image
    d_spfeats = d_spatial

    bilateral_filter = PermutohedralNP(n_feats, d_bifeats)
    spatial_filter = PermutohedralNP(n_feats, d_spfeats)

    rfn = inference(unary, image, spatial_filter, bilateral_filter, height, width, n_classes, theta_alpha=80., theta_beta=.0625, theta_gamma=3., spatial_compat=3., bilateral_compat=10., num_iterations=10)
    
    MAP = np.argmax(rfn, axis=-1)
    print(MAP.shape)
    plt.imshow(MAP, cmap='gray')
    plt.show()

    # pred = inference(tf.constant(unary.astype(np.float32)), tf.constant((img / 255.).astype(np.float32)), spatial_filter, bilateral_filter, height, width, n_labels, theta_alpha=80., theta_beta=.0625, theta_gamma=3., spatial_compat=3., bilateral_compat=10., num_iterations=10)
    # pred = pred.numpy()
    
    # MAP = np.argmax(pred, axis=-1)
    # plt.imshow(MAP)
    # plt.show()