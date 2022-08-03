# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp
from libcpp cimport bool

cdef extern from "densecrf_cppfactory.h":

    cdef cppclass DenseCRF:
        DenseCRF(int H, int W, int n_classes, int d_bifeats, int d_spfeats, float theta_alpha, float theta_beta, float theta_gamma, float bilateral_compat, float spatial_compat, int n_iterations) except +

        void inference(const float *unary_1d, const float *image_1d, float *out_1d) except +

cdef class PyDenseCRF:
    cdef DenseCRF *dcrf

    def __cinit__(self, int H, int W, int n_classes, int d_bifeats, int d_spfeats, float theta_alpha, float theta_beta, float theta_gamma, float bilateral_compat, float spatial_compat, int n_iterations):
        self.dcrf = new DenseCRF(H, W, n_classes, d_bifeats, d_spfeats, theta_alpha, theta_beta, theta_gamma, bilateral_compat, spatial_compat, n_iterations)

    def inference(self, float[:] unary_1d, float[:] image_1d, float[:] out_1d):
        self.dcrf.inference(&unary_1d[0], &image_1d[0], &out_1d[0])

    def __dealloc__(self):
        del self.dcrf

