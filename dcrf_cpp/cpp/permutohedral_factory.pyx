# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp
from libcpp cimport bool

cdef extern from "permutohedral_factory.h":
    void cppInit(const float *features, const int N, const int d, int *os, float *ws, int *blur_neighborsT, int *M) except +

    void cppCompute(const float *inp, const int N, const int value_size, const int M, const int d, const int *os, const float *ws, const int *blur_neighborsT, const bool reverse, float *out) except +

def init(float [:] features, int N, int d, int [:] os, float [:] ws, int [:] blur_neighborsT):
    cdef int M
    cppInit(&features[0], N, d, &os[0], &ws[0], &blur_neighborsT[0], &M)

    return M

def compute(float[:] inp, int N, int value_size, int M, int d, int[:] os, float[:] ws, int[:] blur_neighborsT, bool reverse, float[:] out):
    cppCompute(&inp[0], N, value_size, M, d, &os[0], &ws[0], &blur_neighborsT[0], reverse, &out[0])