# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp
from libcpp cimport bool

cdef extern from "permutohedral_cppfactory.h":

    cdef cppclass Permutohedral:
        Permutohedral(int N, int d) except +
        void init(const float *features) except +
        void testCompute(const float *inp_1d, const int N, const int value_size, const bool reversal, float *out_1d) except +

cdef class PyPermutohedral:
    cdef Permutohedral *lattice

    def __cinit__(self, int N, int d):
        self.lattice = new Permutohedral(N, d)

    def init(self, float[:] features):
        self.lattice.init(&features[0])

    def compute(self, float[:] inp_1d, int N, int value_size, bool reversal, float[:] out_1d):
        self.lattice.testCompute(&inp_1d[0], N, value_size, reversal, &out_1d[0])

    def __dealloc__(self):
        del self.lattice