
# Example:
# /reg/g/psdm/sw/releases/ana-current/ConfigSvc/pyext/_ConfigSvc.pyx

# passing numpy arrays:
# http://stackoverflow.com/questions/17855032/passing-and-returning-numpy-arrays-to-c-methods-via-cython

# issue with outdated numpy
# https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api

from libcpp.string cimport string
#from libcpp.vector cimport vector
#from libcpp cimport bool
from libc.time cimport time_t, ctime

#------------------------------

cdef extern from "<stdint.h>" nogil:
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long  uint64_t

#------------------------------

import numpy as np
cimport numpy as np

#ctypedef fused dtypes2d :
#    np.ndarray[np.double_t, ndim=2]
#    np.ndarray[np.int16_t,  ndim=2]
#    np.ndarray[np.uint16_t, ndim=2]

#------------------------------
# TEST example
#------------------------------

cdef extern from "psalgos/cfib.h":
    double cfib(int n)

def fib(n):
    """Returns the n-th Fibonacci number"""
    return cfib(n)

#------------------------------
