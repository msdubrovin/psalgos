
# Example:
# /reg/g/psdm/sw/releases/ana-current/ConfigSvc/pyext/_ConfigSvc.pyx

# passing numpy arrays:
# http://stackoverflow.com/questions/17855032/passing-and-returning-numpy-arrays-to-c-methods-via-cython

# issue with outdated numpy
# https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api

#------------------------------

#from libcpp.vector cimport vector
#from libcpp cimport bool

from libc.time cimport time_t, ctime
from libcpp.string cimport string

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

cimport numpy as np
import numpy as np

#------------------------------

ctypedef fused nptype2d :
    np.ndarray[np.double_t,  ndim=2, mode="c"]
    np.ndarray[np.float64_t, ndim=2, mode="c"]
    np.ndarray[np.int16_t,   ndim=2, mode="c"]
    np.ndarray[np.int32_t,   ndim=2, mode="c"]
    np.ndarray[np.int64_t,   ndim=2, mode="c"]
    np.ndarray[np.uint16_t,  ndim=2, mode="c"]
    np.ndarray[np.uint32_t,  ndim=2, mode="c"]
    np.ndarray[np.uint64_t,  ndim=2, mode="c"]

#------------------------------
# TEST example
#------------------------------

cdef extern from "psalgos/cfib.h":
    double cfib(int n)

def fib(n):
    """Returns the n-th Fibonacci number"""
    return cfib(n)

#------------------------------
# TEST numpy
#------------------------------

cdef extern from "psalgos/ctest_nda.h":
    void ctest_nda[T](T *arr, int r, int c) except +
    #void ctest_nda_f8(double   *arr, int r, int c) except +
    #void ctest_nda_i2(int16_t  *arr, int r, int c) except +
    #void ctest_nda_u2(uint16_t *arr, int r, int c) except +

#------------------------------

def test_nda_v1(nptype2d nda): ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])
    
#------------------------------

cdef extern from "psalgos/LocalExtrema.h" namespace "localextrema":

    void mapOfLocalMinimums[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , const size_t& rank
                              , uint16_t *map
                              )

    void mapOfLocalMaximums[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , const size_t& rank
                              , uint16_t *map
                              )

    void mapOfLocalMaximumsRank1Cross[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , uint16_t *map
                              )

    void printMatrixOfDiagIndexes(const size_t& rank)
    void printVectorOfDiagIndexes(const size_t& rank)



def local_minimums(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   int32_t rank,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] map\
                  ): mapOfLocalMinimums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &map[0,0])

def local_maximums(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   int32_t rank,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] map\
                  ): mapOfLocalMaximums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &map[0,0])

def local_maximums_rank1_cross(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] map\
                  ): mapOfLocalMaximumsRank1Cross(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], &map[0,0])

def print_matrix_of_diag_indexes(int32_t& rank) : printMatrixOfDiagIndexes(rank)

def print_vector_of_diag_indexes(int32_t& rank) : printVectorOfDiagIndexes(rank)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

#cdef extern from "ndarray/ndarray.h" : # namespace "std":
#    cdef cppclass ndarray[ElemType, NDim]:
#        double x, y, time
#        int32_t method
#        hit_class(sort_class *) except +

#cdef class py_ndarray:
#    """ Python wrapper for C++ class. 
#    """
#    cdef hit_ndarray* cptr  # holds a C++ instance

#    def __cinit__(self, py_sort_class sorter, int i=0):
#        #print "In py_hit_class.__cinit__ index: %d" % i
#        self.cptr = sorter.cptr.output_hit_array[i]

##    def __dealloc__(self):
##        print "In py_hit_class.__dealloc__"
##        del self.cptr

#    @property
#    def x(self) : return self.cptr.x

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
