
# Example:
# /reg/g/psdm/sw/releases/ana-current/ConfigSvc/pyext/_ConfigSvc.pyx

# passing numpy arrays:
# http://stackoverflow.com/questions/17855032/passing-and-returning-numpy-arrays-to-c-methods-via-cython

# issue with outdated numpy
# https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api

#------------------------------

from libcpp.vector cimport vector
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


# ctypedef uint16_t mask_t

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
#------------------------------
#------------------------------
#------------------------------
#------------------------------

cdef extern from "psalgos/LocalExtrema.h" namespace "localextrema":

    void mapOfLocalMinimums[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , const size_t& rank
                              , uint16_t *arr2d
                              )

    void mapOfLocalMaximums[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , const size_t& rank
                              , uint16_t *arr2d
                              )

    void mapOfLocalMaximumsRank1Cross[T](const T *data
                              , const uint16_t *mask
                              , const size_t& rows
                              , const size_t& cols
                              , uint16_t *arr2d
                              )

    void printMatrixOfDiagIndexes(const size_t& rank)
    void printVectorOfDiagIndexes(const size_t& rank)



def local_minimums(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   int32_t rank,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] arr2d\
                  ): mapOfLocalMinimums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &arr2d[0,0])

def local_maximums(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   int32_t rank,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] arr2d\
                  ): mapOfLocalMaximums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &arr2d[0,0])

def local_maximums_rank1_cross(nptype2d data,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] mask,\
                   np.ndarray[uint16_t, ndim=2, mode="c"] arr2d\
                  ): mapOfLocalMaximumsRank1Cross(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], &arr2d[0,0])

def print_matrix_of_diag_indexes(int32_t& rank) : printMatrixOfDiagIndexes(rank)

def print_vector_of_diag_indexes(int32_t& rank) : printVectorOfDiagIndexes(rank)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

cdef extern from "psalgos/PeakFinderAlgos.h" namespace "psalgos":
    cdef cppclass Peak :
        Peak() except +
        Peak(const Peak& o) except +
        Peak operator=(const Peak& rhs) except +
        float seg
        float row
        float col
        float npix
        float npos
        float amp_max
        float amp_tot
        float row_cgrav 
        float col_cgrav
        float row_sigma
        float col_sigma
        float row_min
        float row_max
        float col_min
        float col_max
        float bkgd
        float noise
        float son       

#------------------------------

cdef class py_peak :
    cdef Peak* cptr  # holds a C++ pointer to instance

    def __cinit__(self, _make_obj=True):
        #print "In py_peak.__cinit__"
        if _make_obj:
            self.cptr = new Peak()

    def __dealloc__(self):
        #print "In py_peak.__dealloc__"
        if self.cptr is not NULL :
            del self.cptr
            self.cptr = NULL


    # https://groups.google.com/forum/#!topic/cython-users/39Nwqsksdto
    @staticmethod
    cdef factory(Peak cpp_obj):
        py_obj = py_peak.__new__(py_peak, _make_obj=False)
        (<py_peak> py_obj).cptr = new Peak(cpp_obj) # C++ copy constructor
        return py_obj


#    @staticmethod
#    cdef factory(Peak cpp_obj):
#        return self.peak_pars(cpp_obj)


    def peak_pars(self) : #, cpp_obj=None) : 
        p = self.cptr     # if cpp_obj is None else <Peak*> &cpp_obj
        return (p.seg, p.row, p.col, p.npix, p.npos, p.amp_max, p.amp_tot,\
                p.row_cgrav, p.col_cgrav, p.row_sigma, p.col_sigma,\
                p.row_min, p.row_max, p.col_min, p.col_max, p.bkgd, p.noise, p.son)

    @property
    def seg      (self) : return self.cptr.seg       

    @property
    def row      (self) : return self.cptr.row       

    @property
    def col      (self) : return self.cptr.col       

    @property
    def npix     (self) : return self.cptr.npix      

    @property
    def npos     (self) : return self.cptr.npos      

    @property
    def amp_max  (self) : return self.cptr.amp_max   

    @property
    def amp_tot  (self) : return self.cptr.amp_tot   

    @property
    def row_cgrav(self) : return self.cptr.row_cgrav 

    @property
    def col_cgrav(self) : return self.cptr.col_cgrav 

    @property
    def row_sigma(self) : return self.cptr.row_sigma 

    @property
    def col_sigma(self) : return self.cptr.col_sigma 

    @property
    def row_min  (self) : return self.cptr.row_min   

    @property
    def row_max  (self) : return self.cptr.row_max   

    @property
    def col_min  (self) : return self.cptr.col_min   

    @property
    def col_max  (self) : return self.cptr.col_max   

    @property
    def bkgd     (self) : return self.cptr.bkgd      

    @property
    def noise    (self) : return self.cptr.noise     

    @property
    def son      (self) : return self.cptr.son       

#------------------------------
#------------------------------
#------------------------------
#------------------------------

cdef extern from "psalgos/PeakFinderAlgos.h" namespace "psalgos":
    cdef cppclass PeakFinderAlgos:
         #float  m_r0
         #float  m_dr
         #size_t m_rank
         #size_t m_pixgrp_max_size
         #size_t m_img_size
         #float  m_nsigm

         PeakFinderAlgos(const size_t& seg, const unsigned& pbits) except +

         void setPeakSelectionPars(const float& npix_min
                                  ,const float& npix_max
                                  ,const float& amax_thr
                                  ,const float& atot_thr
                                  ,const float& son_min)


         void peakFinderV3r3[T](const T *data
                               ,const uint16_t *mask
                               ,const size_t& rows
                               ,const size_t& cols
                               ,const size_t& rank
	                       ,const double& r0
	                       ,const double& dr
	                       ,const double& nsigm)

         const Peak& peak(const int& i)

         const Peak& peakSelected(const int& i)

         const vector[Peak]& vectorOfPeaks()

         const vector[Peak]& vectorOfPeaksSelected()

         void localMaxima    (uint16_t *arr2d, const size_t& rows, const size_t& cols)

         void localMinima    (uint16_t *arr2d, const size_t& rows, const size_t& cols)

         void connectedPixels(uint32_t *arr2d, const size_t& rows, const size_t& cols)

         void pixelStatus    (uint16_t *arr2d, const size_t& rows, const size_t& cols)
 
#------------------------------

cdef class peak_finder_algos :
    """ Python wrapper for C++ class. 
    """
    cdef PeakFinderAlgos* cptr  # holds a C++ pointer to instance

    cdef uint16_t rows, cols

    def __cinit__(self, seg=0, pbits=0):
        #print "In peak_finder_algos.__cinit__"
        self.cptr = new PeakFinderAlgos(seg, pbits)

    def __dealloc__(self):
        #print "In peak_finder_algos.__dealloc__"
        del self.cptr


    def set_peak_selection_parameters(self\
                                     ,const float& npix_min\
                                     ,const float& npix_max\
                                     ,const float& amax_thr\
                                     ,const float& atot_thr\
                                     ,const float& son_min) :
        self.cptr.setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min)


    def peak_finder_v3r3(self\
                        ,nptype2d data\
                        ,np.ndarray[uint16_t, ndim=2, mode="c"] mask\
                        ,const size_t& rank\
                        ,const double& r0\
                        ,const double& dr\
                        ,const double& nsigm) :
        self.cptr.peakFinderV3r3(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, r0, dr, nsigm)
        self.rows = data.shape[0]
        self.cols = data.shape[1]


    def peak(self, int i=0) : return py_peak.factory(self.cptr.peak(i))


    def peak_selected(self, int i=0) : return py_peak.factory(self.cptr.peakSelected(i))


    def list_of_peaks(self) :
        cdef vector[Peak] peaks = self.cptr.vectorOfPeaks()	
        return [py_peak.factory(p) for p in peaks]


    def list_of_peaks_selected(self) :
        cdef vector[Peak] peaks = self.cptr.vectorOfPeaksSelected()
        return [py_peak.factory(p) for p in peaks]


    def local_maxima(self) :
        sh = (self.rows, self.cols)
        cdef np.ndarray[np.uint16_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint16))
        self.cptr.localMaxima(&arr2d[0,0], self.rows, self.cols)
        return arr2d


    def local_minima(self) :
        sh = (self.rows, self.cols)
        cdef np.ndarray[np.uint16_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint16))
        self.cptr.localMinima(&arr2d[0,0], self.rows, self.cols)
        return arr2d


    def connected_pixels(self) :
        sh = (self.rows, self.cols)
        cdef np.ndarray[np.uint32_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint32))
        self.cptr.connectedPixels(&arr2d[0,0], self.rows, self.cols)
        return arr2d


    def pixel_status(self) :
        sh = (self.rows, self.cols)
        cdef np.ndarray[np.uint16_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint16))
        self.cptr.pixelStatus(&arr2d[0,0], self.rows, self.cols)
        return arr2d


#    @property
#    def r0(self) : return self.cptr.m_r0

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
