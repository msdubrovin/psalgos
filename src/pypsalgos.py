#!/usr/bin/env python
#------------------------------

import psalgos
import numpy as np
from time import time

#------------------------------

def local_minimums_2d(data, mask=None, rank=3, extrema=None) :
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    psalgos.local_minimums(data, _mask, rank, extrema)

#------------------------------

def local_maximums_2d(data, mask=None, rank=3, extrema=None) :
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    psalgos.local_maximums(data, _mask, rank, extrema)

#------------------------------

def local_maximums_rank1_cross_2d(data, mask=None, extrema=None) :
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    psalgos.local_maximums_rank1_cross(data, _mask, extrema)

#------------------------------

def print_matrix_of_diag_indexes(rank=5) : psalgos.print_matrix_of_diag_indexes(rank)

#------------------------------

def print_vector_of_diag_indexes(rank=5) : psalgos.print_vector_of_diag_indexes(rank)

#------------------------------

def peak_finder_v3r3(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=0, npixmin=1, son=8) :
    #t0_sec = time()
    # pbits NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16
    o = psalgos.peak_finder_algos(seg=0, pbits=2) #377) # ~17 microsecond
    npixmax = (2*rank+1)
    npixmax *= npixmax
    o.set_peak_selection_parameters(npix_min=npixmin, npix_max=npixmax, amax_thr=0, atot_thr=0, son_min=son)
    o.peak_finder_v3r3(data, mask, rank, r0, dr, nsigm)
    #print 'peak_finder_v3r3: total time = %.6f(sec)' % (time()-t0_sec)

#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 'See tests in examples, e.g.,\n  python psalgos/examples/ex-02-localextrema.py 3'

#------------------------------
