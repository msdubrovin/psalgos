#!/usr/bin/env python
#------------------------------

import psalgos
import numpy as np

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
#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 'See tests in examples, e.g.,\n  python psalgos/examples/ex-02-localextrema.py 3'

#------------------------------
