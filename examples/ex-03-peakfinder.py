#!/usr/bin/env python
#------------------------------

from psalgos.pypsalgos import *
from time import time

#------------------------------

def test01():

    import numpy as np
    from pyimgalgos.GlobalUtils import print_ndarr
    print 'test of peak_finder_v3r3'
    
    sh = (1000,1000)
    #sh = (200,300)
    #sh = (185,388)#, (11,5)
    mu, sigma = 200, 25
    data = np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.float64)
    #data = np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.double)
    mask = np.ones(sh, dtype=np.uint16)

    print_ndarr(data, 'input data')
    print_ndarr(mask, 'input mask')

    t0_sec = time()
    peak_finder_v3r3(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=3, npixmin=2, son=8)
    print 'peak_finder_v3r3: img.shape=%s consumed time = %.6f(sec)' % (str(sh), time()-t0_sec)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '1' : test01()
    elif tname == '2' : test02()
    elif tname == '3' : test03()
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
