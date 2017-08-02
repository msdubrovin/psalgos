#!/usr/bin/env python
#------------------------------

import psalgos

#------------------------------

def test01():
    print 'call pure python'

#------------------------------

def test02():
    print 'call psalgos.fib(90)'
    print psalgos.fib(90)

#------------------------------
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
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
