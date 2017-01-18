#
# Test unroll
#
from chill import *

source('unroll2.c')
destination('unroll2amodified.c')
procedure('foo')

# fully unroll a loop with known iteration count
loop(0)
original()
unroll(0,1,0)
#print
#print space

