#
# Test unroll
#

from chill import *

source('unroll1.c')
destination('unroll1bmodified.c')
procedure('foo')

# fully unroll a loop with known iteration count
loop(0)
original()
unroll(0,1,6)
#print
#print space
