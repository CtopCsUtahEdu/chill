#
# Test unroll
#
from chill import *

source('unroll2.c')
destination('unroll2cmodified.c')
procedure('foo')

# unroll a loop with known iteration count 15 by a factor of 20 (should be the same as complete unroll)
loop(0)
original()
unroll(0,1,20)
#print
#print space

