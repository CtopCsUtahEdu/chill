#
# Test unroll
#

from chill import *

source('unroll1.c')
destination('unroll1cmodified.c')
procedure('foo')

# fully unroll a loop with known iteration count
loop(0)
original()
unroll(0,1,20)
#print
#print space
