#
# Test unroll
#
from chill import *

source('unroll2.c')
destination('unroll2bmodified.c')
procedure('foo')

# unroll a loop (x6) with known iteration count of 15.
# should create a second loop to handle the last 3 remaining iterations
loop(0)
original()
unroll(0,1,6)
#print
#print space

