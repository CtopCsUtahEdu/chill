# Test unroll-and-jam. The last loop adapted from the simple
# convolution example from p463 of "Optimizing Compilers for Modern
# Architectures", by Randy Allen and Ken Kennedy. Being able to
# unroll-and-jam non-rectangular imperfect loop nest with data
# dependence between those statements is all done in CHiLL's general
# approach.

from chill import *

source('unroll.c')
destination('unrollmodified.c')
procedure('foo')

# fully unroll a loop with known iteration count
loop(0)
original()
unroll(0,1,0)
print_code()

# a strided loop
loop(1)
original()
unroll(0,1,2)
print_code()

# lower and upper bounds are not constant
loop(2)
original()
unroll(0,1,20)
print_code()

# parallelogram iteration space
loop(3)
original()
unroll(0,1,2)
print_code()
