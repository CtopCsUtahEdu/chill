#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

from chill import *

source('p48.c')
destination('p48modified.c')

procedure('foo')



loop(0)
original()

print_dep() 

unroll(0,1,0)
unroll(0,2,0)



