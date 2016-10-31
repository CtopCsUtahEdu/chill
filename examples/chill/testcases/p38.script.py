#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

from chill import *

source('p38.c')
destination('p38modified.c')

procedure('zzfoo')


loop(0)
original()

unroll(0, 1, 0)

#print dep 


