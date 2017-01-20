#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

## Test Harness flags:

from chill import *

source('fuseexample.c')
destination('fuseexamplemodified.c')

procedure('mm')

# fuse example from the Chill manual 
loop(0, 1)
fuse([0,1], 1)




