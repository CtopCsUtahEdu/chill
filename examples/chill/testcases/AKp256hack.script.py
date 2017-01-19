#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

## Test Harness flags:

from chill import *

source('AKp256hack.c')
destination('AKp256hackmodified.c')

procedure('foo')

# page 256
# fuse 2 identically iterated loops
loop(0)
fuse([0,1], 2 )




