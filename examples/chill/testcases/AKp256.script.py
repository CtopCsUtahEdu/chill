#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

from chill import *

source('AKp256.c')
destination('AKp256modified.c')

procedure('foo')

# page 256
# fuse 2 identically iterated loops
loop(0, 1)
fuse([0,1], 1)

