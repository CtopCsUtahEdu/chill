#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

## Test Harness flags:

from chill import *

source('p49.c')
destination('p49modified.c')

procedure('foo')



loop(0)
original()

distribute([0],1) 

print_dep() 

print_structure()

