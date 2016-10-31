#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

from chill import *

source('p40.c')
destination('p40modified.c')

procedure('foo')



loop(0)
original()

print_dep()


