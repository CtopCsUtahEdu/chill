#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

from chill import *

source('p46.c')
destination('p46modified.c')

procedure('foo')



loop(0)
original()

print_dep() 

print_structure()
