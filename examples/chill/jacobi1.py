#
# tiling perfect jacobi loop nest with time step, use
# unimodular transformation first (only applicable to the
# perfect loop nest) to make tiling legal.
#

from chill import *

source('jacobi1.c')
procedure('main')

loop(0)

print_dep()

nonsingular([[1,0],[1,1]])  # unimodular matrix, determinant is one
tile(0,2,64)

print_dep()
print_code()

