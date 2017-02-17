# Old fashion way to tile perfect jacobi loop nest with time step
# using unimodular transformation.

from chill import *

source('jacobi_simple.c')
destination('jacobi_simplemodified.c')
procedure('jacobi')
loop(0)

print_dep()

nonsingular([[1,0],[1,1]])  # unimodular matrix, determinant is one
tile(0,2,64)

print_code()
print_dep()

