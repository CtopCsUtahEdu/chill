# Extractiong data dependence relations of outer most loop
# for Incomplete Cholesky CSR (CSC) code

from chill import *

source('dep_extraction_ic0.c')
destination('dep_extraction_ic0modified.c')
procedure('ic0_csr')

loop(0)
original()
print_dep_ufs(1,1)

