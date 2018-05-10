# Extractiong data dependence relations of outer most loop
# for Forward Solve CSC code

from chill import *

source('dep_extraction_fs_csc.c')
destination('dep_extraction_fs_cscmodified.c')
procedure('fs_csc')

loop(0)
original()
print_dep_ufs(1,1)


