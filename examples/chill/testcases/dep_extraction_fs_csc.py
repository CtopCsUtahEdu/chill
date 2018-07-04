from chill import *

source('dep_extraction_fs_csc.c')
destination('dep_extraction_fs_cscmodified.c')
procedure('fs_csc')

loop(0)
#original()
print_dep_ufs('dep_extraction_fs_csc.dep', '', '1')


