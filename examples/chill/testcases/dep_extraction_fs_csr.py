from chill import *

source('dep_extraction_fs_csr.c')
destination('dep_extraction_fs_csrmodified.c')
procedure('fs_csr')

loop(0)
#original()
print_dep_ufs('dep_extraction_fs_csr.dep', '', '')


