from chill import *

source('dep_extraction_spMVMul_csr.c')
destination('dep_extraction_spMVMul_csrmodified.c')
procedure('spMVMul_csr')

loop(0)
#original()
print_dep_ufs('dep_extraction_spMVMul_csr.dep', '', '')

