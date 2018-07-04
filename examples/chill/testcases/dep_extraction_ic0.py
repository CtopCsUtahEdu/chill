from chill import *

source('dep_extraction_ic0.c')
destination('dep_extraction_ic0modified.c')
procedure('ic0_csr')

loop(0)
print_dep_ufs('dep_extraction_ic0.dep', '', '')

