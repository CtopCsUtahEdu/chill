from chill import *

source('dep_extraction_gs_bcsr.c')
destination('dep_extraction_gs_bcsrmodified.c')
procedure('gs_bcsr')

loop(0)
#original()
print_dep_ufs('dep_extraction_gs_bcsr.dep', '', '')
