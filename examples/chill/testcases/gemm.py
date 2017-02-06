# Optimizing matrix multiply for large array size on Intel machine.

from chill import *

source('gemm.c')
procedure('gemm')
destination('gemmmodified.c')
loop(0)

TI = 128
TJ = 8
TK = 512
UI = 2
UJ = 2
       
permute([3,1,2])
tile(0,2,TJ)
tile(0,2,TI)
tile(0,5,TK)
datacopy(0,3,'A',False,1)
datacopy(0,4,'B')
unroll(0,4,UI)
unroll(0,5,UJ)
