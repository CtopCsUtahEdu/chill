from chill import *

source('jacobi_box_4_64.c')
procedure('smooth_box_4_64')
loop(0)

original()
print()


skew([0,1,2,3,4,5],2,[2,1])
print()

permute([2,1,3,4])
print()


distribute([0,1,2,3,4,5],2)
print()


stencil_temp(0)
