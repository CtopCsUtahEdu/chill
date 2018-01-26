from chill import *

## Issue #53
#>SKIP

source('jacobi_box_4_64.c')
destination('jacobi_box_4_64modified.c')
procedure('smooth_box_4_64')

loop(0)


original()
print_code()


skew([0,1,2,3,4,5],2,[2,1])
print_code()

permute([2,1,3,4])
print_code()


distribute([0,1,2,3,4,5],2)
print_code()


stencil_temp(0)
#print
#print space

stencil_temp(5)
#print
#print space

fuse([2,3,4,5,6,7,8,9],1)
#print
fuse([2,3,4,5,6,7,8,9],2)
#print
fuse([2,3,4,5,6,7,8,9],3)
#print
fuse([2,3,4,5,6,7,8,9],4)
#print
