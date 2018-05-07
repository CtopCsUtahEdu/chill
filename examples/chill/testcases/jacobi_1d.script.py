# Tiling stencil code using example of jacobi relaxation. Compare with
# paper "Automatic Tiling of Iterative Stencil Loops" by Zhiyuan Li and
# Yonghong Song, TOPLAS 2004.

from chill import *

source('jacobi_1d.c')
destination('jacobi_1dmodified.c')
procedure('jacobi')
loop(0)

original()

print_code()
print_dep()

shift([1], 2, 1) # L2 <- L2+1
fuse([0,1], 2)  # optional
skew([0,1], 2, [2,1]) # L2 <- 2*L1+L2
tile(0, 2, 32, 1)

print_code()
print_dep()

