#
# tiling imperfect jacobi loop nest, more details in the paper
# "Automatic Tiling of Iterative Stencil Loops" by Zhiyuan Li and
# Yonghong Song, TOPLAS, 2004.
#

from chill import *


source('jacobi2.c')
destination('jacobi2modified.c')
procedure('main')
loop(0)

print_dep()

original()
shift([1], 2, 1)
fuse([0,1], 2)  # optional
skew([0,1], 2, [2,1])
tile(0, 2, 32, 1)

print_dep()
print_code()

