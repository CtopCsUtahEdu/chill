from chill import *

source('jacobi_box_1_64.c')
destination('jacobi_box_1_64modified.c')
procedure('smooth_box_1_64')

loop(0)


original()
print_code()

find_stencil_shape(3)
print_code()

print_space()


