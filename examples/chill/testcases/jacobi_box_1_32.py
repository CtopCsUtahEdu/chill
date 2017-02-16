from chill import *

source('jacobi_box_1_32.c')
destination('jacobi_box_1_32modified.c')
procedure('smooth_box_1_32')

loop(0)


original()
print_code()

stencil_temp(3)
print_code()

print_space()


