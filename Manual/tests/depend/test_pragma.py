from chill import *

source('mm.c')
procedure('mm')
loop(0)
permute([3,1,2])
print_code()
print_space()


print("num statements = ", num_statements())
print_dep()
