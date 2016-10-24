from chill import *

source('vm.c')
procedure('vm')
loop(0)
known('n > 0')
permute([2, 1])
print "original"
print_code()
print_space()
print_dep()

print_dep()
