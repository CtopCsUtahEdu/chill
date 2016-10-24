from chill import *

source('dep.c')
procedure('d')
loop(0)
known(['n > 0', 'm > 0'])
#known(['is > 0', 'js > 0'])
print_code()
print_dep()
unroll(0, 2, 4)
