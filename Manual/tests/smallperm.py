from chill import *

source('smallperm.c')
procedure('f')
#format: rose
loop(0)
known(['n > 0', 'm > 1'])
print_code()
print_dep()
skew([0], 2, [1, 1])
print_code()
print_dep()
