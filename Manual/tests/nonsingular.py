from chill import *

source('smallperm.c')
procedure('f')
loop(0)
known(['n > 0', 'm > 1'])
print_code()
print_dep()
nonsingular([[1, -1], [1, 1]])
print_code()
print_dep()
