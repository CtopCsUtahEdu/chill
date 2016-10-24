from chill import *
source('wave.c')
procedure('f')
loop(0)
known(['n > 0'])
print_dep()
print "------------"
nonsingular([[0, 1], [1, 0]])
print_dep()
print_code()
