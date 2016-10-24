from chill import *

source('mm.c')
procedure('mm')
loop(0)
known('ambn > 0')
known('an > 0')
known('bm > 0')
print_code()
permute([0, 1], [3,1,2])
print_code()
print_dep()
