from chill import *

source('mm.c')
procedure('mm')
loop(0)
known(['ambn > 0', 'an > 0', 'bm > 0'])
tile(0,2,4)
print_code()
print_dep()
