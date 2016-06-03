from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

known(['ambn > 4', 'an > 0', 'bm > 0'])
peel(1,3,4)
print_code()
