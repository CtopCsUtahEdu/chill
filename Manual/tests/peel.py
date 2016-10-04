from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

known(['ambn > 4', 'an > 0', 'bm > 0'])
peel(0, 3)
print_code()
