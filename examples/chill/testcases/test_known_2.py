from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

known(['ambn > 0', 'an > 0', 'bm > 0'])
print_code()
