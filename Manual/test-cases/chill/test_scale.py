from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

known(['ambn > 0', 'an > 0', 'bm > 0'])
distribute([0,1],1)
scale([1],1,4)
scale([1],2,4)
print_code()
