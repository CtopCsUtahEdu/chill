from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')
shift([1],1,4)
print_code()
