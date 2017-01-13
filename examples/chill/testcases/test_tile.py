from chill import *

source('mm.c')
procedure('mm')
#format: rose
loop(0)

#known('ambn > 0')
#known('an > 0')
#known('bm > 0')
#tile(1, 1, 4, 1)
#tile(1, 3, 4, 2)
tile(0,2,4)
print_code()
