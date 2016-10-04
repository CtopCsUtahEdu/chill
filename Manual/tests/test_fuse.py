from chill import *

source('dist.c')
procedure('mm')
loop(0, 1)
known(['ambn > 0', 'an > 0', 'bm > 0'])
# distribute([0,1], 1)
# print_code()
fuse([0, 1], 2)
print_code()
