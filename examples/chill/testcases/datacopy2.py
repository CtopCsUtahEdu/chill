from chill import *

source('datacopy12.c')
destination('datacopy2modified.c')
procedure('mm')

loop(0)

original()
tile(0,3,16)
datacopy([(0,[2])],4)
print_code()
