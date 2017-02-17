from chill import *

source('datacopy34.c')
destination('datacopy4modified.c')
procedure('mm')

loop(0)

original()
tile(0,3,16)
datacopy([(0,[0,1])],4)
print_code()
