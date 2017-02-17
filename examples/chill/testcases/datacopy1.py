from chill import *

source('datacopy12.c')
destination('datacopy1modified.c')
procedure('mm')

loop(0)

original()
tile(0,3,16)
datacopy(0,4,'A')
print_code()
