from chill import *

source('datacopy34.c')
destination('datacopy3modified.c')
procedure('mm')

loop(0)

original()
tile(0,3,16)
datacopy(0,4,'C')
print_code()
