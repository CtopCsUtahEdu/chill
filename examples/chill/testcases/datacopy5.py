from chill import *

source('datacopy34.c')
destination('datacopy5modified.c')
procedure('mm')

loop(0)

original()
datacopy(0,3,"A")
print_code()
