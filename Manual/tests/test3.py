from chill import *

source('mm2.c')
procedure('mm')
loop(0)
known('ambn > 0')
known('an > 0')
known('bm > 0')

print "original"
print_code()
print_space()
print_dep()

# reverse([0, 1], 1)
# print_code()

