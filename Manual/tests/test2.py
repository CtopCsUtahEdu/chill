from chill import *

source('mm.c')
procedure('mm')
loop(0)
known('ambn > 0')
known('an > 0')
known('bm > 0')

print "original"
print_code()

print "\nafter distribute"
distribute([0,1], 1)
print_code()
print_dep()
print_space()

print "\nafter permute"
#tile(1, 3, 10)
permute([1], [3, 2, 1])
#permute(1, 2, [3, 2, 1])
print_code()
