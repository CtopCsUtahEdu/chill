from chill import *

source('bigperm.c')
procedure('f')
#format: rose
loop(0)
known(['n1 > 0', 'n2 > 0', 'n3 > 0', 'n4 > 0', 'n5 > 0'])
skew([0], 3, [2, 4, 6])
print_code()
# print_code()
