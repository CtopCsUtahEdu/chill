# Basic illustration of loop fusion and distribution.

from chill import *

source('fuse_distribute.c')
destination('fuse_distributemodified.c')
procedure('foo')
loop(0)

# initially fused as much as possible
original()
print_code()

# distribute the first two statements
distribute([0,1], 2)
print_code()

# prepare the third statement for fusion
shift([2], 2, 1)
print_code()

# fuse the last two statements
fuse([1,2],2)
print_code()

