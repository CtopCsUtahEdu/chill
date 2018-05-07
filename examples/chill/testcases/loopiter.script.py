from chill import *

#>SKIP

source('loopiter.c')
destination('loopitermodified.c')

procedure('main')

loop(0)

# fully unroll the loop
unroll(0,1,0)
print_code()

