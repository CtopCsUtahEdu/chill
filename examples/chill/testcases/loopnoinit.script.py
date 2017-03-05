from chill import *

#>SKIP

source('loopnoinit.c')
destination('loopnoinitmodified.c')

procedure('main')

loop(0)

# fully unroll the loop
unroll(0,1,0)
print_code()

