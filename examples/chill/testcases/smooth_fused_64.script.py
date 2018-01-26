from chill import *

## Issue #54
#>SKIP

source('smooth_fused_64.c')
destination('smooth_fused_64modified.c')
procedure('main')
loop(0)


known ('ghosts < 5')
known ('ghosts > 3')
known ('K < 65')
known ('K > 63')
known ('J < 65')
known ('J > 63')
known ('I < 65')
known ('I > 63')

original()
print_code()



