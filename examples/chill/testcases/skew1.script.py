

#
#  example from CHiLL manual page 19
#
#  skew a loop 
#

## Test Harness flags:

from chill import *

source('skew.c')
destination('skew1modified.c')

procedure('f')
loop(0)

known('n > 0')
known('m > 1')

skew( [0], 2, [1,1] )



