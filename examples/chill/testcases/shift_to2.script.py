

#
#  example from CHiLL manual page 18
#
#  shift a loop 
#

## Test Harness flags:

from chill import *

source('shift_to.c')
destination('shift_to2modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an   > 0')
known('bm   > 3')

shift_to( 1, 2, 3 )



