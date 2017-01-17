

#
#  example from CHiLL manual page 18
#
#  shift a loop 
#

## Test Harness flags:

from chill import *

source('shift_to.c')
destination('shift_to3modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an   > 0')
known('bm   > 0')

shift_to( 1, 3, -5 )



