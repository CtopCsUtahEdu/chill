#
#  example from CHiLL manual page 13
#
#  peel 4 statements from the END of innermost loop
#

## Test Harness flags:
#>SKIP

from chill import *

source('bug1.c')
destination('bug1modified.c')

procedure('mm')

loop(0)

known('ambn > 4') # this fails ??? 






