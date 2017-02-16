#
#  example from CHiLL manual page 13
#
#  peel 4 statements from the END of innermost loop
#

## Test Harness flags:

from chill import *

source('peel1234.c')
destination('peel4modified.c')

procedure('mm')

loop(0)

known('an   > 0')
known('bm   > 4')
known('ambn > 4')

peel(1,2, -4)  # statement 1, loop 2 (middle, for j), 4 statements from END

