#
#  example from CHiLL manual page 13
#
#  peel 4 statements from the END of innermost loop
#

## Test Harness flags:

from chill import *

source('peel1234.c')
destination('peel2modified.c')

procedure('mm')

loop(0)

known('ambn > 4')
known('an   > 0')
known('bm   > 0')

peel(1, 3, -4)  # statement 1, loop 3 (innermost, for n), 4 statements from END


