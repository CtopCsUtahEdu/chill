#
#  example from CHiLL manual page 13 (ALMOST)  KNOWN LOOP BOUNDS
#
#  peel 4 statements from the END of innermost loop
#

from chill import *

source('peel5678.c')
destination('peel8modified.c')

procedure('mm')

loop(0)

peel(1, 2, -4)  # statement 1, loop 2 (middle, for j), 4 statements from END

