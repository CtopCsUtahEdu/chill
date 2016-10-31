#
#  example from CHiLL manual page 13
#
#  peel 4 statements from the END of innermost loop
#

from chill import *

source('peel9101112.c')
destination('peel10modified.c')

procedure('mm')

loop(0)

#known( ambn > 0 )
peel(1, 3,-4)  # statement 0, loop 3 (innermost, for n), 4 statements from END


