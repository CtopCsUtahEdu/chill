#
#  example from CHiLL manual page 13 (ALMOST)  KNOWN LOOP BOUNDS
#
#  peel 4 statements from the BEGINNING of innermost loop
#

from chill import *

source('peel5678.c')
destination('peel5modified.c')

procedure('mm')

loop(0)

peel(1,3, 4)  # statement 1, loop 3 (innermost, for n), 4 statements from BEGINNING

