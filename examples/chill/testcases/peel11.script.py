#
#  example from CHiLL manual page 13
#
#  peel 4 statements from the front of innermost loop
#

from chill import *

source('peel9101112.c')
destination('peel11modified.c')

procedure('mm')

loop(0)

peel(1,2,4)  # statement 1, loop 2 (middle, for j), 4 statements from beginning

