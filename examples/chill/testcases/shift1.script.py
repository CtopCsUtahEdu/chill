

#
#  example from CHiLL manual page 20
#
#  split
#

from chill import *

source('tile.c')
destination('tile1modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

tile( 0,2,4 )



