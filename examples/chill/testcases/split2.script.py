

#
#  example from CHiLL manual page 20
#
#  split
#

from chill import *

source('split.c')
destination('split2modified.c')

procedure('mm')

loop(0)

known('an   > 0')
known('bm   > 0')
known('ambn > 10')

split( 1, 3, 'L3 < 7' )



