

#
#  example from CHiLL manual page 16
#
#  scale a loop 
#

from chill import *

source('scale.c')
destination('scale3modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

distribute( [0,1], 1 )
scale( [1], 2, 4 )


