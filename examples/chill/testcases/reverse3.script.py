

#
#  example from CHiLL manual page 15
#
#  reverse a loop 
#

from chill import *

source('reverse.c')
destination('reverse3modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

distribute( [0,1], 1 )
reverse( [1], 1 )
reverse( [1], 2 )


