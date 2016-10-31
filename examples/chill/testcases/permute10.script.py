

#
#  example from CHiLL manual page 14
#
#  permute 3 loops
#

from chill import *

source('permute8910.c')
destination('permute10modified.c')

procedure('mm')

loop(1)

known('ambn > 0')
known('an > 0')
known('bm > 0')

permute([3,1,2])


