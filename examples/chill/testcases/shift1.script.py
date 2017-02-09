#
#  example from CHiLL manual page 17
#
#  shift a loop 
#
from chill import *

source('shift.c')
destination('shift1modified.c')
procedure('mm')

loop(0)

known(' ambn > 0 ')
known(' an > 4 ')
known(' bm > 0 ')

shift( [1], 1, 4 )

