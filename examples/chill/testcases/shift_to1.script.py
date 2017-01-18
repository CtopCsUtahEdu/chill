

#
#  example from CHiLL manual page 20
#
#  split
#


from chill import *

source('shift_to.c')
destination('shift_to1modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

shift_to( 1, 1, 4 )



