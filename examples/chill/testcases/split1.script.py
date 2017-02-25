

#
#  example from CHiLL manual page 20
#
#  split
#

from chill import *

source('split.c')
destination('split1modified.c')

procedure('mm')

loop(0)

known(' ambn > 0 ')
known(' an > 0 ')
known(' bm > 10 ')

split( 1, 2, 'L2 < 5' )

