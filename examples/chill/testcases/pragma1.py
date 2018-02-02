

from chill import *

source('distribute.c')
destination('pragma1modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

distribute([0,1], 1)

pragma(0, 2, 'some pragma')



