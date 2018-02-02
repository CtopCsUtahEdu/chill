

from chill import *

source('distribute.c')
destination('omp_for1modified.c')

procedure('mm')

loop(0)

known('ambn > 0')
known('an > 0')
known('bm > 0')

distribute([0,1], 1)

omp_for(1, 2)


