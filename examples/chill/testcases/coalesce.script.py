from chill import *


## coalesce function is not working
#>SKIP

source('coalesce.c')
procedure('main')
destination('coalescemodified.c')
loop(0)



original()
coalesce(0, "coalesced_index", [1,2], "c")
#coalesce_by_index(0,"coalesced_index",{"i","j"}, "c")

#print
