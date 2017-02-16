from chill import *

source("mm.c")
procedure("normalMM")
dofile("cudaize.lua")
N=1024
Ti=64
Tj=64
Tk=16
Tii=16
Tjj=16




N=1024






tile7(0,2,Ti,1,"i","ii",0)
cudaize(0,"mm_GPU",{},{"ii","jj"}, {"i","j"},{})



print_code(0)

