from chill import *
execfile("cudaize.py")

#>SKIP

destination("mmmodified.cu")
read_IR("mm.c", "normalMM")

N=1024
Ti=128
Tj=64
Tk=16
Tii=16
Tjj=16


tile_by_index(["i","j"],[Ti,Tj],{'l1_control':"ii",'l2_control':"jj"},["ii","jj","i","j","k"], None)

tile_by_index(["k"],[Tk],{'l1_control':"kk"},["ii","jj","kk","i","j","k"], None)

tile_by_index(["i","j"],[Tii,Tjj],{'l1_control':"iii",'l2_control':"jjj"},["ii","jj","kk","i","iii","j","jjj","k"], 1)

cudaize(0, "mm_GPU",{'a':1048576,'b':1048576,'c':1048576},["ii","jj"], ["i","j"], [])
copy_to_shared("tx","a",-16)
copy_to_shared("tx","b",-16)
copy_to_registers("kk","c")
#--print_code()
unroll_to_depth(2)
