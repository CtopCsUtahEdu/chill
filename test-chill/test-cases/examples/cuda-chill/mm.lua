init("mm.c", "normalMM", 0)
dofile("cudaize.lua")
N=1024
Ti=128
Tj=64
Tk=16
Tii=16
Tjj=16




N=1024













tile_by_index({"i","j"},{Ti,Tj},{l1_control="ii",l2_control="jj"},{"ii","jj","i","j","k"})CU=1

tile_by_index({"k"},{Tk},{l1_control="kk"},{"ii","jj","kk","i","j","k"})CU=3

tile_by_index({"i","j"},{Tii,Tjj},{l1_control="iii",l2_control="jjj"},{"ii","jj","kk","i","iii","j","jjj","k"},1)CU=2

cudaize("mm_GPU",{a=1048576,b=1048576,c=1048576},{block={"ii","jj"}, thread={"i","j"}})CU=2
copy_to_shared("tx","a",-16)
copy_to_shared("tx","b",-16)
copy_to_registers("kk","c")
--print_code()
unroll_to_depth(2)
