init("mm.c", "normalMM", 0)
dofile("cudaize.lua")
N=1024
Ti=128
Tj=64
Tk=16
Tii=16
Tjj=16




N=1024













tile_by_index(0,{"i","j"},{Ti,Tj},{l1_control="ii",l2_control="jj"},{"ii","jj","i","j","k"})CU=1

tile_by_index(0,{"k"},{Tk},{l1_control="kk"},{"ii","jj","kk","i","j","k"})CU=3

tile_by_index(0,{"i","j"},{Tii,Tjj},{l1_control="iii",l2_control="jjj"},{"ii","jj","kk","i","iii","j","jjj","k"},1)CU=2

cudaize(0,"mm_GPU",{},{block={"ii","jj"}, thread={"i","j"}},{})
copy_to_shared(0,"tx","a",-16)
cur= cur_indices(stmt)
print("Cur indices "..list_to_string(cur))

copy_to_shared(0,"tx","b",-16)
cur= cur_indices(stmt)
print("Cur indices "..list_to_string(cur))
print_space()
copy_to_registers(0,"kk","c")
--unroll_to_depth(2)
