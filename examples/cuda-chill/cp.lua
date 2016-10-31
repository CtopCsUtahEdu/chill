--CUBLAS 2 MM Multiply

--This function form intializes "CUDAIZE v2" versus "CUDAIZE v1" if you
--call init() and use global variables to specify procedure and loop

--Second parameter is procedure # and third is loop #
init("cp.c", "cenergy_cpu", 0) 

dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,
                     --copy_to_shared methods
V=512
N=4000
N=1

Tj=32
Ti=16
Tii=16
Tjj=16

--normalize_index("j")
--normalize_index("i")
print_code()
normalize_index(0,"n")
-- TILE COMMANDS ZEROOOOOOOOOOO:3
--permute(0,{"i","j","n"})
--tile_by_index({"i","j"},{Ti,Tj},{l1_control="ii",l2_control="jj"},{"ii","jj","i","j","n"})--CU=-1
tile_by_index( {"j","i"},{Tj,Ti},{l1_control="jj",l2_control="ii"},{"jj","ii","j","i","n"})--CU=-1
--tile_by_index({"n"},{Tn},{l1_control="nn"},{"jj","ii","nn","j","i","n"})--CU=-1

--tile_by_index({"j","i"},{Tjjj,Tiii},{l1_control="jjj",l2_control="iii"},{"jj","ii","nn","jjj","j","iii","i","n"})--CU=3
--tile_by_index({"i","j"},{Tii,Tjj},{l1_control="iii",l2_control="jjj"},{"ii","jj","i","iii","j","jjj","n"})--CU=3
--tile_by_index({"j"}, {Tn}, {l1_control="j",l1_tile="jjj"}, {"ii", "jj", "nn","jjj","j","i","n"})
--tile_by_index({"i"}, {Tii}, {l1_control="iii",l1_tile="i"}, {"ii", "jj", "iii","i","j","n"})
print_code()
cudaize( 0, "kernel_GPU",{atoms=N*4,energy=V*V*1},{block={"jj","ii"}, thread={"j","i"}})--CU=3
--cudaize("kernel_GPU",{atoms=N*4,energy=V*V*1},{block={"ii","jj"}, thread={"i","j"}})--CU=3
print_code()
copy_to_shared( "tx","atoms",-16)
copy_to_registers( "tx","energy")
--copy_to_texture("atoms")
--unroll_to_depth(1)
--unroll(0,9,0)
--unroll(0,5,0)

--unroll(0,8,256)
print_code()
