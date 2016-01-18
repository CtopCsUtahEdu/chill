--CUBLAS 2 MM Multiply

--This function form intializes "CUDAIZE v2" versus "CUDAIZE v1" if you
--call init() and use global variables to specify procedure and loop

--Second parameter is procedure # and third is loop #
init("nbody.c", "nbody_cpu" , 0) 

dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,
                     --copy_to_shared methods
NBODIES=16384


--Tj=128 CHANGE FOR BEST..... BEST IS 64BLOCKS 128THREADS
--Ti=256
Tj=64
Ti=32
Tjjj=1
Tiii=1
Tn=0.1
--normalize_index("j")
--
--print_code()
--normalize_index("n")
-- TILE COMMANDS ZEROOOOOOOOOOO:3
--tile_by_index({"i","j"},{Ti,Tj},{l1_control="ii",l2_control="jj"},{"ii","jj","i","j"})--CU=-1
tile_by_index({"i"},{Ti},{l1_control="ii"},{"ii","i","j"})--CU=-1
--normalize_index("i")
--tile_by_index({"n"},{Tn},{l1_control="nn"},{"jj","ii","nn","j","i","n"})--CU=-1

--tile_by_index({"j","i"},{Tjjj,Tiii},{l1_control="jjj",l2_control="iii"},{"jj","ii","nn","jjj","j","iii","i","n"})--CU=3
--tile_by_index({"j"}, {Tn}, {l1_control="j",l1_tile="jjj"}, {"ii", "jj", "nn","jjj","j","i","n"})
--tile_by_index({"i"}, {Ti/2}, {l1_control="iii"}, {"ii","iii", "jj","i","j"})
--print_code()
cudaize("kernel_GPU",{oldpos=4*NBODIES,oldpos1=4*NBODIES,oldvel=4*NBODIES,force=4*NBODIES,newpos=4*NBODIES,newvel=4*NBODIES},{block={"ii"}, thread={"i"}})--CU=3
print_code()
--tile(0,6,6)
--copy_to_shared("tx","oldpos",-16)
--copy_to_registers("j","oldpos")
--copy_to_registers("j","oldpos1")
--copy_to_registers("j","force")

--copy_to_texture("oldpos")
--tile(1,3,3)
--tile(2,3,3)

print_code()
--unroll_to_depth(1)
--
--tile(2,3,3)
--unroll(2,3,0)
--unroll(0,5,0)
--print_code()
