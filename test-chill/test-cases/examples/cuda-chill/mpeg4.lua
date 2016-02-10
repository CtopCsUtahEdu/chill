--CUBLAS 2 MM Multiply

--This function form intializes "CUDAIZE v2" versus "CUDAIZE v1" if you
--call init() and use global variables to specify procedure and loop

--Second parameter is procedure # and third is loop #
init("mpeg4.c", "mpeg4_cpu", 0) 

--dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,copy_to_shared methods
dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,copy_to_shared methods

N=4096
M=4096
W=16

--TI 4ust be <= M
--TJ must be <=TI
Ti=32
Tj=32
Tii=16
Tjj=16
Tk=4
--permute(0,{"j","i","k","l"})
tile_by_index({"i","j"},{Ti,Tj},{l1_control="ii",l2_control="jj"},{"ii","jj","i","j","k","l"})
--tile_by_index({"k","l"},{Tk*2,Tk*2},{l1_control="kk",l2_control="ll"},{"ii","jj","kk","ll","i","j","k","l"})
--print_code()
--tile_by_index({"k","l"},{Tk,Tk},{l1_control="kk",l2_control="ll"},{"ii","jj","i","j","kk","k","ll","l"})
tile_by_index({"i","j"},{Tii,Tjj},{l1_control="iii",l2_control="jjj"},{"ii","jj","iii","i","jjj","j","k","l"})
--print_code()
--normalize_index("j")
--normalize_index("i")
--print_code()
cudaize("kernel_GPU",{curr=W*W,prev=(N+W)*(M+W),result=N*M},{block={"ii","jj"}, thread={"i","j"}})
--print_code()
copy_to_shared("iii","prev",16)

copy_to_registers("jjj","result")

--print_code()
--copy_to_constant_no_tile("curr")
unroll_to_depth(2)
print_code()
print_space()


