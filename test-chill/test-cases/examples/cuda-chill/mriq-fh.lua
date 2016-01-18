--CUBLAS 2 MM Multiply

--This function form intializes "CUDAIZE v2" versus "CUDAIZE v1" if you
--call init() and use global variables to specify procedure and loop

--Second parameter is procedure # and third is loop #
init("mriq-fh.c", "mriFH_cpu", 0) 

dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,
                      --copy_to_shared methods
N=32768
M=256
Tx=256


print_code()
--permute(0,{"j","i"})
--tile_by_index({"j","i"}, {TI,TJ}, {l1_control="jj", l2_control="ii"}, {"jj","ii", "j", "i"})
tile_by_index({"x"},{Tx},{l1_control="xx"},{"xx","x","k"})
--tile_by_index({"x"},{16},{l1_control="xx1"},{"xx","x","xx1","k"})
--tile_by_index({"j"}, {TI}, {l1_control="jj"}, {"ii","jj", "j", "i"})
--tile_by_index({"i"}, {TI}, {l1_control="ii"}, {"ii", "i", "j"})
print_code()

normalize_index("x")
--normalize_index("i")
print_code()
--tile_by_index({"i"}, {TI}, {l1_control="iii",l1_tile="i"}, {"ii","jj", "iii","j","i"})
--print_code()
--cudaize("Kernel_GPU", {x=N,y=N,z=N,Qr=N,Qi=N,kVals=M},{block={"jj"}, thread={"j"}})
cudaize("kernel_GPU",{dx=N,dy=N,dz=N,iRho=M,kx=M,ky=M,kz=M,rFHref=N,iFHref=N,rRho=M},{block={"xx"}, thread={"x"}})
--copy_to_shared("tx","iRho",-16)
--copy_to_shared("tx","dz",1)
--copy_to_shared("tx","rRho",-16)
--copy_to_registers("tx","rFHref")
--copy_to_registers("tx","rRho")
--copy_to_registers("tx","iRho")
--copy_to_registers("tx","kx")
--copy_to_registers("tx","dx")
--copy_to_registers("tx","ky")
--copy_to_registers("tx","dy")
--copy_to_registers("tx","kz")
--copy_to_registers("tx","dz")
--copy_to_registers("tx","iFHref")
--copy_to_texture("rRho")
--copy_to_texture("kx")
--copy_to_texture("dx")
--copy_to_texture("ky")
--copy_to_texture("dy")
--copy_to_texture("kz")
--copy_to_texture("dz")
--copy_to_texture("iRho")
--print_code()--]]
--unroll(0,4,0)
--copy_to_constant_no_tile("kx")
--copy_to_constant_no_tile("ky")
--copy_to_constant_no_tile("kz")
--copy_to_constant_no_tile("rRho")
--copy_to_constant_no_tile("iRho")

--unroll_to_depth(1)
print_code()
--[[
copy_to_Texture("rRho")
copy_to_Texture("kx")
copy_to_Texture("dx")
copy_to_Texture("ky")
copy_to_Texture("dy")
copy_to_Texture("kz")
copy_to_Texture("dz")
copy_to_Texture("iRho")
--unroll_to_depth(2)
--]]
