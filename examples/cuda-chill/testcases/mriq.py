#--CUBLAS 2 MM Multiply

#--This function form intializes "CUDAIZE v2" versus "CUDAIZE v1" if you
#--call init() and use global variables to specify procedure and loop

#--Second parameter is procedure # and third is loop #
from chill import *
execfile("cudaize.py")

destination("mriqmodified.cu")
read_IR("mriq.c", "ComputeQCPU")

N=32768
M=3072
TI=128
TJ=128

permute(0, ["j", "i"])

tile_by_index(["j","i"], [TI,TJ], {'l1_control': "jj", 'l2_control':"ii"}, ["jj","ii", "j", "i"], None)
#tile_by_index(["i"],     [TJ],    {'l1_control': "ii", 'l1_tile': "i"},    ["ii", "j","i"],       None)
#tile_by_index(["j"],     [TI],    {'l1_control': "jj"},                    ["ii","jj", "j", "i"], None)
#tile_by_index(["i"],     [TI],    {'l1_control': "ii"},                    ["ii", "i", "j"],      None)
#--print_code()

normalize_index("j")
normalize_index("i")
#--print_code()
#--tile_by_index({"i"}, {TI}, {l1_control="iii",l1_tile="i"}, {"ii","jj", "iii","j","i"})
#--print_code()

cudaize(0, "Kernel_GPU", {'x':N,'y':N,'z':N,'Qr':N,'Qi':N,'kVals':M}, ["jj"], ["j"], [])

##copy_to_shared("tx","kVals",1)
#--copy_to_shared("tx","x",1)
#--copy_to_shared("tx","y",1)
#--copy_to_shared("tx","z",1)

#--copy_to_texture("kVals")
#--datacopy(0, 3, "kVals", {"tt","t"},false,0,1,-16,true)
#--print_code()
#--datacopy_privatized(0,"tx","kVals",{"tx"})
#--copy_to_registers("tx","kVals")
copy_to_registers("ii","x")
copy_to_registers("ii","y")
copy_to_registers("ii","z")
copy_to_registers("ii","Qi")
copy_to_registers("ii","Qr")
"""datacopy_privatized(0,"tx","x",{"tx"})
datacopy_privatized(0,"tx","y",{"tx"})
datacopy_privatized(0,"tx","z",{"tx"})
datacopy_privatized(0,"tx","Qi",{"tx"})
datacopy_privatized(0,"tx","Qr",{"tx"})


"""
#--unroll(0,5,64)
print_code()
#--unroll_to_depth(1) #--won't unroll past thread/loop mapping, unrolls up to two loop levels
