print "\ncoofirstonly.lua calling init()"
   io.flush()
init("spmv.cpp", "spmv", 0)
print "coofirstonly.lua calling init() DONE\n"
   io.flush()


print "\ncoofirstonly.lua calling dofile(cudaize.lua)"
   io.flush()
dofile("cudaize.lua")
print "coofirstonly.lua calling dofile(cudaize.lua) DONE\n"
   io.flush()


NNZ=1666
N=494
Ti = 1024
Tj = 256
Tk = 32
PAD_FACTOR=64
TILE_FACTOR_FOR_2ND_LEVEL = Tj
segment = "c.i"
shared_memory=1
NO_PAD=0
ASSIGN_THEN_ACCUMULATE=1
stmt_to_reduce = {}

--flatten loop levels 1 and 2 with NNZ being uninterpreted omega function name
print "\ncoofirstonly.lua calling coalesce_by_index()"
   io.flush()
first_kernel = coalesce_by_index(0,"coalesced_index",{"i","j"}, "c")
print "coofirstonly.lua calling coalesce_by_index() DONE\n"
   io.flush()


--tile for number of non zeros per CUDA block (1024)
print "\ncoofirstonly.lua calling tile_by_index() per block"
   io.flush()
tile_by_index(first_kernel,{"coalesced_index"},{Ti},{l1_control="block"},{"i","j","block","coalesced_index"})
print "coofirstonly.lua calling tile_by_index() DONE\n"
   io.flush()

--tile for number of nonzeros per warp (256)
print "\ncoofirstonly.lua calling tile_by_index() per warp"
   io.flush()
tile_by_index(first_kernel,{"coalesced_index"},{Tj},{l1_control="warp"},{"i", "j", "block","warp","coalesced_index"})
print "coofirstonly.lua calling tile_by_index() DONE\n"
   io.flush()

--tile for warp size(32) to get(256/32= 8) iterations
   print "\ncoofirstonly.lua calling tile_by_index() for warp size(32)  (by_warp)"
   io.flush()
tile_by_index(first_kernel,{"coalesced_index"},{Tk},{l1_control="by_warp"},{"i", "j", "block", "warp","by_warp","coalesced_index"})
print "coofirstonly.lua calling tile_by_index() DONE\n"
   io.flush()


--Designate block and thread dimensions via cudaize 
print "\ncoofirstonly.lua calling cudaize()"
   io.flush()
cudaize5arg(first_kernel,
        "spmv_GPU",
        { a=NNZ,x=N,y=N,col=NNZ,temp=NNZ, c_j=NNZ, c_i=NNZ },
        {block={"block"}, thread={"coalesced_index", "warp"}},
        {"_P_DATA1", "_P_DATA2"})
print "coofirstonly.lua calling cudaize() DONE\n"
   io.flush()

--reduction
print "\ncoofirstonly.lua calling reduce_by_index()  by_warp"
   io.flush()
reduce_by_index(first_kernel,{"tx"},"segreduce_warp",{"by_warp"}, {})
print "coofirstonly.lua calling reduce_by_index() DONE\n"
   io.flush()



