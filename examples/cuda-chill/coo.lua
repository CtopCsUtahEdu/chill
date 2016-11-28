init("spmv.cpp", "spmv", 0)
dofile("cudaize.lua")
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
first_kernel = coalesce_by_index(0,"coalesced_index",{"i","j"}, "c")

--split flattened loop level to be a perfect multiple of warp size (1024)
last_kernel = split_with_alignment_by_index(first_kernel,"coalesced_index",Ti)

--distribute remainder of splitted statement as it is not cudaized
distribute_by_index({first_kernel,last_kernel},"i")

--tile for number of non zeros per CUDA block (1024)
tile_by_index(first_kernel,{"coalesced_index"},{Ti},{l1_control="block"},{"i","j","block","coalesced_index"})

--tile for number of nonzeros per warp (256)
tile_by_index(first_kernel,{"coalesced_index"},{Tj},{l1_control="warp"},{"i", "j", "block","warp","coalesced_index"})

--tile for warp size(32) to get(256/32= 8) iterations
tile_by_index(first_kernel,{"coalesced_index"},{Tk},{l1_control="by_warp"},{"i", "j", "block", "warp","by_warp","coalesced_index"})

--setup for segmented 2-level reduction
second_kernel = setup_for_segreduce(first_kernel,"warp",{"block","warp","coalesced_index"},segment,PAD_FACTOR,shared_memory, TILE_FACTOR_FOR_2ND_LEVEL, "k", stmt_to_reduce)

--tile splitted remainder, and scalar expand for shared memory reduction
tile_by_index(last_kernel,{"coalesced_index"},{512},{l1_control="block"},{"i", "j", "block",  "coalesced_index"},strided)

scalar_expand_by_index(last_kernel,{"coalesced_index"},segment,shared_memory, NO_PAD, ASSIGN_THEN_ACCUMULATE)
scalar_expand_by_index(last_kernel,{"coalesced_index"},"RHS",shared_memory, NO_PAD, ASSIGN_THEN_ACCUMULATE)

--Designate block and thread dimensions via cudaize 
cudaize5arg(first_kernel,"spmv_GPU",{ a=NNZ,x=N,y=N,col=NNZ,temp=NNZ, c_j=NNZ, c_i=NNZ },{block={"block"}, thread={"coalesced_index", "warp"}},{"_P_DATA1", "_P_DATA2"})
cudaize5arg(second_kernel,"spmv_second_level_GPU",{ a=NNZ,x=N,y=N,col=NNZ,temp=NNZ,c_i=NNZ},{block={}, thread={"warp","block"}}, {"_P_DATA1", "_P_DATA2"})
cudaize5arg(last_kernel,"spmv_final_level_GPU",{ a=NNZ,x=N,y=N,col=NNZ, temp=NNZ, c_j=NNZ, c_i=NNZ},{block={}, thread={"coalesced_index"}}, {})

--reduction
reduce_by_index(first_kernel,{"tx"},"segreduce_warp",{"by_warp"}, {})
reduce_by_index(second_kernel,{"ty","tx"},"segreduce_block",{},{"ty"})
reduce_by_index(last_kernel,{"tx"},"segreduce_block2",{},{"tx"})
reduce_by_index(stmt_to_reduce[1],{"tx"},"segreduce_warp",{},{})



