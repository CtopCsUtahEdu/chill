init("mv.cu","normalMV",0)
--defines custom tile_by_index, copy_to_registers,
dofile("cudaize.lua") 

N=129
TI=32
TJ=64

N=1024

--Tile the i and j loop, introducing "ii" as the control loop for the "i"
--tile, "k" for the control loop fo the "j" tile, with the final order
--of {"ii", "k", "i", "j"}

tile_by_index(0, {"i","j"}, 
		{TI,TJ}, {l1_control="ii", l2_control="k"}, 
		{"ii", "k", "i", "j"})

print_code(0)
--normalize_index(0,"i")
--print_code(0)


cudaize(0, "mv_GPU", {a=N, b=N, c=N*N},
        {block={"ii"}, thread={"i"}}, {})


--copy_to_registers(0, "k", "a")


--unroll_to_depth( 1) --won't unroll past thread/loop mapping, unrolls up to two loop levels

