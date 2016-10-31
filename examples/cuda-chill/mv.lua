init("mv.c","normalMV",0)
dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,
                      --copy_to_shared methods

N=129
TI=32
TJ=64

N=1024
















--Tile the i and j loop, introducing "ii" as the control loop for the "i"
--tile, "k" for the control loop fo the "j" tile, with the final order
--of {"ii", "k", "i", "j"}
tile_by_index({"i","j"}, {TI,TJ}, {l1_control="ii", l2_control="k"}, {"ii", "k", "i", "j"})
--tile_by_index({"i"}, {TI}, {l1_control="iii"}, {"ii", "k", "iii","i", "j"})
--tile_by_index({"j"}, {TI}, {l2_control="k"}, { "k", "i", "j"})
--tile_by_index({"i"}, {TI}, {l1_control="ii"}, {"ii", "i", "j"})
--print_code()
--Normalize indx will do a tile size of one over the loop level specified
--by the input index. This is useful to get a zero lower bound and hard
--upper bound on a loop instead of it being relative to previous loop
--levels.
--normalize_index("ii")
normalize_index("i")
print_code()

--Cudaize now determines the grid dimentions from the loops themselves
--(the upper bounds of the block and thread loops). It also renames the
--given block and thread loops's indexes to the approviate values from
--the set {"bx","by","tx","ty","tz"}. The second parameter specifies the
--size of the arrays to be copied in the CUDA scaffolding.
cudaize("mv_GPU", {a=N, b=N, c=N*N}, {block={"ii"}, thread={"i"}})

--print_code()

--Does a datacopy, tile, and add_sync to get a shared memory copy

--copy_to_shared("tx", "b", 1)
--copy_to_shared("tx", "c", -16)
--print_code()
--copy_to_texture("b")
--copy_to_texture("c")
copy_to_registers("k", "a")
--print_code()

unroll_to_depth(1) --won't unroll past thread/loop mapping, unrolls up to two loop levels
--copy_to_texture("b")
--print_code()
--unroll(0,5,0)
--print_code()
