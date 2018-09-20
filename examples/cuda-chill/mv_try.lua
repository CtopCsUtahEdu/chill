init("mv_try.c","normalMV",0)
dofile("cudaize.lua") --defines custom tile_by_index, copy_to_registers,
                      --copy_to_shared methods

TI=96

N=4096


tile_by_index(0,{"i"}, {TI}, {l1_control="ii"}, {"ii", "i", "j"})
cudaize(0,"mv_GPU", {a=N, b=N, c=N*N},
        {block={"ii"}, thread={"i"}})

print_code(0)
