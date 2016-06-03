init("_orio_chill_0.c","local_grad_3",0)
dofile("cudaize.lua")

N=10
J=10
NELT=1000
M=10
I=10
L=10
K=10
tile_by_index(0,{},{},{},{"dummyLoop","nelt","n","k","m","l"})
cudaize(0,"local_grad_3_GPU_0",{U=NELT*N*M*L,ur=NELT*N*J*K,us=NELT*N*M*K,Dt=I*N,ut=NELT*I*J*K,D=L*K},{block={"nelt","n"},thread={"k","m"}},{})

unroll(0,6,10)

