from chill import *
execfile("cudaize.py")

destination("mvmodified.cu")
read_IR("mv.c", "normalMV")

N=1024
Ti=32
Tj=64

tile_by_index(["i", "j"],[Ti, Tj], {'l1_control': "ii", 'l2_control': "k"}, ["ii", "k", "i", "j"], None)

cudaize(0, "mv_GPU", {'a':N, 'b':N, 'c':N*N}, ["ii"], ["i"], [])

copy_to_registers("k", "a")

