from chill import *
execfile("cudaize.py")

destination("modifiedcudaize_mm.cu")
read_IR("mm.c", "normalMM")


cudaize(0, "kernel_gpu", {'a':1048576,'b':1048576,'c':1048576}, ['i'], [], [])


