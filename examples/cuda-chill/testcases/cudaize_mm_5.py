from chill import *
execfile("cudaize.py")

destination("modifiedcudaize_mm_5.cu")
read_IR("mm.c", "normalMM")


cudaize(0, "kernel_gpu", {'a':1024**2,'b':1024**2,'c':1024**2}, ['i', 'j'], ['k'], [])


