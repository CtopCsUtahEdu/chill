#!/usr/bin/python

import struct
import numpy as np

with open('mm.out.data','rb') as f:
    data = f.read()

mat = np.array([struct.unpack_from('f',data,n*4) for n in range(len(data)/4)]).reshape((3,2))
print(mat)
