#!/usr/bin/python

import struct

data = list(range(15)) + list(range(10)) + [0]*6
bindata = ''.join([struct.pack('f',n) for n in data])
with open('mm.in.data','wb') as f:
    f.write(bindata)

