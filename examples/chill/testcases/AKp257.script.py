#
# loop adapted from  "Optimizing Compilers for
# Modern Architectures", by Randy Allen and Ken Kennedy.
#

## Test Harness flags:
#>EXFAIL

source('AKp257.c')
destination('AKp257modified.c')

procedure('foo')

# page 257
# fuse 2 identically iterated loops - BUT ILLEGAL (changes the meaning of the code)
loop(0,1)
fuse([0,1], 1)

