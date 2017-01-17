from chill import *

## Test Harness flags:

source('include.c')
destination('includemodified.c')

procedure('main')
loop(0)


original()
print_code()


