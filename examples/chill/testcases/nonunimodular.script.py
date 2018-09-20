## Test various nonunimodular transformations using nonsingular
## matrices. Using a matrix is acutally a deprecated way since it only
## works on perfect loop nest. A general approach in CHiLL is to use
## skew/shift directly in combination. Nevertheless, these examples
## showcase the approach of using a general polyhedra scanning backend
## CodeGen+ (there are incorrect code outputs in the referred papers
## below though I didn't verify their root causes).

from chill import *

source('nonunimodular.c')
destination('nonunimodularmodified.c')
procedure('foo')

##############################################################################

loop(0)

## Example 1 from "Beyond unimodular transformations, J. Ramanujam,
## The Journal of Supercomputing, Vol.9, No.4, Dec. 1995".
#
# Python test for the correctness of output in the above paper:
# for i in range(2, 2*n+1):
#     for j in range(max(0,i-2*n)+i%2, min(i-2,2*n-i)-i%2+1,2):
#         print (i-j)/2,(i+j)/2
#
# Please note intMod in CodeGen+'s output defined as nonnegative.

nonsingular([[1,1],[-1,1]])
print_code()


##############################################################################

loop(1)

## Fig. 1 and 4 from "Loop Transformation Using Nonunimodular Matrices,
## A. Fernandez, et. al., IEEE Transactions on Parallel and Distributed
## Systems, Vol.6, No.8, Aug 1995".
#
# Python test for the correctness of output in the above paper:
# gap2 = 0 
# for j1 in range(0,15):
#     Lt = max(2*j1-12,int(math.ceil(float(j1)/2)))
#     Ut = min(int(math.floor(18+float(j1)/2)),2*j1)
#     for j2 in range(int(math.ceil(float(Lt-gap2)/3))*3+gap2, Ut+1, 3):
#         print (2*j1-j2)/3, (2*j2-j1)/3
#     gap2 = (gap2+2) % 3
#
# Please note intMod in CodeGen+'s output defined as nonnegative.

nonsingular([[2,1],[1,2]])
print_code()


##############################################################################

loop(2)

## Example 4 from "Automating Non-Unimodular Loop Transformations for
## Massive Parallelism, Jingling Xue, Parallel Computing, Vol.20, No.5,
## May 1994".

nonsingular([[1,1,1],[1,0,-1],[0,1,-1]])
print_code()

##############################################################################

loop(3)

## Fig. 1 from "A Singular Loop Transformation Framework Based on
## Non-Singular Matrices, W. Li and K. Pingali, International Journal
## of Parallel Programming, Vol.22, No.2, April 1994"

nonsingular([[-2,4],[1,1]])
print_code()

