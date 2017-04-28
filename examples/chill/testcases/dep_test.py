# Data dependence examples from "Loop Transformations for
# Restructuring Compilers, The Foundations" by Utpal Banerjee. It
# shows that a complete mathematical system Omega+ subsumes previous
# ad-hoc methods.

from chill import *

source('dep_test.c')
destination('dep_testmodified.c')
procedure('foo1')

loop(0)
original()
print_space()
print_dep()

#
# Example 5.7 from p135 
#
loop(1)

original()
print_space()
print_dep()

#
# Example 5.10 from p144
#

loop(2)

original()
print_space()
print_dep()

#
# Example 5.11 from p150 
#
loop(3)

original()
print_space()
print_dep()

#
# Example 5.12 from p156
#
loop(4)

original()
print_space()
print_dep()

