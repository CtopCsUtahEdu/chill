# CHiLL - The Composable High-Level Loop Source-to-Source Translator

CHiLL is a source-to-source translator for composing high level loop transformations to improve the performance of nested loop calculations written in C. CHiLL’s operations are driven by a script which is generated or supplied by the user that specifies the location of the original source file, the function and loops to modify and the transformations to apply.

## Quick Start

### Install Prerequisites

1. Install [a boost version for Rose](install-boost).
2. Install [Rose](install-rose).
3. Install isl from repository. [optional but recomended]
4. Install [IEGenLib](install-iegenlib)

### Build CHiLL

CHill can be built from two build systems, CMake or automake. (CMake is recomended for use with CLion)

#### CMake

1. clone repository, for example: `git clone https://github.com/CtopCsUtahEdu/chill.git`
2. change directory into the cloned repository & make build directory: `mkdir build; cd build`
3. build with dependencies `cmake .. -DROSEHOME=<...> -DBOOSTHOME=<...> -DIEGENHOME=<...>` or `cmake .. -DCHILLENV=<...>`
4. build `make`

Note: you can also create the build directory somewhere else and substitute `..` with where your source is located.

#### Automake

1. clone repository, for example: `git clone https://echo12.cs.utah.edu/dhuth/chill-dev.git`
2. make a build directory somewhere outside of the cloned repository.
3. cd into the newly created build directory and run `.bootstrap`
4. run `./configure --with-rose=<...> --with-boost=<...> --with-iegen=<...>` (optionally specify `--enable-cuda=yes` to build cuda-chill intsead)
5. run make

Note that for both CMake and automake builds, all these extra variables may be specified in the environment so that they don't need to be specified each time. If something is wrong please following the error message, they usually provides a detailed report of what is missing or possibly misplaces.

### Running CHiLL (example)

CHiLL takes a single python script file as an argument, and the script file will reference a C source file.

For example, here is the script file fuse_distribute.script.py
```Python
# Basic illustration of loop fusion and distribution.

from chill import *

source('fuse_distribute.c')
destination('fuse_distributemodified.c')
procedure('foo')
loop(0)

# initially fused as much as possible
original()
print_code()

# distribute the first two statements
distribute([0,1], 2)
print_code()

# prepare the third statement for fusion
shift([2], 2, 1)
print_code()

# fuse the last two statements
fuse([1,2],2)
print_code()

```

And the source file fuse_distribute.c
```C

void foo(double A[100], double B[100]) {
    int i, j;
    for(i = 0; i < 100; i++) {
        for(j = 0; j < 100; j++) {
            A[j] = 1.0;
        }
        for(j = 0; j < 100; j++) {
            B[j] = 1.0;
        }
        for(j = 0; j < 99; j++) {
            B[j] = B[j+1]*A[j];
        }
    }
}
```

`chill fuse_distribute.script.py` will generate the destination source file fuse_distributemodified.c
```C
void foo(double A[100], double B[100]) {
  int t4;
  int t2;
  for (t2 = 0; t2 <= 99; t2 += 1) {
    for (t4 = 0; t4 <= 99; t4 += 1) 
      A[t4] = 1;
    B[0] = 1;
    for (t4 = 1; t4 <= 99; t4 += 1) {
      B[t4] = 1;
      B[t4 - 1] = B[t4 - 1 + 1] * A[t4 - 1];
    }
  }
}
```

### Examples and Testcases for CHiLL

Additional examples and testcases for CHiLL can be found under examples/chill/testcases. Testcases and exemples for CUDA-CHiLL can be found under exaples/cuda-chill/testcases.


