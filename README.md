# CHiLL - The Composable High-Level Loop Source-to-Source Translator

CHiLL is a source-to-source translator for composing high level loop transformations to improve the performance of nested loop calculations written in C. CHiLL’s operations are driven by a script which is generated or supplied by the user that specifies the location of the original source file, the function and loops to modify and the transformations to apply.

## Quick Start

### Install Prerequisites

1. Install a boost version for Rose.
2. Install Rose.
3. Install isl from repository. [optional but recomended]
4. Install IEGenLib

### Build CHiLL

Chill can be built from two build systems, CMake or automake. (CMake is recomended for use with CLion)

#### CMake

1. clone repository, for example: `git clone https://echo12.cs.utah.edu/dhuth/chill-dev.git`
2. change directory into the cloned repository & make build directory: `mkdir build; cd build`
3. build with dependencies `cmake .. -DROSEHOME=<...> -DBOOSTHOME=<...> -DIEGENHOME=<...>` or `cmake .. -DCHILLENV=<>`
4. build `make`

Note: you can also create the build directory somewhere else and substitute `..` with where your source is located.

#### Automake

1. clone repository, for example: `git clone https://echo12.cs.utah.edu/dhuth/chill-dev.git`
2. make a build directory somewhere outside of the cloned repository.
3. cd into the newly created build directory and run /path/to/the/cloned/repository/bootstrap
4. run `./configure --with-rose=<...> --with-boost=<...> --with-iegen=<...>` (optionally specify `--enable-cuda=yes` to build cuda-chill intsead)
5. run make

Note that for both CMake and automake builds, all these extra variables may be specified in the environment so that they don't need to be specified each time. If something is wrong please following the error message, they usually provides a detailed report of what is missing or possibly misplaces.

### Running CHiLL

`chill SCRIPT_FILE_NAME`

### Examples and Testcases for CHiLL

Testcases and examples for CHiLL can be found under examples/chill/testcases. Testcases and exemples for Cuda-CHiLL can be found under exaples/cuda-chill/testcases.


