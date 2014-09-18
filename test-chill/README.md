# testchill

## Description  
TODO: better description  
testchill is a Python module that runs a series of tests to aid in the development and maintence of CHiLL.
testchill tests that chill compiles successfully, that scripts can be run without error, and that they generate compilable code.
It can also optionally test optimized code for correctness and provide code coverage.  


## Running testchill  

testchill is a Python module, and can be run like any other Python module:  
`python -m testchill <suite-args>* <sub-command> <sub-command-args>*`  

The most basic subcommands run the testsuite are [`local`](#-local-chill-home-) and [`repo`](#-repo-svn-username). `local` runs a set of tests on a local chill source directory, and `repo` will grab the latest version of both omega and chill and run the same set of tests.  

`python -m testchill [-O <path-to-omega>] local <path-to-chill>` If the environment variable $OMEGAHOME is set, the `-O` argument can be ommited.  
`python -m testchill repo <svn-user-name>`  

### Arguments common to all sub commands (with the exception of `repo` and `local`):  
- `-w <working-directory>, --working-dir <working-directory>`

   Sets the working directory where testchill will compile and run test scripts. If not set, the current working  directory will be used.

- `-R <rose-home>, --rose-home <rose-home>`

   Set ROSEHOME environment variable for building omega. If not set, the current ROSEHOME environment variable will be used.

- `-C <chill directory>, --chill-home <chill-home>`

   Set the path to chill. If not set, the current CHILLHOME environment variable will be used.

- `-O <omega directory>, --omega-home <omega-home>`

   Set the path to omega. If not set, the current OMEGAHOME environment variable will be used.

- `-b <binary directory>, --binary-dir <binary directory>`

   Set the directory were all chill binary files will be placed after being compiled. The chill directory will be used by default.

### Subcommands for running individual tests:  
- <h4> `build-chill-testcase ...`

   Build chill. It will fail if the build process returns non zero.  
   Optional arguments:  
   - `-v {release | dev}` or `--chill-branch {release | dev}`
   
     `release` will build the old release version, and `dev` will build the current development version.  
     `dev` is used by default.
   
   - `-u | -c` or `--target-cuda | --target-c`
   
     `-c` will build chill, and `-u` will build cuda-chill.  
     `-c` is used by default.
   
   - `-i {script | lua | python}` or `--interface-lang {script | lua | python}`
   
     Set the interface language chill will be build for.  
     `script` will build chill with the original chill script language.  
     `lua` will build chill with lua as the interface language.  
     `python` will build chill with python as the interface language.  
     By default, `script` is used for chill and `lua` is used for cuda-chill.  
   
   - `--build-coverage | --no-build-coverage`
   
     `--build-coverage` will build chill to work with gcov.  
     `--no-build-coverage` will build chill normally.  
     It is on by default.  
   
- <h4> `chill-testcase <chill-script> <chill-src> ...`

   Run a chill test script.  
   Arguments:  
   - `chill-script`
     
     Path to the script file.  
     
   - `chill-src`
     
     Path to the source file.  
     
   Optional arguments:
   - `-v {release | dev}` or `--chill-branch {release | dev}`
   
     `release` will run scripts as the old release version, and `dev` will run them  as the current development version.  
     `dev` is used by default.
   
   - `-u | -c` or `--target-cuda | --target-c`
   
     `-c` will run chill, and `-u` will run cuda-chill.  
     `-c` is used by default.
   
   - `-i {script | lua | python}` or `--interface-lang {script | lua | python}`
   
     Set the interface language chill will be run with.  
     `script` will run chill with the original chill script language.  
     `lua` will run chill with lua as the interface language.  
     `python` will run chill with python as the interface language.  
     By default, `script` is used for chill and `lua` is used for cuda-chill.  
     
   - `--compile-src | --no-compile-src`
     
     Turns source compilation test on or off. If on, the source file will be compiled prior to transormation.  
     On by default.  
     
   - `--run-script | --no-run-script`
     
     If on, the script file will be run.  
     On by default.  
     
   - `--compile-gensrc | --no-compile-gensrc`
     
     If on, the generated source file will be compiled.  
     On by default.  
     
   - `--check-run-script | --no-check-run-script`
     
     If on, the generated object file will be run. If there are any validation tests, each one will be compiled and run.  
     On by default.  
     
   - `--test-coverage | --no-test-coverage`
     
     If on, coverage data will be compiled during the run-script test.  
     On by default.  
   
- <h4> `batch <batch-file>`
   
   Run a test case list (*.tclist) file. Each line constists of a subcommand to be passed to testchill (including additional `batch` commands).  
   Arguments:
   - `<batch-file>`
     
     Path to a test case list file.
   
- <h4> `local <chill-home> ...`
  
  Compile and test a local chill source directory.  
  Arguments:
  - `<chill-home>`
    
    Path to chill.  
  
  Optional arguments:  
  - `-v {release | dev}` or `--chill-branch {release | dev}`
   
     `release` will run scripts as the old release version, and `dev` will run them  as the current development version.  
     `dev` is used by default.  
  
- <h4> `repo <svn-username>`
  
  Checkout the latest version of omega and chill, compile both and test chill.  
  Arguments:  
  - `<svn-username>`
    
    Svn username.  
  


