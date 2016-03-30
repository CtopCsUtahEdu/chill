
noinst_LIBRARIES =libcodegen.a

libcodegen_a_SOURCES    =../src/codegen.cc
libcodegen_a_SOURCES   +=../src/CG_chillBuilder.cc
libcodegen_a_SOURCES   +=../src/CG_chillRepr.cc
libcodegen_a_SOURCES   +=../src/CG_roseRepr.cc
libcodegen_a_SOURCES   +=../src/CG_stringBuilder.cc
libcodegen_a_SOURCES   +=../src/CG.cc
libcodegen_a_SOURCES   +=../src/CG_utils.cc
libcodegen_a_CFLAGS     =-DFRONTEND_ROSE -Wno-write-strings
libcodegen_a_CPPFLAGS   =-DFRONTEND_ROSE -I../../.. -I../include -I../../omega_lib/include -I$(ROSEHOME)/include/rose -I$(BOOSTHOME)/include
#if BUILD_ROSE
libcodegen_a_SOURCES   += ../src/rose_attributes.cc ../src/CG_roseRepr.cc ../src/CG_roseBuilder.cc
#endif

