
noinst_LIBRARIES =libcodegen.a

libcodegen_a_SOURCES    =../src/codegen.cc ../src/CG_stringBuilder.cc ../src/CG.cc ../src/CG_utils.cc
libcodegen_a_CFLAGS     =-Wno-write-strings
libcodegen_a_CPPFLAGS   =-I../include -I../../omega_lib/include -I$(ROSEHOME)/include -I$(BOOSTHOME)/include
#if BUILD_ROSE
libcodegen_a_SOURCES   += ../src/rose_attributes.cc ../src/CG_roseRepr.cc ../src/CG_roseBuilder.cc
#endif

