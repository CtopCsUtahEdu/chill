
noinst_LIBRARIES =libcodegen.a

codegen_includes =-I$(srcdir)/../../..
codegen_includes+=-I$(srcdir)/../include
codegen_includes+=-I$(srcdir)/../../../include/chill
codegen_includes+=-I$(srcdir)/../../omega_lib/include
codegen_includes+=-I$(srcdir)/../../basic/include
codegen_includes+=-I$(ROSEHOME)/include/rose
codegen_includes+=-I$(BOOSTHOME)/include

libcodegen_a_SOURCES    =../src/codegen.cc
libcodegen_a_SOURCES   +=../src/CG_chillBuilder.cc
libcodegen_a_SOURCES   +=../src/CG_chillRepr.cc
libcodegen_a_SOURCES   +=../src/CG_roseRepr.cc
libcodegen_a_SOURCES   +=../src/CG_stringBuilder.cc
libcodegen_a_SOURCES   +=../src/CG.cc
libcodegen_a_SOURCES   +=../src/CG_utils.cc
libcodegen_a_CFLAGS     =-DFRONTEND_ROSE -Wno-write-strings
libcodegen_a_CXXFLAGS   =-std=c++11
libcodegen_a_CPPFLAGS   =-DFRONTEND_ROSE $(codegen_includes)
#if BUILD_ROSE
libcodegen_a_SOURCES   += ../src/rose_attributes.cc ../src/CG_roseRepr.cc ../src/CG_roseBuilder.cc
#endif

