noinst_PROGRAMS=oc

OC_SRCDIR=$(srcdir)/../src


build_date = "\"`date +%m/%d/%Y`\""

BUILT_SOURCES=lex.yy.cc parser.tab.cc
AM_YFLAGS=-d -t
EXTRA_DIST=

calc_includes =-I$(OC_SRCDIR)
calc_includes+=-I$(srcdir)/../include
calc_includes+=-I$(srcdir)/../../omega_lib/include
calc_includes+=-I$(srcdir)/../../code_gen/include
calc_includes+=-I$(srcdir)/../../basic/include

#calc_libs =../../omega_lib/obj/libomega.a
#calc_libs+=../../code_gen/obj/libcodegen.a
calc_libs =-lomega -lcodegen
calc_lib_paths =-L../../omega_lib/obj -L../../code_gen/obj
calc_lib_paths+=-L$(ROSEHOME)/lib -L$(BOOSTHOME)/lib

oc_SOURCES=../src/AST.cc ../src/myflex.cc lex.yy.cc parser.tab.cc
oc_LDADD=$(calc_libs)
oc_LDFLAGS=$(calc_lib_paths)
oc_CPPFLAGS=$(calc_includes) -DBUILD_CODEGEN -DOMEGA_BUILD_DATE=$(build_date)
oc_CXXFLAGS=-std=c++11

EXTRA_DIST+=lex.yy.cc
$(OC_SRCDIR)/lex.yy.cc: parser.tab.cc $(OC_SRCDIR)/parser.ll
	flex++ -o lex.yy.cc $(OC_SRCDIR)/parser.ll

EXTRA_DIST+=parser.tab.cc
parser.tab.cc: $(OC_SRCDIR)/parser.yy
	bison -t -d -o parser.tab.cc $(OC_SRCDIR)/parser.yy

