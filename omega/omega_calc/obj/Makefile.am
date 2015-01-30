noinst_PROGRAMS=oc

SRCDIR=../src


build_date = "\"`date +%m/%d/%Y`\""

BUILT_SOURCES=$(SRCDIR)/lex.yy.cc $(SRCDIR)/parser.tab.cc
AM_YFLAGS=-d -t

calc_includes =-I$(SRCDIR)
calc_includes+=-I../include
calc_includes+=-I../../omega_lib/include -I../../code_gen/include
calc_includes+=-I../../basic/include

#calc_libs =../../omega_lib/obj/libomega.a
#calc_libs+=../../code_gen/obj/libcodegen.a
calc_libs =-lomega -lcodegen
calc_lib_paths =-L../../omega_lib/obj -L../../code_gen/obj
calc_lib_paths+=-L$(ROSEHOME)/lib -L$(BOOSTHOME)/lib

oc_SOURCES=$(SRCDIR)/AST.cc $(SRCDIR)/myflex.cc $(SRCDIR)/lex.yy.cc $(SRCDIR)/parser.tab.cc
oc_LDADD=$(calc_libs)
oc_LDFLAGS=$(calc_lib_paths)
oc_CPPFLAGS=$(calc_includes) -DBUILD_CODEGEN -DOMEGA_BUILD_DATE=$(build_date)

$(SRCDIR)/lex.yy.cc: $(SRCDIR)/parser.tab.cc $(SRCDIR)/parser.ll
	flex++ -o $(SRCDIR)/lex.yy.cc $(SRCDIR)/parser.ll

$(SRCDIR)/parser.tab.cc:
	bison -t -d -o $(SRCDIR)/parser.tab.cc $(SRCDIR)/parser.yy

