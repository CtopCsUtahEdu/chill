# Makefile for release 1.0 of the omega system
# DON'T EDIT -- put any locally required changes in Makefile.config.
# See INSTALL for more details.

EXEC_TARGET=oc


all: $(EXEC_TARGET)

clean: clean_self

veryclean: veryclean_self


SRCS := ../src/AST.cc ../src/myflex.cc
OBJS := $(SRCS:../src/%.cc=%.o)

YACC_SRCS := lex.yy.cc parser.tab.cc
YACC_OBJS := $(YACC_SRCS:.cc=.o)

depend: $(YACC_SRCS) depend_self

SRCS := $(SRCS) $(YACC_SRCS)
OBJS := $(OBJS) $(YACC_OBJS)

BASEDIR=../..
include $(BASEDIR)/Makefile.config

LIB_PATH := $(LIB_PATH) -L../../omega_lib/obj
LIBS := $(LIBS) -lomega

ifeq ($(BUILD_CODEGEN), true)
LIB_PATH := $(LIB_PATH) -L../../code_gen/obj
LIBS := $(LIBS) -lcodegen
CFLAGS := $(CFLAGS) -DBUILD_CODEGEN
endif

CFLAGS := $(CFLAGS)
INCL_PATH := -I. $(INCL_PATH) -I../../omega_lib/include -I../../code_gen/include

include $(BASEDIR)/Makefile.rules

# lex/yacc related
.PHONY: parser.tab.o

lex.yy.cc: ../src/parser.ll
	flex++ $<

parser.tab.hh parser.tab.cc: ../src/parser.yy
	bison -t -d $<

parser.tab.o: parser.tab.cc
	$(CC) $(CFLAGS) -DOMEGA_BUILD_DATE="\"$(OMEGA_BUILD_DATE)\"" $(INCL_PATH) -c $<

ifeq ($(shell test -f Makefile.deps && echo "true"), true)
include Makefile.deps
endif
