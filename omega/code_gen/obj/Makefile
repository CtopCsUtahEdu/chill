
# DON'T EDIT -- put any locally required changes in Makefile.config.


LIB_TARGET=libcodegen.a

all: $(LIB_TARGET)
depend: depend_self
clean: clean_self
veryclean: veryclean_self

BASIC_SRC = ../src/codegen.cc ../src/CG_stringBuilder.cc ../src/CG.cc ../src/CG_utils.cc 
BASIC_OBJ = $(BASIC_SRC:../src/%.cc=%.o)


ROSE_SRC = ../src/rose_attributes.cc ../src/CG_roseRepr.cc ../src/CG_roseBuilder.cc
ROSE_OBJ = $(ROSE_SRC:../src/%.cc=%.o)

OBJS = ${BASIC_OBJ}
SRCS = ${BASIC_SRC}

BASEDIR=../..
include $(BASEDIR)/Makefile.config

INCL_PATH := $(INCL_PATH) -I../../omega_lib/include




ifeq ($(BUILD_ROSE), true)
OBJS := $(OBJS) $(ROSE_OBJ)
SRCS := $(SRCS) $(ROSE_SRC)
INCL_PATH := $(INCL_PATH) -I${ROSEHOME}/include -I${BOOSTHOME}/include
endif

ifeq ($(BUILD_ROSE), true)
IR_OBJ = $(ROSE_SRC:../src/%.cc=%.o)
endif

include $(BASEDIR)/Makefile.rules


# suppress warning of constant char * conversion nuiance in suif
$(IR_OBJ): %.o: ../src/%.cc
	$(CC) -Wno-write-strings $(CFLAGS) $(INCL_PATH) -c $< -o $@

ifeq ($(shell test -f Makefile.deps && echo "true"), true)
include Makefile.deps
endif
