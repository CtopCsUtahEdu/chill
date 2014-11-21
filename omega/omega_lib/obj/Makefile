# Makefile for release 1.0 of the omega system
# DON'T EDIT -- put any locally required changes in Makefile.config.
# See INSTALL for more details.

LIB_TARGET = libomega.a

all: $(LIB_TARGET)
depend: depend_self
clean: clean_self
veryclean: veryclean_self



BASIC_SRC = $(BASEDIR)/basic/src/ConstString.cc $(BASEDIR)/basic/src/Link.cc
BASIC_OBJ = ConstString.o Link.o

OC_SRC = ../src/omega_core/oc.cc ../src/omega_core/oc_solve.cc \
         ../src/omega_core/oc_simple.cc ../src/omega_core/oc_eq.cc \
         ../src/omega_core/oc_problems.cc ../src/omega_core/oc_print.cc \
         ../src/omega_core/oc_quick_kill.cc ../src/omega_core/oc_exp_kill.cc \
         ../src/omega_core/oc_query.cc ../src/omega_core/oc_util.cc \
         ../src/omega_core/oc_global.cc
OC_OBJ = $(addsuffix .o, $(basename $(notdir $(OC_SRC))))

PRES_SRC = ../src/pres_print.cc ../src/pres_rear.cc ../src/pres_beaut.cc \
           ../src/pres_dnf.cc ../src/pres_conj.cc ../src/pres_quant.cc \
           ../src/pres_decl.cc ../src/pres_logic.cc ../src/pres_form.cc \
           ../src/pres_cnstr.cc ../src/pres_var.cc ../src/pres_gen.cc \
           ../src/pres_col.cc ../src/pres_subs.cc
PRES_OBJ = $(addsuffix .o, $(basename $(notdir $(PRES_SRC))))

REL_SRC = ../src/Relations.cc ../src/Relation.cc ../src/RelBody.cc ../src/RelVar.cc \
          ../src/evac.cc ../src/farkas.cc ../src/hull_legacy.cc ../src/hull_simple.cc
REL_OBJ = $(addsuffix .o, $(basename $(notdir $(REL_SRC))))

FANCY_SRC = ../src/reach.cc ../src/closure.cc
FANCY_OBJ = $(addsuffix .o, $(basename $(notdir $(FANCY_SRC))))


LIB_SRC = ${FANCY_SRC} ${REL_SRC} ${PRES_SRC} ${OC_SRC} ${BASIC_SRC}
LIB_OBJ = ${OC_OBJ} ${BASIC_OBJ} ${FANCY_OBJ} ${REL_OBJ} ${PRES_OBJ}

OBJS=$(LIB_OBJ)
SRCS=$(LIB_SRC)
OBJS=$(LIB_OBJ)

BASEDIR=../..
include $(BASEDIR)/Makefile.config
include $(BASEDIR)/Makefile.rules

#Extra rule for nonstandard source layout
%.o: ../src/omega_core/%.cc
	$(CC) $(CFLAGS) $(INCL_PATH) -c $< -o $@

ifeq ($(shell test -f Makefile.deps && echo "true"), true)
include Makefile.deps
endif
