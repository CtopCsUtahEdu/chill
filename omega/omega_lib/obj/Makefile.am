noinst_LIBRARIES   = libomega.a

basic_src   =../../basic/src/ConstString.cc
basic_src  +=../../basic/src/Link.cc

oc_src   =../src/omega_core/oc.cc
oc_src  +=../src/omega_core/oc_eq.cc
oc_src  +=../src/omega_core/oc_exp_kill.cc
oc_src  +=../src/omega_core/oc_global.cc
oc_src  +=../src/omega_core/oc_print.cc
oc_src  +=../src/omega_core/oc_problems.cc
oc_src  +=../src/omega_core/oc_simple.cc
oc_src  +=../src/omega_core/oc_solve.cc
oc_src  +=../src/omega_core/oc_query.cc
oc_src  +=../src/omega_core/oc_quick_kill.cc
oc_src  +=../src/omega_core/oc_util.cc

pres_src   =../src/pres_beaut.cc
pres_src  +=../src/pres_cnstr.cc
pres_src  +=../src/pres_col.cc
pres_src  +=../src/pres_conj.cc
pres_src  +=../src/pres_decl.cc
pres_src  +=../src/pres_dnf.cc
pres_src  +=../src/pres_form.cc
pres_src  +=../src/pres_gen.cc
pres_src  +=../src/pres_logic.cc
pres_src  +=../src/pres_print.cc
pres_src  +=../src/pres_rear.cc
pres_src  +=../src/pres_quant.cc
pres_src  +=../src/pres_subs.cc
pres_src  +=../src/pres_var.cc

rel_src   =../src/evac.cc
rel_src  +=../src/farkas.cc
rel_src  +=../src/hull_legacy.cc
rel_src  +=../src/hull_simple.cc
rel_src  +=../src/Relation.cc
rel_src  +=../src/Relations.cc
rel_src  +=../src/RelBody.cc
rel_src  +=../src/RelVar.cc

fancy_src   =../src/closure.cc
fancy_src  +=../src/reach.cc


libomega_a_SOURCES  =$(basic_src) $(oc_src) $(pres_src) $(rel_src) $(fancy_src)
libomega_a_CPPFLAGS =-I$(srcdir)/../include -I$(srcdir)/../../basic/include
libomega_a_CXXFLAGS =-std=c++11
