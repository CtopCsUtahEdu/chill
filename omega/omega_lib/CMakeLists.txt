set(OMEGAROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(OC_SRC 
    src/omega_core/oc.cc
    src/omega_core/oc_eq.cc
    src/omega_core/oc_exp_kill.cc
    src/omega_core/oc_global.cc
    src/omega_core/oc_print.cc
    src/omega_core/oc_problems.cc
    src/omega_core/oc_simple.cc
    src/omega_core/oc_solve.cc
    src/omega_core/oc_query.cc
    src/omega_core/oc_quick_kill.cc
    src/omega_core/oc_util.cc
    )

set(PRES_SRC
    src/pres_beaut.cc
    src/pres_cnstr.cc
    src/pres_col.cc
    src/pres_conj.cc
    src/pres_decl.cc
    src/pres_dnf.cc
    src/pres_form.cc
    src/pres_gen.cc
    src/pres_logic.cc
    src/pres_print.cc
    src/pres_rear.cc
    src/pres_quant.cc
    src/pres_subs.cc
    src/pres_var.cc
    )

set(REL_SRC
    src/evac.cc
    src/farkas.cc
    src/hull_legacy.cc
    src/hull_simple.cc
    src/Relation.cc
    src/Relations.cc
    src/RelBody.cc
    src/RelVar.cc
    )

set(FANCY_SRC
    src/closure.cc
    src/reach.cc
    )

include_directories(
    ${OMEGA_INC}
    )

add_library(omega_lib
    ${OC_SRC}
    ${PRES_SRC}
    ${REL_SRC}
    ${FANCY_SRC}
    )

install(TARGETS omega_lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY include
    DESTINATION .)
