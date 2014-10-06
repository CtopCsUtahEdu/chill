#if !defined(Already_included_debugging)
#define Already_included_debugging

// Debugging flags.  Can set any of these.

#include <stdio.h>
#include <ctype.h>

namespace omega {



extern int omega_core_debug;
extern int pres_debug;
extern int relation_debug;
extern int closure_presburger_debug;
extern int hull_debug;
extern int farkas_debug;
extern int code_gen_debug;

enum negation_control { any_negation, one_geq_or_eq, one_geq_or_stride };
extern negation_control pres_legal_negations;

#if defined STUDY_EVACUATIONS
extern int evac_debug;
#endif

} // namespace

#endif
