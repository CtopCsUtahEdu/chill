#ifndef Already_Included_Affine_Closure
#define Already_Included_Affine_Closure

#include <omega/Relation.h>

namespace omega {

enum Farkas_Type {Basic_Farkas, Decoupled_Farkas,
                  Linear_Combination_Farkas, Positive_Combination_Farkas,
                  Affine_Combination_Farkas, Convex_Combination_Farkas };

Relation Farkas(NOT_CONST Relation &R, Farkas_Type op, bool early_bailout = false);

extern coef_t farkasDifficulty;
extern Global_Var_ID coefficient_of_constant_term;

} // namespace

#endif
