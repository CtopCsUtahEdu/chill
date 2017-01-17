#ifndef Already_Included_Affine_Closure
#define Already_Included_Affine_Closure

#include <omega/Relation.h>

/** @file
 *
 * This file describes the Farkas' Lemma, which let A be a real (m,n) matrix and b be an m-dimensional real vector,
 * EXACTLY one of the following two statements is true:
 *
 * * \f$\exists x \in R^n: Ax = b, x \geq 0\f$
 * * \f$\exists y \in R^m: y^TA \geq 0, y^Tb < 0\f$
 */

namespace omega {

enum Farkas_Type {Basic_Farkas, Decoupled_Farkas,
                  Linear_Combination_Farkas, Positive_Combination_Farkas,
                  Affine_Combination_Farkas, Convex_Combination_Farkas };

Relation Farkas(NOT_CONST Relation &R, Farkas_Type op, bool early_bailout = false);

extern coef_t farkasDifficulty;
extern Global_Var_ID coefficient_of_constant_term;

} // namespace

#endif
