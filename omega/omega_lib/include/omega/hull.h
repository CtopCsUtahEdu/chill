#ifndef Already_Included_Hull
#define Already_Included_Hull

#include <omega/farkas.h>

/** @file */

namespace omega {

Relation SimpleHull(const Relation &R, bool allow_stride_constraint = false, bool allow_irregular_constraint = false);
Relation SimpleHull(const std::vector<Relation> &Rs, bool allow_stride_constraint = false, bool allow_irregular_constraint = false);


// All of the following first call approximate on R to
// eliminate any wildcards and strides.

/**
 * @brief Calculate the Convex Hull of R using the convex combination
 *
 * Tightest inequality constraints who's intersection contains all of S.
 * \f[x \in Convex(S) \iff \exists y_i \in S, a_i \geq 0, \sum_i a_i = 1: x = \sum_i a_iy_i\f]
 *
 * This will first call approximate on R to eliminate any wildcards and strides.
 *
 * Expensive
 */
Relation ConvexHull(NOT_CONST Relation &R);

/**
 * @brief Calculate the DecoupledConvex Hull of R
 *
 * DecoupledConvexHull is the same as ConvexHull,
 * except that it only finds constraints that involve
 * both variables x&y if there is a constraint
 * that involves both x&y in one of the conjuncts
 * of R. Always contains the convex hull.
 *
 * This will first call approximate on R to eliminate any wildcards and strides
 */
Relation DecoupledConvexHull(NOT_CONST Relation &R);

/**
 * @brief Calculate the Affine Hull using affine combination
 *
 * Tightest set of equality constraints who's intersection contains all of S.
 *
 * Affine combination is the convex combination without the positivity constraint a_i:
 * \f[x \in Affine(S) \iff \exists y_i in S, \exists a_i \in R, \sum_i a_i = 1: x = \sum_i a_iy_i\f]
 * This will first call approximate on R to eliminate any wildcards and strides
 * \sa ConvexHull
 *
 * Expensive
 */
Relation AffineHull(NOT_CONST Relation &R);

/**
 * @brief Calculate the Linear Hull using linear combination
 *
 * Linear combination is the convex combination without constraints on a_i:
 * \f[x \in Linear(S) \iff \exists y_i \in S, \exists a_i \in R: x = \sum_i a_iy_i\f]
 * This will first call approximate on R to eliminate any wildcards and strides
 * \sa ConvexHull
 *
 * Expensive
 */
Relation LinearHull(NOT_CONST Relation &R);

/**
 * @brief Calculate the Conic Hull using conic combination
 *
 * Conic hull is the tightest cone that contains S(R).
 *
 * Conic combination is the convex combination without the unit constraint on a_i:
 * \f[x \in Conic(R) \iff \exists y_i \in R, a_i \geq 0: x = \sum_i a_iy_i\f]
 * This will first call approximate on R to eliminate any wildcards and strides
 * \sa ConvexHull
 *
 * Expensive
 */
Relation ConicHull(NOT_CONST Relation &R);

/**
 * @brief Calculate the Rect Hull
 *
 * RectHull includes readily-available constraints from relation
 * that can be part of hull, plus rectangular bounds calculated
 * from input/output/set variables' range.
 *
 * This uses gist and value range to calculate a quick rectangular hull. It
 * intends to replace all hull calculations (QuickHull, BetterHull,
 * FastTightHull) beyond the method of ConvexHull (dual
 * representations). In the future, it will support max(...)-like
 * upper bound. So RectHull complements ConvexHull in two ways: first
 * for relations that ConvexHull gets too complicated, second for
 * relations where different conjuncts have different symbolic upper
 * bounds.
 */
Relation RectHull(NOT_CONST Relation &Rel);

/**
 * A constraint is in the result of QuickHull only if it appears in one of
 * the relations and is directly implied by a single constraint in each of
 * the other relations.
 */
__attribute__((deprecated)) Relation QuickHull(Relation &R);
__attribute__((deprecated)) Relation QuickHull(Tuple<Relation> &Rs);

/**
 * Will guess the computation complexity to decide whether to use a simpler
 * hull or a more precise one. The guess is not always correct and if trying
 * to tackle something that is too complex it will fail by generating a fatal
 * error.
 */
Relation FastTightHull(NOT_CONST Relation &input_R,
                        NOT_CONST Relation &input_H);
Relation  Hull(NOT_CONST Relation &R, 
			bool stridesAllowed = false,
			int effort=1,
			NOT_CONST Relation &knownHull = Relation::Null()
			);
Relation Hull(Tuple<Relation> &Rs, 
              const std::vector<bool> &validMask, 
              int effort = 1, 
              bool stridesAllowed = false,
              NOT_CONST Relation &knownHull = Relation::Null());

/**
 * @brief Simplify a union of sets/relations to a minimal (may not be optimal) number of convex regions.
 *
 * If a union of several conjuncts is a convex, their union
 * representaition can be simplified by their convex hull.
 */
Relation ConvexRepresentation(NOT_CONST Relation &R);
__attribute__((deprecated)) Relation CheckForConvexPairs(NOT_CONST Relation &S);
__attribute__((deprecated)) Relation CheckForConvexRepresentation(NOT_CONST Relation &R_In);

}

#endif
