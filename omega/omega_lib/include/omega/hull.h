#ifndef Already_Included_Hull
#define Already_Included_Hull

#include <omega/farkas.h>

namespace omega {

Relation SimpleHull(const Relation &R, bool allow_stride_constraint = false, bool allow_irregular_constraint = false);
Relation SimpleHull(const std::vector<Relation> &Rs, bool allow_stride_constraint = false, bool allow_irregular_constraint = false);


// All of the following first call approximate on R to
// eliminate any wildcards and strides.

// x in Convex Hull of R
// iff
// exist a_i, y_i s.t. 
//    x = Sum_i  a_i y_i s.t.
//    forall i, y_i in R
//    forall i, a_i >=0
//    sum_i  a_i = 1
Relation ConvexHull(NOT_CONST Relation &R);

// DecoupledConvexHull is the same as ConvexHull, 
// except that it only finds constraints that involve
// both variables x&y if there is a constraint 
// that involves both x&y in one of the conjuncts 
// of R.
Relation DecoupledConvexHull(NOT_CONST Relation &R);

// The affine hull just consists of equality constraints
// but is otherwise the tightest hull on R.
// x in Affine Hull of R
// iff
// exist a_i, y_i s.t. 
//    x = Sum_i  a_i y_i s.t.
//    forall i, y_i in R
//    sum_i  a_i = 1
Relation AffineHull(NOT_CONST Relation &R);

// x in Linear Hull of R
// iff
// exist a_i, y_i s.t. 
//    x = Sum_i  a_i y_i s.t.
//    forall i, y_i in R
Relation LinearHull(NOT_CONST Relation &R);

// The conic hull is the tighest cone that contains R
// x in Conic Hull of R.
// iff
// exist a_i, y_i s.t. 
//    x = Sum_i  a_i y_i s.t.
//    forall i, y_i in R
//    forall i, a_i >=0
Relation ConicHull(NOT_CONST Relation &R);

// RectHull includes readily-available constraints from relation
// that can be part of hull, plus rectangular bounds calculated
// from input/output/set variables' range.
Relation RectHull(NOT_CONST Relation &Rel);

// A constraint is in the result of QuickHull only if it appears in one of
// the relations and is directly implied by a single constraint in each of
// the other relations.
Relation QuickHull(Relation &R); // deprecated
Relation QuickHull(Tuple<Relation> &Rs); // deprecated

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

// If a union of several conjuncts is a convex, their union
// representaition can be simplified by their convex hull.
Relation ConvexRepresentation(NOT_CONST Relation &R);
Relation CheckForConvexPairs(NOT_CONST Relation &S); // deprecated
Relation CheckForConvexRepresentation(NOT_CONST Relation &R_In); // deprecated

}

#endif
