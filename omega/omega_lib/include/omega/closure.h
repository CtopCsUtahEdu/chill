#if ! defined _closure_h
#define _closure_h

#include <omega/Relation.h>

namespace omega {

Relation VennDiagramForm(
                Tuple<Relation> &Rs,
                NOT_CONST Relation &Context_In);
Relation VennDiagramForm(
                NOT_CONST Relation &R_In,
                NOT_CONST Relation &Context_In = Relation::Null());

// Given a Relation R, returns a relation deltas
// that correspond to the ConicHull of the detlas of R
Relation ConicClosure (NOT_CONST Relation &R);

Relation  TransitiveClosure (NOT_CONST Relation &r, 
                             int maxExpansion = 1,
                             NOT_CONST Relation &IterationSpace=Relation::Null());

/* Tomasz Klimek */
Relation calculateTransitiveClosure(NOT_CONST Relation &r);

/* Tomasz Klimek */
Relation ApproxClosure(NOT_CONST Relation &r);

} // namespace
 
#endif
