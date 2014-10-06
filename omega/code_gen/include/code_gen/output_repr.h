#ifndef OUTPUT_REPR_H
#define OUTPUT_REPR_H

#include <omega.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_outputRepr.h>
#include <vector>
#include <set>

namespace omega {
extern int last_level;

CG_outputRepr* outputIdent(CG_outputBuilder* ocg, const Relation &R, Variable_ID v, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
std::pair<CG_outputRepr *, bool> outputAssignment(CG_outputBuilder *ocg, const Relation &R_, Variable_ID v, Relation &enforced, CG_outputRepr *&if_repr, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
std::pair<CG_outputRepr *, bool> outputBounds(CG_outputBuilder* ocg, const Relation &bounds, Variable_ID v, int indent, Relation &enforced, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
Tuple<CG_outputRepr*> outputSubstitution(CG_outputBuilder* ocg, const Relation &R, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr* outputStatement(CG_outputBuilder* ocg, CG_outputRepr *stmt, int indent,  const Relation &mapping, const Relation &known, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr* outputGuard(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr* output_as_guard(CG_outputBuilder* ocg, const Relation &guards_in, Constraint_Handle e, bool is_equality, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr* output_EQ_strides(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr* output_GEQ_strides(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr *outputLBasRepr(CG_outputBuilder* ocg, const GEQ_Handle &g, Relation &bounds, Variable_ID v, coef_t stride, const EQ_Handle &strideEQ, Relation known, const std::vector<CG_outputRepr *> &assigned_on_the_fly);
CG_outputRepr *outputUBasRepr(CG_outputBuilder* ocg, const GEQ_Handle &g,  Relation & bounds, Variable_ID v,
                              coef_t /*stride*/, // currently unused
                              const EQ_Handle &/*strideEQ*/,
                              const std::vector<CG_outputRepr *> &assigned_on_the_fly = std::vector<CG_outputRepr *>(last_level, static_cast<CG_outputRepr *>(NULL)));
CG_outputRepr* outputEasyBoundAsRepr(CG_outputBuilder* ocg, Relation &bounds, const Constraint_Handle &g, Variable_ID v, bool ignoreWC, int ceiling, const std::vector<CG_outputRepr *> &assigned_on_the_fly);


bool boundHitsStride(const GEQ_Handle &g, Variable_ID v, const EQ_Handle &strideEQ, coef_t /*stride, currently unused*/, Relation known);
Relation greatest_common_step(const Tuple<Relation> &I, const Tuple<int> &active, int level, const Relation &known = Relation::Null());
bool findFloorInequality(Relation &r, Variable_ID v, GEQ_Handle &h, Variable_ID excluded);
Relation project_onto_levels(Relation R, int last_level, bool wildcards);
bool isSimpleStride(const EQ_Handle &g, Variable_ID v);
int countStrides(Conjunct *c, Variable_ID v, EQ_Handle &strideEQ, bool &simple);
bool hasBound(Relation r, int level, int UB);
bool find_any_constraint(int s, int level, Relation &kr, int direction, Relation &S, bool approx);
bool has_nonstride_EQ(Relation r, int level);
Relation pickOverhead(Relation r, int liftTo);
Relation minMaxOverhead(Relation r, int level);
int max_fs_arity(const Constraint_Handle &c);


}

#endif
