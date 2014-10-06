#ifndef _CG_UTILS_H
#define _CG_UTILS_H

#include <omega.h>
#include <code_gen/CG_outputBuilder.h>
#include <basic/boolset.h>
#include <vector>
#include <set>
#include <map>

namespace omega {

class CG_loop;

CG_outputRepr *output_inequality_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, std::set<Variable_ID> excluded_floor_vars = std::set<Variable_ID>());
CG_outputRepr *output_substitution_repr(CG_outputBuilder *ocg, const EQ_Handle &equality, Variable_ID v, bool apply_v_coef, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
CG_outputRepr *output_upper_bound_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
CG_outputRepr *output_lower_bound_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const EQ_Handle &stride_eq, Variable_ID wc, const Relation &R, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);

CG_outputRepr *output_ident(CG_outputBuilder *ocg, const Relation &R, Variable_ID v, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> > output_assignment(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
CG_outputRepr *output_loop(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
CG_outputRepr *output_guard(CG_outputBuilder *ocg, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
std::vector<CG_outputRepr *> output_substitutions(CG_outputBuilder *ocg, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);

bool bound_must_hit_stride(const GEQ_Handle &inequality, Variable_ID v, const EQ_Handle &stride_eq, Variable_ID wc, const Relation &bounds, const Relation &known);
std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R, Variable_ID v, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly = std::vector<std::pair<CG_outputRepr *, int> >());
std::pair<bool, GEQ_Handle> find_floor_definition(const Relation &R, Variable_ID v, std::set<Variable_ID> excluded_floor_vars = std::set<Variable_ID>());
std::pair<EQ_Handle, Variable_ID> find_simplest_stride(const Relation &R, Variable_ID v);
Variable_ID replicate_floor_definition(const Relation &R, const Variable_ID floor_var, Relation &r, F_Exists *f_exists, F_And *f_root, std::map<Variable_ID, Variable_ID> &exists_mapping);

Relation pick_one_guard(const Relation &R, int level = 0);
CG_outputRepr *leaf_print_repr(BoolSet<> active, const std::map<int, Relation> &guards,
                               CG_outputRepr *guard_repr, const Relation &known,
                               int indent, CG_outputBuilder *ocg, const std::vector<int> &remap,
                               const std::vector<Relation> &xforms, const std::vector<CG_outputRepr *> &stmts,
                               const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);
CG_outputRepr *loop_print_repr(const std::vector<CG_loop *> &loops, int start, int end,
                               const Relation &guard, CG_outputRepr *guard_repr,
                               int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts,
                               const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly);

}

#endif
