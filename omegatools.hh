#ifndef OMEGATOOLS_HH
#define OMEGATOOLS_HH

#include <string>
#include <omega.h>
#include "dep.hh"
#include "ir_code.hh"
#include "loop.hh"

std::string tmp_e();

/**
 * @brief Convert expression tree to omega relation.
 * @param destroy shallow deallocation of "repr", not freeing the actual code inside.
 */
void exp2formula(Loop *loop, IR_Code *ir, omega::Relation &r, omega::F_And *f_root,
                 std::vector<omega::Free_Var_Decl*> &freevars, omega::CG_outputRepr *repr,
                 omega::Variable_ID lhs, char side, IR_CONDITION_TYPE rel, bool destroy,
                 std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                 std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                 std::map<std::string, std::vector<omega::Relation> > &index_variables, bool extractingDepRel=false);

omega::Relation arrays2relation(Loop *loop, IR_Code *ir,
                         std::vector<omega::Free_Var_Decl*> &freevars, const IR_ArrayRef *ref_src,
                         const omega::Relation &IS_w, const IR_ArrayRef *ref_dst, const omega::Relation &IS_r,
                         std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                         std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                         std::map<std::string, std::vector<omega::Relation> > &unin_rel);

std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > relation2dependences(
                                                                                              const IR_ArrayRef *ref_src, const IR_ArrayRef *ref_dst, const omega::Relation &r);

/**
 * @brief Convert a boolean expression to omega relation.
 * "destroy" means shallow deallocation of "repr", not freeing the actual code inside.
 */
void exp2constraint(Loop *loop, IR_Code *ir, omega::Relation &r, omega::F_And *f_root,
                    std::vector<omega::Free_Var_Decl *> &freevars, omega::CG_outputRepr *repr,
                    bool destroy,
                    std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                    std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                    std::map<std::string, std::vector<omega::Relation> > &index_variables);

bool is_single_iteration(const omega::Relation &r, int dim);
void assign_const(omega::Relation &r, int dim, int val);
int get_const(const omega::Relation &r, int dim, omega::Var_Kind type);
omega::Variable_ID find_index(omega::Relation &r, const std::string &s, char side);
omega::Relation permute_relation(const std::vector<int> &pi);
omega::Relation get_loop_bound(const omega::Relation &r, int dim);
bool is_single_loop_iteration(const omega::Relation &r, int level, const omega::Relation &known);
omega::Relation get_loop_bound(const omega::Relation &r, int level, const omega::Relation &known);
omega::Relation get_max_loop_bound(const std::vector<omega::Relation> &r, int dim);
omega::Relation get_min_loop_bound(const std::vector<omega::Relation> &r, int dim);
void add_loop_stride(omega::Relation &r, const omega::Relation &bound, int dim, int stride);
bool is_inner_loop_depend_on_level(const omega::Relation &r, int level, const omega::Relation &known);
// void adjust_loop_bound(omega::Relation &r, int dim, int adjustment, std::vector<omega::Free_Var_Decl *> globals = std::vector<omega::Free_Var_Decl *>());
omega::Relation adjust_loop_bound(const omega::Relation &r, int level, int adjustment);
bool lowerBoundIsZero(const omega::Relation &bound, int dim);
omega::Relation and_with_relation_and_replace_var(const omega::Relation &R, omega::Variable_ID v1,
                                                  omega::Relation &g);
omega::Relation replicate_IS_and_add_at_pos(const omega::Relation &R, int level, omega::Relation &bound);
omega::Relation replicate_IS_and_add_bound(const omega::Relation &R, int level, omega::Relation &bound) ;

omega::CG_outputRepr * construct_int_floor(omega::CG_outputBuilder * ocg, const omega::Relation &R, const omega::GEQ_Handle &h, omega::Variable_ID v,const std::vector<std::pair<omega::CG_outputRepr *, int> > &assigned_on_the_fly,
                                           std::map<std::string, std::vector<omega::CG_outputRepr *> > unin);
//omega::CG_outputRepr * modified_output_subs_repr(omega::CG_outputBuilder * ocg, const omega::Relation &R, const omega::EQ_Handle &h, omega::Variable_ID v,const std::vector<std::pair<omega::CG_outputRepr *, int> > &assigned_on_the_fly,
//    std::map<std::string, std::vector<omega::CG_outputRepr *> > unin);
std::pair<omega::Relation, bool> replace_set_var_as_existential(const omega::Relation &R,int pos, std::vector<omega::Relation> &bound);
omega::Relation replace_set_var_as_Global(const omega::Relation &R,int pos,std::vector<omega::Relation> &bound);
//! Return names of global vars with arity 0
std::set<std::string> get_global_vars(const omega::Relation &r);


// Mahdi: Adding this specifically for dependence extraction
//        This function basically renames tuple variables of old_relation
//        using tuple declaration of new_relation, and updates all constraints in the old_relation
//        and returns the updated relation. It can be used for doing something like following:
//        INPUTS:
//          old_relation: {[i,j]: col(j)=i && 0 < i < n}
//          new_relation: {[ip,jp]}
//        OUTPUT:
//                      : {[ip,jp]: col(jp)=ip && 0 < ip < n}
// NOte: replace_set_var_as_another_set_var functions are suppose to do the same thing
//       but 1 tuple variable at a time, which is not efficient for dependence extractiion.
//       Also, those functions are not handling all kinds of variables.
omega::Relation replace_set_vars(const omega::Relation &new_relation, const omega::Relation &old_relation);

//! Replicates old_relation's bounds for set var at old_pos into new_relation at new_pos
/**
 * position's bounds must involve constants, only supports GEQs
 */
omega::Relation replace_set_var_as_another_set_var(const omega::Relation &old_relation, const omega::Relation &new_relation, int old_pos, int new_pos);
// TODO Merge Anand's
omega::Relation replace_set_var_as_another_set_var(const omega::Relation &new_relation, const omega::Relation &old_relation, int new_pos, int old_pos, std::map<int, int> &pos_mapping);
omega::Relation extract_upper_bound(const omega::Relation &R, omega::Variable_ID v1);

// void adjust_loop_bound(Relation &r, int dim, int adjustment);
// void adjust_loop_bound(Relation &r, int dim, Free_Var_Decl *global_var, int adjustment);
// boolean is_private_statement(const omega::Relation &r, int dim);

// coef_t mod(const Relation &r, Variable_ID v, int dividend);


enum LexicalOrderType {LEX_MATCH, LEX_BEFORE, LEX_AFTER, LEX_UNKNOWN};

// template <typename T>
// LexicalOrderType lexical_order(const std::vector<T> &a, const std::vector<T> &b) {
//   int size = min(a.size(), b.size());
//   for (int i = 0; i < size; i++) {
//     if (a[i] < b[i])
//       return LEX_BEFORE;
//     else if (b[i] < a[i])
//       return LEX_AFTER;
//   }
//   if (a.size() < b.size())
//     return LEX_BEFORE;
//   else if (b.size() < a.size())
//     return LEX_AFTER;
//   else
//     return LEX_MATCH;
// }

// struct LoopException {
//   std::string descr;
//   LoopException(const std::string &s): descr(s) {};
// };

#endif
