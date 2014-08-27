#ifndef OMEGATOOLS_HH
#define OMEGATOOLS_HH

#include <string>
#include <omega.h>
#include "dep.hh"
#include "ir_code.hh"

std::string tmp_e();

void exp2formula(IR_Code *ir, omega::Relation &r, omega::F_And *f_root,
                 std::vector<omega::Free_Var_Decl *> &freevars,
                 omega::CG_outputRepr *repr, omega::Variable_ID lhs, char side,
                 IR_CONDITION_TYPE rel, bool destroy);
omega::Relation arrays2relation(IR_Code *ir, std::vector<omega::Free_Var_Decl*> &freevars,
                                const IR_ArrayRef *ref_src, const omega::Relation &IS_w,
                                const IR_ArrayRef *ref_dst, const omega::Relation &IS_r);
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > relation2dependences(
  const IR_ArrayRef *ref_src, const IR_ArrayRef *ref_dst, const omega::Relation &r);

void exp2constraint(IR_Code *ir, omega::Relation &r, omega::F_And *f_root,
                    std::vector<omega::Free_Var_Decl *> &freevars,
                    omega::CG_outputRepr *repr, bool destroy);

// suif legacy code
// void suif2formula(Relation &r, F_And *f_root,
//                   std::vector<Free_Var_Decl*> &freevars,
//                   operand op, Variable_ID lhs,
//                   char side, char rel);
// void suif2formula(Relation &r, F_And *f_root,
//                   std::vector<Free_Var_Decl*> &freevars,
//                   instruction *ins, Variable_ID lhs,
//                   char side, char rel);
// void add_loop_stride_constraints(omega::Relation &r, omega::F_And *f_root,
//                                  std::vector<omega::Free_Var_Decl*> &freevars,
//                                  tree_for *tnf, char side);
// void add_loop_bound_constraints(IR_Code *ir, omega::Relation &r, omega::F_And *f_root,
//                                 std::vector<omega::Free_Var_Decl*> &freevars,
//                                 tree_for *tnf,
//                                 char upper_or_lower, char side, IR_CONDITION_TYPE rel);
// Relation loop_iteration_space(std::vector<Free_Var_Decl*> &freevars,
//                               tree_node *tn, std::vector<tree_for*> &loops);

// Relation arrays2relation(std::vector<Free_Var_Decl*> &freevars,
//                          in_array *ia_w, const Relation &IS1,
//                          in_array *ia_r, const Relation &IS2);
// std::vector<DependenceVector> relation2dependences(IR_Code *ir, in_array *ia_w,
//                                                    in_array *ia_r, const Relation &r);

// end of suif legacy code

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
