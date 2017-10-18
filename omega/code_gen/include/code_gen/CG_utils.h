#ifndef _CG_UTILS_H
#define _CG_UTILS_H

#include <omega.h>
#include <code_gen/CG_outputBuilder.h>
#include <basic/boolset.h>
#include <vector>
#include <set>
#include <map>

namespace omega {

  /**
   * @brief Output the inequality constraints containing v
   * The return is only one side of the inequality without v
   */
  CG_outputRepr *
  output_inequality_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const Relation &R,
                         const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                         const std::map<std::string, std::vector<CG_outputRepr *> > &unin,
                         std::set<Variable_ID> excluded_floor_vars = std::set<Variable_ID>());

  /**
   * @brief Create substituting value from equality constraint
   * @param apply_v_coef Whether v's coefficient has any effect
   */
  CG_outputRepr *
  output_substitution_repr(CG_outputBuilder *ocg, const EQ_Handle &equality, Variable_ID v, bool apply_v_coef,
                           const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                           const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief Wrapper to output_inequality_repr
   * When returning NULL, it will replace it with literal 0 in output
   */
  CG_outputRepr *output_upper_bound_repr(CG_outputBuilder *ocg,
                                         const GEQ_Handle &inequality,
                                         Variable_ID v,
                                         const Relation &R,
                                         const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                         const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief output lower bound with respect to lattice(starting iteration)
   */
  CG_outputRepr *output_lower_bound_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v,
                                         const EQ_Handle &stride_eq, Variable_ID wc, const Relation &R,
                                         const Relation &known,
                                         const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                         const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief output the reference to variable v
   * Return the variable by its name, however if this variable need to be substituted,
   * as in assigned_on_the_fly, return the substitution.
   */
  CG_outputRepr *output_ident(CG_outputBuilder *ocg, const Relation &R, Variable_ID v,
                              const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                              const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief output the assignment for loop variable at level
   * It will print if condition when the assignment constains mod constraint.
   * such that coefficient is not 1.
   */
  std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> >
  output_assignment(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known,
                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                    const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief output the loop control structure at level
   * Finding stride using find_simplest_stride and calculating bound using GEQs.
   * Multiple same sided bound will generate min/max operation.
   */
  CG_outputRepr *output_loop(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known,
                             const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                             const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief Output the guard condition
   * Output the guard conditions as captured in R
   */
  CG_outputRepr *output_guard(CG_outputBuilder *ocg, const Relation &R,
                              const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                              const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief Find all substitutions based on current mapping
   * Find substitution for each output variable in R, this can handle integer division
   */
  std::vector<CG_outputRepr *> output_substitutions(CG_outputBuilder *ocg, const Relation &R,
                                                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                                    const std::map<std::string, std::vector<CG_outputRepr *> > &unin);

  /**
   * @brief If the stride equality is guaranteed to hit bound in inequality
   */
  bool bound_must_hit_stride(const GEQ_Handle &inequality, Variable_ID v, const EQ_Handle &stride_eq, Variable_ID wc,
                             const Relation &bounds, const Relation &known);

  /**
   * @brief Find the simplest(cheapest by cost function) assignment of variable v
   * This handles floor definition wildcards in equality, the second in returned pair
   * is the cost.
   */
  std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R, Variable_ID v,
                                                     const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly = std::vector<std::pair<CG_outputRepr *, int> >(),
                                                     bool *has_global_inspector = NULL);

  /**
   * @breif find floor definition for wildcard variable v
   * e.g. m-c <= 4v <= m, (c is constant and 0 <= c < 4). this translates to
   * v = floor(m, 4) and return 4v<=m in this case. All wildcards in such
   * inequality are also floor defined.
   */
  std::pair<bool, GEQ_Handle> find_floor_definition(const Relation &R, Variable_ID v,
                                                    std::set<Variable_ID> excluded_floor_vars = std::set<Variable_ID>());

  /**
   * @brief replicate the floor definition(possibly cascaded to new relation)
   * parameter f_root is inside f_exists, not the other way around.
   * return replicated variable in new relation, with all cascaded floor definitions
   * using wildcards defined in the same way as in the original relation.
   */
  Variable_ID replicate_floor_definition(const Relation &R, const Variable_ID floor_var, Relation &r,
                                         F_Exists *f_exists, F_And *f_root,
                                         std::map<Variable_ID, Variable_ID> &exists_mapping);

  /**
   * @brief find the stride involving the specified variable
   * e.g. v = 2alpha + c
   * The stride equality can have other wildcards as long as they are defined as
   * floor variables.
   */
  std::pair<EQ_Handle, Variable_ID> find_simplest_stride(const Relation &R, Variable_ID v);

  /**
   * @brief pick one guard condition from relation.
   * It can involve multiple constraints when involving wildcards,
   * as long as its complement is a single conjunct.
 */
  Relation pick_one_guard(const Relation &R, int level = 0);

  //! Check if a set/input var is projected out of a inequality by a global variable with arity > 0
  Relation checkAndRestoreIfProjectedByGlobal(const Relation &R1, const Relation &R2, Variable_ID v);

  std::string print_to_iegen_string(Relation &R);
}

#endif
