#if ! defined _Relation_h
#define _Relation_h 1

#include <omega/RelBody.h>
#include <omega/pres_cnstr.h>
#include <iostream>
#include <limits.h>

namespace omega {

/**
 * @brief Relation representative.
 *
 * Body and representative are separated to do reference counting to optimize copy of formulas. This owns a Rel_Body
 * that contains the actual formula. Contains a lot of pipe that calls the corresponding functions in Rel_Body.
 *
 * Could be a "set" or "relation",
 */
class Relation {
public:
  //! create a null relation
  Relation();

  /**
   * @brief create a new relation with n_input variables and n_output variables
   *
   * Doesn't contain any Presburger formula can't be used just yet.
   * A set is a relation without output variables.
   */
  Relation(int n_input, int n_output = 0);
  Relation(const Relation &r);
  /**
   * @brief Create a relation by copying a conjunction of constraints c from some other relation r.
   *
   * Conjuncts are created when a relation is simplified into disjunctive normal form.
   */
  Relation(const Relation &r, Conjunct *c);
  Relation &operator=(const Relation &r);
  Relation(Rel_Body &r, int foo);

  static Relation Null();
  static Relation Empty(const Relation &R);
  static Relation True(const Relation &R);
  static Relation True(int setvars);
  static Relation True(int in, int out);
  static Relation False(const Relation &R);
  static Relation False(int setvars);
  static Relation False(int in, int out);
  static Relation Unknown(const Relation &R);
  static Relation Unknown(int setvars);
  static Relation Unknown(int in, int out);


  bool is_null() const;

  ~Relation();

  inline F_Forall *add_forall()
  { return rel_body->add_forall(); }
  inline F_Exists *add_exists()
  { return rel_body->add_exists(); }
  inline F_And    *add_and()
  { return rel_body->add_and(); }
  inline F_And    *and_with()
  { return rel_body->and_with(); }
  inline F_Or     *add_or()
  { return rel_body->add_or(); }
  inline F_Not    *add_not()
  { return rel_body->add_not(); }
  inline void finalize()
  { rel_body->finalize(); }
  inline bool is_finalized() const
  { return rel_body->finalized; }
  inline bool is_set() const
  { return rel_body->is_set();   }  
  inline int n_inp() const
  { return rel_body->n_inp(); }
  inline int n_out() const
  { return rel_body->n_out(); }
  inline int n_set() const
  { return rel_body->n_set(); }

  inline const Variable_ID_Tuple *global_decls() const
  { return rel_body->global_decls(); }
  inline int max_ufs_arity() const
  { return rel_body->max_ufs_arity(); }
  /**
   * @brief Maximum arity of uninterpreted function over input tuple
   */
  inline int max_ufs_arity_of_in() const
  { return rel_body->max_ufs_arity_of_in(); }
  /**
   * @brief Maximum arity of uninterpreted function over set tuple
   */
  inline int max_ufs_arity_of_set() const
  { return rel_body->max_ufs_arity_of_set(); }
  /**
   * @brief Maximum arity of uninterpreted function over output tuple
   */
  inline int max_ufs_arity_of_out() const
  { return rel_body->max_ufs_arity_of_out(); }
  /**
   * @brief Maximum arity of uninterpreted function over input&output tuple
   */
  inline int max_shared_ufs_arity() const
  { return rel_body->max_shared_ufs_arity(); }

  //! Return the n-th input variable, illegal for set
  inline Variable_ID input_var(int nth)
  { return rel_body->input_var(nth); }
  //! Return the n-th output variable, illegal for set
  inline Variable_ID output_var(int nth)
  { return rel_body->output_var(nth); }
  //! Return the n-th set variable, illegal for relation
  inline Variable_ID set_var(int nth)
  { return rel_body->set_var(nth); }
  inline bool has_local(const Global_Var_ID G)
  { return  rel_body->has_local(G); } 
  inline bool has_local(const Global_Var_ID G, Argument_Tuple of)
  { return  rel_body->has_local(G, of); } 
  inline Variable_ID get_local(const Variable_ID v)
  { return split()->get_local(v); }
  /**
   * @brief Find or declare global variable.
   *
   * If the VarID does not exist, it is created. Otherwise it's returned.
   * Note that this version now works only for 0-ary functions.
   */
  inline Variable_ID get_local(const Global_Var_ID G)
  { return split()->get_local(G); }
  /**
   * @brief Find or declare global variable.
   *
   * If the VarID does not exist, it is created. Otherwise it's returned.
   */
  inline Variable_ID get_local(const Global_Var_ID G, Argument_Tuple of)
  { return split()->get_local(G, of); }

  inline void        name_input_var(int nth, Const_String S)
  { split()->name_input_var(nth, S); }
  inline void        name_output_var(int nth, Const_String S)
  { split()->name_output_var(nth, S); }
  inline void        name_set_var(int nth, Const_String S)
  { split()->name_set_var(nth, S); }


  inline F_And      *and_with_and()
  { return split()->and_with_and(); }
  /**
   * Create a top-level EQ constraint that is and-ed with the formula in this relation.
   */
  inline EQ_Handle   and_with_EQ()
  { return split()->and_with_EQ(); }
  /**
   * Set the new EQ constaint's coefficient to be the same as c.
   *
   * Can be used to convert GEQ to EQ.
   */
  inline EQ_Handle   and_with_EQ(const Constraint_Handle &c)
  { return split()->and_with_EQ(c); }
  /**
   * Create a top-level GEQ constraint that is and-ed with the formula in this relation.
   */
  inline GEQ_Handle  and_with_GEQ()
  { return split()->and_with_GEQ(); }
  /**
   * Set the new GEQ constaint's coefficient to be the same as c.
   */
  inline GEQ_Handle  and_with_GEQ(const Constraint_Handle &c)
  { return split()->and_with_GEQ(c); }

  inline void print()
  { rel_body->print(); }
  inline void print(FILE *output_file)
  { rel_body->print(output_file); }
  inline void print_with_subs()
  { rel_body->print_with_subs(); }
  /**
   * @brief Print the relation in an easy-to-understand format
   *
   * At each input variable and output variable, it will try to print the variable as an affine function of the
   * variables to the left.
   *
   * @param printSym Whether the set of symbolic variables used in the relation are printed.
   */
  inline void print_with_subs(FILE *output_file, bool printSym=false,
                              bool newline=true)
  { rel_body->print_with_subs(output_file, printSym, newline); }
  inline std::string print_with_subs_to_string(bool printSym=false, 
                                          bool newline=true)
  { return rel_body->print_with_subs_to_string(printSym, newline); }
  inline std::string print_outputs_with_subs_to_string()
  { return rel_body->print_outputs_with_subs_to_string(); }
  inline std::string print_outputs_with_subs_to_string(int i)
  { return rel_body->print_outputs_with_subs_to_string(i); }
  inline void prefix_print()
  { rel_body->prefix_print(); }
  /**
   * @brief Debug print the structure in prefix format
   *
   * Used primarily to debug programs use this library. Designed to make clear the structure of the formula tree and
   * show the details of the variables used.
   */
  inline void prefix_print(FILE *output_file, int debug = 1)
  { rel_body->prefix_print(output_file, debug); }
  /**
   * @brief Print the formula
   *
   * This allows a printed representation of the relation's formula, without the input and output variables.
   */
  inline std::string print_formula_to_string() {
    return rel_body->print_formula_to_string();
  }
  void dimensions(int & ndim_all, int &ndim_domain);

  /**
   * Return True if the relation's lower-bound is satisfiable. Treating UNKNOWN constraints as False.
   */
  inline bool is_lower_bound_satisfiable()
  { return rel_body->is_lower_bound_satisfiable(); }
  /**
   * Return True if the relation's upper-bound is satisfiable. Treating UNKNOWN constraints as True.
   */
  inline bool is_upper_bound_satisfiable()
  { return rel_body->is_upper_bound_satisfiable(); }
  /**
   * @brief If both bounds are satisfiable or not. ABORT if only one is.
   *
   * Included for compatibility with older releases.
   */
  inline bool is_satisfiable()
  { return rel_body->is_satisfiable(); }

  inline bool is_tautology()
  { return rel_body->is_obvious_tautology(); }  // for compatibility
  /**
   * @brief if formula evaluates to a single conjunction with no constraints
   */
  inline bool is_obvious_tautology()
  { return rel_body->is_obvious_tautology(); }
  /**
   * @brief if the relation's formula is a tautology
   */
  inline bool is_definite_tautology()
  { return rel_body->is_definite_tautology(); }

  /**
   * @return the number of conjuncts
   */
  inline int    number_of_conjuncts()
  { return rel_body->query_DNF()->length(); }

  /**
   * @return x s.t. forall conjuncts c, c has >= x leading 0s(in=out)
   * for set or there are no conjuncts return -1
   */
  inline int    query_guaranteed_leading_0s()
  { return rel_body->query_DNF()->query_guaranteed_leading_0s(this->is_set() ? -1 : 0); }

  /**
   * @return x s.t. forall conjuncts c, c has <= x leading 0s(in=out)
   * if no conjuncts return min of input and output tuple sizes, or -1 if relation is a set
   */
  inline int    query_possible_leading_0s()
  { return rel_body->query_DNF()->query_possible_leading_0s(
      this->is_set()? -1 : min(n_inp(),n_out())); }

  /**
   * @return +-1 according to sign of leading dir, or 0 if we don't know
   */
  inline int    query_leading_dir()
  { return rel_body->query_DNF()->query_leading_dir(); }

  //! Request this to be simplified to DNF
  inline DNF*    query_DNF()
  { return rel_body->query_DNF(); }
  /**
   * @brief Request this to be simplified to DNF
   *
   * rdt_conjs and rdt_constrs specifies the level of effort to eliminate redundant informations
   *
   * value | rdt_conjs                                 | rdt_constrs
   * ----- | ----------------------------------------- | ----------------------------------------------
   * 0     | Nothing extra                             | Nothing extra
   * 1     | Simple check                              | Remove redundant ones by any other two
   * 2     | Exact test(if one is subset of any other) | Exact test
   * 4     |                                           | Also perform simplification on the constraints
   */
  inline DNF*    query_DNF(int rdt_conjs, int rdt_constrs)
  { return rel_body->query_DNF(rdt_conjs, rdt_constrs); }
  /**
   * @brief Simplify a given relation.
   *
   * Store the resulting DNF in the relation, clean out the formula.
   *
   * Called by query_DNF.
   */
  inline void    simplify(int rdt_conjs = 0, int rdt_constrs = 0)
  { rel_body->simplify(rdt_conjs, rdt_constrs); }
  inline bool is_simplified()
  { return rel_body->is_simplified(); }
  inline bool is_compressed() const
  { return rel_body->is_compressed(); }
  inline Conjunct *rm_first_conjunct()
  { return rel_body->rm_first_conjunct(); }
  inline Conjunct *single_conjunct()
  { return rel_body->single_conjunct(); }
  inline bool has_single_conjunct()
  { return rel_body->has_single_conjunct(); }

  /**
   * @brief Determining the bounds of the difference of two variables
   *
   * This is used to calculate leading zeros
   * @param lowerBound[out] negInfinity if not bounded below
   * @param upperBound[out] posInfinity if not bounded above
   * @param guaranteed[out] True if the bounds is guaranteed to be tight
   */
  void query_difference(Variable_ID v1, Variable_ID v2, coef_t &lowerBound, coef_t &upperBound, bool &guaranteed) {
    rel_body->query_difference(v1, v2, lowerBound, upperBound, guaranteed);
  }
  void query_variable_bounds(Variable_ID v, coef_t &lowerBound, coef_t &upperBound) {
    rel_body->query_variable_bounds(v,lowerBound,upperBound);
  }
  coef_t query_variable_mod(Variable_ID v, coef_t factor) {
    assert(factor > 0);
    return rel_body->query_variable_mod(v, factor);
  }
  int query_variable_mod(Variable_ID v, int factor) {
    assert(sizeof(int) <= sizeof(coef_t));
    coef_t result = rel_body->query_variable_mod(v, static_cast<coef_t>(factor));
    if (result == posInfinity)
      return INT_MAX;
    else
      return static_cast<int>(result);
  }
  

  inline void make_level_carried_to(int level)
  {
    split()->make_level_carried_to(level);
  }

  inline Relation extract_dnf_by_carried_level(int level, int direction)
  {
    return split()->extract_dnf_by_carried_level(level, direction);
  }

  inline void compress()
  {
#if defined(INCLUDE_COMPRESSION)
    split()->compress(); 
#endif
  }
  void uncompress()
  { rel_body->uncompress(); }

  //! If it doesn't contains UNKNOWN
  inline bool is_exact() const
  { return !(rel_body->unknown_uses() & (and_u | or_u)) ; }
  //! If it contains UNKNOWN
  inline bool is_inexact() const
  { return !is_exact(); }
  //! If it is a single UNKNOWN
  inline bool is_unknown() const
  { return rel_body->is_unknown(); }
  inline Rel_Unknown_Uses unknown_uses() const
  { return rel_body->unknown_uses(); }

  void setup_names() {rel_body->setup_names();}
  void copy_names(const Relation &r) {
    copy_names(*r.rel_body);
  };
  void copy_names(Rel_Body &r);

private:
  // Functions that have to create sets from relations:
  friend class Rel_Body;
  friend_rel_ops;

  /**
   * @brief Create a separate body for this representation
   *
   * One of the representatives using the body wants to be changed.
   * Return the address of the new body, with the old representative
   * pointed to the new body.
   */
  Rel_Body *split();

  DNF* simplified_DNF() {
    simplify();
    return rel_body->simplified_DNF;
  };

  inline void invalidate_leading_info(int changed = -1)
  { split()->invalidate_leading_info(changed); }
  inline void enforce_leading_info(int guaranteed, int possible, int dir)
  {
    split()->enforce_leading_info(guaranteed, possible, dir);
  }


  void    makeSet();
  void    markAsSet();
  void    markAsRelation();

  friend bool operator==(const Relation &, const Relation &);

  void reverse_leading_dir_info()
  { split()->reverse_leading_dir_info(); }
  void interpret_unknown_as_true()
  { split()->interpret_unknown_as_true(); }
  void interpret_unknown_as_false()
  { split()->interpret_unknown_as_false(); }


  Rel_Body *rel_body;


  friend Relation merge_rels(Tuple<Relation> &R, const Tuple<std::map<Variable_ID, std::pair<Var_Kind, int> > > &mapping, const Tuple<bool> &inverse, Combine_Type ctype, int number_input, int number_output);
};

inline std::ostream & operator<<(std::ostream &o, Relation &R)
{
  return o << R.print_with_subs_to_string();
}

Relation copy(const Relation &r);

} // namespace

#include <omega/Relations.h>

#endif
