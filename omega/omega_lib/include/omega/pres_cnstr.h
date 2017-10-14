#if ! defined _pres_cnstr_h
#define _pres_cnstr_h 1

#include <omega/pres_var.h>
#include <vector>

namespace omega {

//
// Constraint handles
//



void copy_constraint(Constraint_Handle H, const Constraint_Handle initial);

class Constraint_Handle {
public:
  Constraint_Handle() {}
  virtual ~Constraint_Handle() {}
  //! ADD delta to the coefficient of the variable
  void   update_coef(Variable_ID, coef_t delta);
  //! ADD delta to the constant term of the constraint
  void   update_const(coef_t delta);
  coef_t get_coef(Variable_ID v) const;
  coef_t get_const() const;
  bool   has_wildcards() const;
  int    max_tuple_pos() const;
  int    min_tuple_pos() const;
  //! Constraint is constant on variable v
  bool is_const(Variable_ID v);
  //! Constraint is constant that only involves v and globals
  bool is_const_except_for_global(Variable_ID v);

  virtual std::string print_to_string() const;
  virtual std::string print_term_to_string() const;
  
  Variable_ID get_local(const Global_Var_ID G);
  Variable_ID get_local(const Global_Var_ID G, Argument_Tuple of);
  // not sure that the second one can be used in a meaningful
  // way if the conjunct is in multiple relations

  //! Multiply each coefficient and the constant term by multiplier
  void   finalize();
  void   multiply(int multiplier);
  Rel_Body *relation() const;


protected:
  Conjunct *c;
  eqn      **eqns;
  int      e;

  friend class Constr_Vars_Iter;
  friend class Constraint_Iterator; 

  Constraint_Handle(Conjunct *, eqn **, int);

#if defined PROTECTED_DOESNT_WORK
  friend class EQ_Handle;
  friend class GEQ_Handle;
#endif

  void   update_coef_during_simplify(Variable_ID, coef_t delta);
  void   update_const_during_simplify(coef_t delta);
  coef_t    get_const_during_simplify() const;
  coef_t    get_coef_during_simplify(Variable_ID v) const;

  
public:
  friend class Conjunct;  // assert_leading_info updates coef's
  // as does move_UFS_to_input
  friend class DNF;       // and DNF::make_level_carried_to

  friend void copy_constraint(Constraint_Handle H,
                              const Constraint_Handle initial);
  // copy_constraint does updates and gets at c and e

};
/**
 * @brief Handle to access GEQ constraints
 *
 * Represent constraints in the form
 * \f[\sum_i a_ix_i + a_0 \geq 0\f]
 */
class GEQ_Handle : public Constraint_Handle {
public:
  inline GEQ_Handle() {}

  virtual std::string print_to_string() const;
  virtual std::string print_term_to_string() const;
  bool operator==(const Constraint_Handle &that);

  void   negate();

private:
  friend class Conjunct;
  friend class GEQ_Iterator;

  GEQ_Handle(Conjunct *, int);
};

/**
 * @brief Handle to access EQ constraints
 *
 * Represent constraints in the form
 * \f[\sum_i a_ix_i + a_0 = 0\f]
 *
 * Note that a stride constraint(\f$(\sum a_ix_i+a_0) \% s = 0\f$) is the same as
 * \f[\exists k, \sum_i a_ix_i + a_0 = ks\f]
 */
class EQ_Handle : public Constraint_Handle {
public:
  inline EQ_Handle() {}

  virtual std::string print_to_string() const;
  virtual std::string print_term_to_string() const;
  bool operator==(const Constraint_Handle &that);

private:
  friend class Conjunct;
  friend class EQ_Iterator;
  
  EQ_Handle(Conjunct *, int);
};


//
// Conjuct iterators -- for querying resulting DNF.
//
class Constraint_Iterator : public Generator<Constraint_Handle> {
public:
  Constraint_Iterator(Conjunct *);
  int  live() const;
  void operator++(int);
  void operator++();
  Constraint_Handle operator* ();
  Constraint_Handle operator* () const;

private:
  Conjunct *c;
  int current,last;
  eqn **eqns;
};


class EQ_Iterator : public Generator<EQ_Handle> {
public:
  //! Get the EQs in a conjunct
  EQ_Iterator(Conjunct *);
  int  live() const;
  void operator++(int);
  void operator++();
  EQ_Handle operator* ();
  EQ_Handle operator* () const;

private:
  Conjunct *c;
  int current, last;
};


class GEQ_Iterator : public Generator<GEQ_Handle> {
public:
  //! Get the GEQs in a conjunct
  GEQ_Iterator(Conjunct *);
  int  live() const;
  void operator++(int);
  void operator++();
  GEQ_Handle operator* ();
  GEQ_Handle operator* () const;

private:
  Conjunct *c;
  int current, last;
};


//
// Variables of constraint iterator.
//
struct Variable_Info {
  Variable_ID var;
  coef_t      coef;
  Variable_Info(Variable_ID _var, coef_t _coef)
    { var = _var; coef = _coef; }
};

class Constr_Vars_Iter : public Generator<Variable_Info> {
public:
  Constr_Vars_Iter(const Constraint_Handle &ch, bool _wild_only = false);
  int         live() const;
  void        operator++(int);
  void        operator++();
  Variable_Info operator*() const;

  Variable_ID curr_var() const;
  coef_t      curr_coef() const;

private:
  eqn               **eqns;
  int               e;
  Problem           *prob;
  Variable_ID_Tuple &vars;
  bool              wild_only;
  int               current;
};

} // namespace

#endif
