#if ! defined _Relations_h
#define _Relations_h 1

#include <map>
#include <omega/Relation.h>

/** @file */

namespace omega {

// UPDATE friend_rel_ops IN pres_gen.h WHEN ADDING TO THIS LIST
// REMEMBER TO TAKE OUT DEFAULT ARGUMENTS IN THAT FILE

/* The following allows us to avoid warnings about passing 
   temporaries as non-const references.  This is useful but 
   has suddenly become illegal.  */
Relation consume_and_regurgitate(NOT_CONST Relation &R);

/**
 * @defgroup RelOps Operations over relations
 * Operations on relations that produce a relation are assumed to destroy the old relation.
 *
 * Work-around includes passing in another relation created specifically for this purpose or use Relation::copy
 * @{
 */
Relation  Union(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
Relation  Intersection(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
//! Add 1-more input variable to relation
Relation  Extend_Domain(NOT_CONST Relation &R);
//! Add more-more input variables to relation
Relation  Extend_Domain(NOT_CONST Relation &R, int more);
//! Add 1-more output variable to relation
Relation  Extend_Range(NOT_CONST Relation &R);
//! Add more-more output variables to relation
Relation  Extend_Range(NOT_CONST Relation &R, int more);
//! Add 1-more variable to set
Relation  Extend_Set(NOT_CONST Relation &R);
//! Add more-more variables to set
Relation  Extend_Set(NOT_CONST Relation &R, int more);
Relation  Restrict_Domain(NOT_CONST Relation &r1, NOT_CONST Relation &r2); // Takes set as 2nd
Relation  Restrict_Range(NOT_CONST Relation &r1, NOT_CONST Relation &r2);  // Takes set as 2nd
Relation  Domain(NOT_CONST Relation &r);      // Returns set
Relation  Range(NOT_CONST Relation &r);       // Returns set
/**
 * Give two sets, A and B, create a relation whose
 * domain is A and whose range is B.
 */
Relation  Cross_Product(NOT_CONST Relation &A, NOT_CONST Relation &B);  // Takes two sets
//! Inverse the input and output tuple
Relation  Inverse(NOT_CONST Relation &r);
Relation  After(NOT_CONST Relation &r, int carried_by, int new_output,int dir=1);
/**
 * Works for relations only. Input arity must be the same as the output.
 * \f[\{z | \exists x,y: f(x,y) \wedge z = y - z\}\f]
 * @param R
 * @return
 */
Relation  Deltas(NOT_CONST Relation &R);            // Returns set
/**
 * Works for relations only. For the first *eq_no(p)* of input var and output var,
 * \f$\{[c_1,\dots,c_p] | f(\overrightarrow{x},\overrightarrow{y}) \wedge \forall 1 \leq j \leq p, c_j = b_j - a_j\}\f$
 * @param eq_no[in]
 */
Relation  Deltas(NOT_CONST Relation &R, int eq_no); // Returns set
Relation  DeltasToRelation(NOT_CONST Relation &R, int n_input, int n_output);
Relation  Complement(NOT_CONST Relation &r);
Relation  Project(NOT_CONST Relation &R, Global_Var_ID v);
Relation  Project(NOT_CONST Relation &r, int pos, Var_Kind vkind);
/**
 * Works for both relations and sets. Return a new relation with all occurrences
 * of v replaced by existentially quantified z.
 */
Relation  Project(NOT_CONST Relation &S, Variable_ID v);
Relation  Project(NOT_CONST Relation &S, Sequence<Variable_ID> &s);
/**
 * Works with both relations and sets. All global variables projected.
 */
Relation  Project_Sym(NOT_CONST Relation &R);
/**
 * Works with both relations and sets. All input and output variables projected.
 */
Relation  Project_On_Sym(NOT_CONST Relation &R,
                         NOT_CONST Relation &context = Relation::Null());
/**
 * @brief Compute (gist r1 given r2).
 * Assuming that r2 has only one conjunct.
 * r2 may have zero input and output OR may have # in/out vars equal to r1.
 */
Relation  GistSingleConjunct(NOT_CONST Relation &R, NOT_CONST Relation &R2, int effort=0);
/**
 * Works for both relation and sets. The arguments must have the same arity.
 * Returns \f$r = \{x \rightarrow y | f(x,y)\}\f$ such that
 * \f$\forall x,y: f(x,y) \wedge f_2(x,y) \Leftrightarrow f_1(x,y)\f$
 * @param effort[in] how hard we try to make f tight
 */
Relation  Gist(NOT_CONST Relation &R1, NOT_CONST Relation &R2, int effort=0);
/**
 * Works for both relation and sets. Arguments must have the same arity.
 * Calculate r1-r2 by (r1 and !r2).
 */
Relation  Difference(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
/**
 * Works for both relations and sets. For all quantified variables are designated as
 * being able to have rational values so as to be able eliminated exactly(via Fourier
 * variable elimination) when simplifying.
 * @param R
 * @param strides_allowed[in] If true, quantified variables in only one constraints(stride)
 * can't be designated as rational thus unable to be eliminated exactly.
 * @return
 */
Relation  Approximate(NOT_CONST Relation &R, bool strides_allowed = false);
Relation  Identity(int n_inp);
Relation  Identity(NOT_CONST Relation &r);
bool      Must_Be_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
bool      Might_Be_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
bool      Is_Obvious_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
/**
 * Works for relations only.
 * @return F(G(x)) or \f$F\circ G\f$
 */
Relation  Composition(NOT_CONST Relation &F, NOT_CONST Relation &G);
bool      prepare_relations_for_composition(Relation &F, Relation &G);
//! Same as Composition
Relation  Join(NOT_CONST Relation &G, NOT_CONST Relation &F);
Relation  EQs_to_GEQs(NOT_CONST Relation &, bool excludeStrides=false);
/**
 * For a relation R, returns a relation \f$S\subseteq R\f$ where each input, output,
 * or set variable in S has exactly one value. Plus constraints on the symbolic
 * variables.
 */
Relation  Symbolic_Solution(NOT_CONST Relation &S);
/**
 * @param T[in] A set of extra variable to be reduced.
 */
Relation  Symbolic_Solution(NOT_CONST Relation &S, Sequence<Variable_ID> &T);
/**
 * For a relation R, returns a relation \f$S\subseteq R\f$ where each input, output,
 * set, or global variable in S has exactly one value. If R is inexact, the result
 * may be as well.
 */
Relation  Sample_Solution(NOT_CONST Relation &S);
Relation  Solution(NOT_CONST Relation &S, Sequence<Variable_ID> &T);
/**
 * @brief Upper bound of the relation in question
 *
 * Return s such that \f$r \subseteq s\f$ is exact.
 * Works by interpreting all UNKNOWN constraints as true.
 */
Relation  Upper_Bound(NOT_CONST Relation &r);
/**
 * @brief Lower bound of the relation in question
 *
 * Return s such that \f$s \subseteq r\f$ is exact.
 * Works by interpreting all UNKNOWN constraints as false.
 */
Relation  Lower_Bound(NOT_CONST Relation &r);

/**
 *
 * Scramble each relation's variables and merge these relations
 * together. Support variable mapping to and from existentials.
 * Unspecified variables in mapping are mapped to themselves by
 * default. It intends to replace MapRel1 and MapAndCombineRel2
 * functions (the time saved by grafting formula tree might be
 * negligible when compared to the simplification cost).
 */
Relation merge_rels(Tuple<Relation> &R, const Tuple<std::map<Variable_ID, std::pair<Var_Kind, int> > > &mapping, const Tuple<bool> &inverse, Combine_Type ctype, int number_input = -1, int number_output = -1);
// The followings might retire in the futrue!!!
/**
 * Discouraged when there are higher level substitutes. Map the variables from input relations to output relation.
 */
void MapRel1(Relation &inputRel,
             const Mapping &map,
             Combine_Type ctype,
             int number_input=-1, int number_output=-1,
             bool invalidate_resulting_leading_info = true,
             bool finalize = true);
/**
 * Discouraged when there are higher level substitutes. Map the variables from input relations to output relation.
 */
Relation MapAndCombineRel2(Relation &R1, Relation &R2,
                           const Mapping &mapping1, const Mapping &mapping2,
                           Combine_Type ctype,
                           int number_input=-1, int number_output=-1);
void align(Rel_Body *originalr, Rel_Body *newr, F_Exists *fe,
           Formula *f, const Mapping &mapping, bool &newrIsSet,
           List<int> &seen_exists,
           Variable_ID_Tuple &seen_exists_ids);
/** @} */
} // namespace

#endif
