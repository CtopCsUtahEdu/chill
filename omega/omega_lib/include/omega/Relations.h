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
Relation  Extend_Domain(NOT_CONST Relation &R);
Relation  Extend_Domain(NOT_CONST Relation &R, int more);
Relation  Extend_Range(NOT_CONST Relation &R);
Relation  Extend_Range(NOT_CONST Relation &R, int more);
Relation  Extend_Set(NOT_CONST Relation &R);
Relation  Extend_Set(NOT_CONST Relation &R, int more);
Relation  Restrict_Domain(NOT_CONST Relation &r1, NOT_CONST Relation &r2); // Takes set as 2nd
Relation  Restrict_Range(NOT_CONST Relation &r1, NOT_CONST Relation &r2);  // Takes set as 2nd
Relation  Domain(NOT_CONST Relation &r);      // Returns set
Relation  Range(NOT_CONST Relation &r);       // Returns set
Relation  Cross_Product(NOT_CONST Relation &A, NOT_CONST Relation &B);  // Takes two sets
Relation  Inverse(NOT_CONST Relation &r);
Relation  After(NOT_CONST Relation &r, int carried_by, int new_output,int dir=1);
Relation  Deltas(NOT_CONST Relation &R);            // Returns set
Relation  Deltas(NOT_CONST Relation &R, int eq_no); // Returns set
Relation  DeltasToRelation(NOT_CONST Relation &R, int n_input, int n_output);
Relation  Complement(NOT_CONST Relation &r);
Relation  Project(NOT_CONST Relation &R, Global_Var_ID v);
Relation  Project(NOT_CONST Relation &r, int pos, Var_Kind vkind);
Relation  Project(NOT_CONST Relation &S, Variable_ID v);
Relation  Project(NOT_CONST Relation &S, Sequence<Variable_ID> &s);
Relation  Project_Sym(NOT_CONST Relation &R);
Relation  Project_On_Sym(NOT_CONST Relation &R,
                         NOT_CONST Relation &context = Relation::Null());
Relation  GistSingleConjunct(NOT_CONST Relation &R, NOT_CONST Relation &R2, int effort=0);
Relation  Gist(NOT_CONST Relation &R1, NOT_CONST Relation &R2, int effort=0);
Relation  Difference(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
Relation  Approximate(NOT_CONST Relation &R, bool strides_allowed = false);
Relation  Identity(int n_inp);
Relation  Identity(NOT_CONST Relation &r);
bool      Must_Be_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
bool      Might_Be_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
bool      Is_Obvious_Subset(NOT_CONST Relation &r1, NOT_CONST Relation &r2);
Relation  Composition(NOT_CONST Relation &F, NOT_CONST Relation &G);
bool      prepare_relations_for_composition(Relation &F, Relation &G);
Relation  Join(NOT_CONST Relation &G, NOT_CONST Relation &F);
Relation  EQs_to_GEQs(NOT_CONST Relation &, bool excludeStrides=false);
Relation  Symbolic_Solution(NOT_CONST Relation &S); 
Relation  Symbolic_Solution(NOT_CONST Relation &S, Sequence<Variable_ID> &T);
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
