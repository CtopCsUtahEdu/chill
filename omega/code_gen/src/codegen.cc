/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   CodeGen class as entry point for code generation.

 Notes:
   Loop variable name prefix should not cause any possible name conflicts
 with original loop variables wrapped in statement holder. This guarantees
 that variable substitution done correctly in the generated code.

 History:
   04/24/96 MMGenerateCode, added by Fortran D people. Lei Zhou
   09/17/08 loop overhead removal based on actual nesting depth -- by chun
   03/05/11 fold MMGenerateCode into CodeGen class, Chun Chen
*****************************************************************************/

#include <typeinfo>
#include <omega.h>
#include <basic/util.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "chill_io.hh"

#include <code_gen/CG.h>
#include <omega/code_gen/include/codegen.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/codegen_error.h>
#include <code_gen/CG_utils.h>

namespace omega {

const std::string CodeGen::loop_var_name_prefix = "t";
const int CodeGen::var_substitution_threshold = 10;

//Anand--adding stuff to make Chun's code work with Gabe's
std::vector< std::vector<int> > smtNonSplitLevels;
std::vector< std::vector<std::string> > loopIdxNames;//per stmt
std::vector< std::pair<int, std::string> > syncs;



CodeGen::CodeGen(const std::vector<Relation> &xforms, const std::vector<Relation> &IS, const Relation &known, std::vector< std::vector<int> > smtNonSplitLevels_ , std::vector< std::vector<std::string> > loopIdxNames_,  std::vector< std::pair<int, std::string> > syncs_) {

  debug_fprintf(stderr, "CodeGen::CodeGen() sanity checking\n");
  // check for sanity of parameters
  int num_stmt = IS.size();
  if (xforms.size() != num_stmt)
    throw std::invalid_argument("number of iteration spaces does not match number of transformations");
  known_ = copy(known);
  if (known_.n_out() != 0)
    throw std::invalid_argument("known condition must be a set relation");
  if (known_.is_null())
    known_ = Relation::True(0);
  else
    known_.simplify(2, 4);
  if (!known_.is_upper_bound_satisfiable())
    throw std::invalid_argument("Known condition is not satisfiable");
  if (known_.number_of_conjuncts() > 1)
    throw std::invalid_argument("only one conjunct allowed in known condition");

  debug_fprintf(stderr, "num_stmt %d  %d xforms\n", num_stmt, xforms.size()); 
  xforms_ = xforms;
  for (int i = 0; i < num_stmt; i++) {
    xforms_[i].simplify();
    if (!xforms_[i].has_single_conjunct())
      throw std::invalid_argument("mapping relation must have only one conjunct");
    if (xforms_[i].n_inp() != IS[i].n_inp() || IS[i].n_out() != 0)
      throw std::invalid_argument("illegal iteration space or transformation arity");
  }

  //protonu--
  //easier to handle this as a global
  smtNonSplitLevels = smtNonSplitLevels_;
  syncs = syncs_;
  loopIdxNames = loopIdxNames_;

  debug_begin
    fprintf(stderr, "codegen.cc loopIdxNames.size() %lu\n", loopIdxNames.size());
    for (int i=0; i<loopIdxNames.size(); i++) {
      fprintf(stderr, "\n");
      for (int j=0; j<loopIdxNames[i].size(); j++)
        fprintf(stderr, "i %d   j %d %s\n", i, j,loopIdxNames[i][j].c_str() );
    }
  debug_end

  //end-protonu



  // find the maximum iteration space dimension we are going to operate on
  int num_level = known_.n_inp();
  for (int i = 0; i < num_stmt; i++)
    if (xforms_[i].n_out() > num_level)
      num_level = xforms_[i].n_out();
  known_ = Extend_Set(known_, num_level-known_.n_inp());
  for (int i = 1; i <= num_level; i++)
    known_.name_set_var(i, loop_var_name_prefix + to_string(i));
  known_.setup_names();

  // split disjoint conjunctions in original iteration spaces
  std::vector<Relation> new_IS;
  for (int i = 0; i < num_stmt; i++) {
    for (int j = 1; j <= IS[i].n_inp(); j++)
      xforms_[i].name_input_var(j, const_cast<std::vector<Relation> &>(IS)[i].input_var(j)->name());
    for (int j = 1; j <= xforms_[i].n_out(); j++)
      xforms_[i].name_output_var(j, loop_var_name_prefix + to_string(j));
    xforms_[i].setup_names();

    Relation S = Restrict_Domain(copy(xforms_[i]), copy(IS[i]));
    Relation R = Range(S);
    R = Intersection(Extend_Set(R, num_level-R.n_inp()), copy(known_));
    R.simplify(2, 4);

    if (R.is_inexact())
      throw codegen_error("cannot generate code for inexact iteration spaces");

    while(R.is_upper_bound_satisfiable()) {
      DNF *dnf = R.query_DNF();
      DNF_Iterator c(dnf);
      Relation R2 = Relation(R, *c);
      R2.simplify();
      new_IS.push_back(copy(R2));
      remap_.push_back(i);
      c.next();
      if (!c.live()) 
        break;
      Relation remainder = Relation::False(R);
      while (c.live()) {
        remainder = Union(remainder, Relation(R, *c));
        c.next();
      }
      R = Difference(remainder, R2);
      R.simplify(2, 4);
    }
  }

  // number of new statements after splitting
  num_stmt = new_IS.size();
  if(!smtNonSplitLevels.empty())
      smtNonSplitLevels.resize(num_stmt);

  // assign negative infinity to extra loops created for the purpose of expanding to maximum dimension
  for (int i = 0; i < num_stmt; i++) {
    if (xforms[remap_[i]].n_out() < num_level) {
      F_And *f_root = new_IS[i].and_with_and();
      for (int j = xforms[remap_[i]].n_out()+1; j <= num_level; j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(new_IS[i].set_var(j), 1);
        h.update_const(posInfinity);
      }
      new_IS[i].simplify();
    }
  }

  // calculate projected subspaces for each loop level once and save for CG tree manipulation later
  projected_IS_ = std::vector<std::vector<Relation> >(num_level);
  for (int i = 0; i < num_level; i++)
    projected_IS_[i] = std::vector<Relation>(num_stmt);
  for (int i = 0; i < num_stmt; i++) {
    if (num_level > 0)
      projected_IS_[num_level-1][i] = new_IS[i];
    for (int j = num_level-1; j >= 1; j--) {
      projected_IS_[j-1][i] = Project(copy(projected_IS_[j][i]), j+1, Set_Var);
      projected_IS_[j-1][i].simplify(2, 4);
      //projected_IS_[j-1][i] = checkAndRestoreIfProjectedByGlobal(projected_IS_[j][i], projected_IS_[j-1][i],
      // 		  projected_IS_[j-1][i].set_var(j));
    }
  }
  debug_fprintf(stderr, "CodeGen::CodeGen() DONE\n"); 
}


CG_result *CodeGen::buildAST(int level, const BoolSet<> &active, bool split_on_const, const Relation &restriction) {
  if (level > num_level())
    return new CG_leaf(this, active);

  int num_active_stmt = active.num_elem();
  if (num_active_stmt == 0)
    return NULL;
  else if (num_active_stmt == 1)
    return new CG_loop(this, active, level, buildAST(level+1, active, true, restriction));

  // use estimated constant bounds for fast non-overlap iteration space splitting
  if (split_on_const) {
    std::vector<std::pair<std::pair<coef_t, coef_t>, int> > bounds;

    for (BoolSet<>::const_iterator i = active.begin(); i != active.end(); i++) {
      Relation r = Intersection(copy(projected_IS_[level-1][*i]), copy(restriction));
      r.simplify(2, 4);
      if (!r.is_upper_bound_satisfiable())
        continue;
      coef_t lb, ub;
      r.single_conjunct()->query_variable_bounds(r.set_var(level),lb,ub);
      bounds.push_back(std::make_pair(std::make_pair(lb, ub), *i));
    }
    sort(bounds.begin(), bounds.end());

    std::vector<Relation> split_cond;
    std::vector<CG_result *> split_child;

    coef_t prev_val = -posInfinity;
    coef_t next_val = bounds[0].first.second;
    BoolSet<> next_active(active.size());
    int i = 0;
    while (i < bounds.size()) {
      if (bounds[i].first.first <= next_val) {
        next_active.set(bounds[i].second);
        next_val = max(next_val, bounds[i].first.second);
        i++;
      }
      else {
        Relation r(num_level());
        F_And *f_root = r.add_and();
        if (prev_val != -posInfinity) {
          GEQ_Handle h = f_root->add_GEQ();
          h.update_coef(r.set_var(level), 1);
          h.update_const(-prev_val-1);
        }
        if (next_val != posInfinity) {
          GEQ_Handle h = f_root->add_GEQ();
          h.update_coef(r.set_var(level), -1);
          h.update_const(next_val);
        }
        r.simplify();

        Relation new_restriction = Intersection(copy(r), copy(restriction));
        new_restriction.simplify(2, 4);
        CG_result *child = buildAST(level, next_active, false, new_restriction);
        if (child != NULL) {
          split_cond.push_back(copy(r));
          split_child.push_back(child);
        }
        next_active.unset_all();
        prev_val = next_val;
        next_val = bounds[i].first.second;
      }
    }
    if (!next_active.empty()) {
      Relation r = Relation::True(num_level());
      if (prev_val != -posInfinity) {
        F_And *f_root = r.and_with_and();
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(r.set_var(level), 1);
        h.update_const(-prev_val-1);
        r.simplify();
      }
      Relation new_restriction = Intersection(copy(r), copy(restriction));
      new_restriction.simplify(2, 4);
      CG_result *child = buildAST(level, next_active, false, new_restriction);
      if (child != NULL) {
        split_cond.push_back(copy(r));
        split_child.push_back(child);
      }
    }

    if (split_child.size() == 0)
      return NULL;
    else if (split_child.size() == 1)
      return split_child[0];
    else
      return new CG_split(this, active, split_cond, split_child);
  }
  // check bound conditions exhaustively for non-overlap iteration space splitting
  else {
    std::vector<Relation> Rs(active.size());
    for (BoolSet<>::const_iterator i = active.begin(); i != active.end(); i++) {
      Rs[*i] = Intersection(Approximate(copy(projected_IS_[level-1][*i])), copy(restriction));
      Rs[*i].simplify(2, 4);
    }
    Relation hull = SimpleHull(Rs);

    //protonu-warn Chun about this change
    //This does some fancy splitting of statements into loops with the
    //fewest dimentions, but that's not necessarily what we want when
    //code-gening for CUDA. smtNonSplitLevels keeps track per-statment of
    //the levels that should not be split on.
    bool checkForSplits = true;
    for (auto i = active.begin(); i != active.end(); i++) {
      if (*i < smtNonSplitLevels.size())
        for (auto lev: smtNonSplitLevels[*i])
          if (lev == (level - 2)) {
            checkForSplits = false;
            break;
          }
    }

    for (auto i = active.begin(); i != active.end() && checkForSplits; i++) {
      Relation r = Gist(copy(Rs[*i]), copy(hull), 1);
      if (r.is_obvious_tautology())
        continue;
      r = EQs_to_GEQs(r);

      for (GEQ_Iterator e = r.single_conjunct()->GEQs(); e; e++) {
        if ((*e).has_wildcards())
          continue;
            
        Relation cond = Relation::True(num_level());
        BoolSet<> first_chunk(active.size());
        BoolSet<> second_chunk(active.size());

        if ((*e).get_coef(hull.set_var(level)) > 0) {
          cond.and_with_GEQ(*e);
          cond = Complement(cond);;
          cond.simplify();
          second_chunk.set(*i);
        }
        else if ((*e).get_coef(hull.set_var(level)) < 0) {
          cond.and_with_GEQ(*e);
          cond.simplify();
          first_chunk.set(*i);
        }
        else
          continue;

        bool is_proper_split_cond = true;
        for (BoolSet<>::const_iterator j = active.begin(); j != active.end(); j++)
          if ( *j != *i) {
          bool in_first = Intersection(copy(Rs[*j]), copy(cond)).is_upper_bound_satisfiable();
          bool in_second = Difference(copy(Rs[*j]), copy(cond)).is_upper_bound_satisfiable();

          if (in_first && in_second) {
            is_proper_split_cond = false;
            break;
          }

          if (in_first)
            first_chunk.set(*j);
          else if (in_second)
            second_chunk.set(*j);
          }

        if (is_proper_split_cond && first_chunk.num_elem() != 0 && second_chunk.num_elem() != 0) {
          CG_result *first_cg = buildAST(level, first_chunk, false, copy(cond));
          CG_result *second_cg = buildAST(level, second_chunk, false, Complement(copy(cond)));
          if (first_cg == NULL)
            return second_cg;
          else if (second_cg == NULL)
            return first_cg;
          else {
            std::vector<Relation> split_cond;
            std::vector<CG_result *> split_child;
            split_cond.push_back(copy(cond));
            split_child.push_back(first_cg);
            split_cond.push_back(Complement(copy(cond)));
            split_child.push_back(second_cg);

            return new CG_split(this, active, split_cond, split_child);
          }
        }
      }
    }
    return new CG_loop(this, active, level, buildAST(level+1, active, true, restriction));
  }
}


CG_result *CodeGen::buildAST(int effort) {
  debug_fprintf(stderr, "CodeGen::buildAST( effort %d )\n", effort); 
  if (remap_.size() == 0)
    return NULL;

  CG_result *cgr = buildAST(1, ~BoolSet<>(remap_.size()), true, Relation::True(num_level()));
  if (cgr == NULL)
    return NULL;


  // break down the complete iteration space condition to levels of bound/guard condtions
  cgr = cgr->recompute(cgr->active_, copy(known_), copy(known_));



  if (cgr == NULL)
    return NULL;

  // calculate each loop's nesting depth
  int depth = cgr->populateDepth();


  // redistribute guard condition locations by additional splittings
  std::pair<CG_result *, Relation> result = cgr->liftOverhead(min(effort,depth), false);

  // since guard conditions are postponed for non-loop levels, hoist them now.
  // this enables proper if-condition simplication when outputting actual code.
  result.first->hoistGuard();




  return result.first;
}

}
