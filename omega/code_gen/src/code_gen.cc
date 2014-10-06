/*****************************************************************************
 Copyright (C) 1994-2000 University of Maryland
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Start code generation process here.

 Notes:

 History:
   04/24/96 MMGenerateCode implementation, added by D people. Lei Zhou
*****************************************************************************/

#include <omega.h>
#include <omega/Rel_map.h>
#include <basic/Collection.h>
#include <basic/Bag.h>
#include <basic/Map.h>
#include <basic/util.h>
#include <basic/omega_error.h>
#include <math.h>
#include <vector>

#include <code_gen/CG.h>
#include <code_gen/code_gen.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_outputRepr.h>
#include <code_gen/CG_stringBuilder.h>
#include <code_gen/CG_stringRepr.h>
#include <code_gen/output_repr.h>

namespace omega {


int last_level;// Should not be global, but it is.
SetTuple new_IS;
SetTupleTuple projected_nIS;
Tuple<CG_outputRepr *> statementInfo;
RelTuple transformations;

//protonu--adding stuff to make Chun's code work with Gabe's
Tuple< Tuple<int> > smtNonSplitLevels;
Tuple< Tuple<std::string> > loopIdxNames;//per stmt
std::vector< std::pair<int, std::string> > syncs;

//protonu-putting this in for now, not sure what all these do
//This lovely ugly hack allows us to extract hard upper-bounds at
//specific loop levels
int checkLoopLevel;
int stmtForLoopCheck;
int upperBoundForLevel;
int lowerBoundForLevel;
bool fillInBounds;

//trick to static init checkLoopLevel to 0
class JunkStaticInit{ public: JunkStaticInit(){ checkLoopLevel=0; fillInBounds=false;} };
static JunkStaticInit junkInitInstance__;

//end--protonu.


CG_result * gen_recursive(int level, IntTuple &isActive);


int code_gen_debug=0;


SetTuple filter_function_symbols(SetTuple &sets, bool keep_fs){
  SetTuple new_sets(sets.size());
  for(int i = 1; i <= sets.size(); i++) {
    Relation R = sets[i];
    Relation &S = new_sets[i];    
    assert(R.is_set());
    
    S = Relation(R.n_set());
    S.copy_names(R);
    F_Exists *fe = S.add_exists();
    F_Or *fo = fe->add_or();
    for(DNF_Iterator D(R.query_DNF()); D; D++) {
      F_And *fa = fo->add_and();
      Variable_ID_Tuple &oldlocals = (*D)->locals();
      Section<Variable_ID> newlocals = fe->declare_tuple(oldlocals.size());

      /* copy constraints.  This is much more difficult than it needs
         to be, but add_EQ(Constraint_Handle) doesn't work because it can't
         keep track of existentially quantified varaibles across calls.
         Sigh.  */

      for(EQ_Iterator e(*D); e; e++)
        if((max_fs_arity(*e) > 0) == keep_fs){
          EQ_Handle n = fa->add_EQ();
          for(Constr_Vars_Iter cvi(*e,false);cvi;cvi++)
            if((*cvi).var->kind() == Wildcard_Var)
              n.update_coef(newlocals[oldlocals.index((*cvi).var)],
                            (*cvi).coef);
            else
              if((*cvi).var->kind() == Global_Var)
                n.update_coef(S.get_local((*cvi).var->get_global_var(),
                                          (*cvi).var->function_of()),
                              (*cvi).coef);
              else
                n.update_coef((*cvi).var,(*cvi).coef);
          n.update_const((*e).get_const());
          n.finalize();
        }

      for(GEQ_Iterator g(*D); g; g++)
        if((max_fs_arity(*g) > 0) == keep_fs) {
          GEQ_Handle n = fa->add_GEQ();
          for(Constr_Vars_Iter cvi(*g,false);cvi;cvi++)
            if((*cvi).var->kind() == Wildcard_Var)
              n.update_coef(newlocals[oldlocals.index((*cvi).var)],
                            (*cvi).coef);
            else
              if((*cvi).var->kind() == Global_Var)
                n.update_coef(S.get_local((*cvi).var->get_global_var(),
                                          (*cvi).var->function_of()),
                              (*cvi).coef);
              else
                n.update_coef((*cvi).var,(*cvi).coef);
          n.update_const((*g).get_const());
          n.finalize();
        }
    }
    S.finalize();
  }

  return new_sets;
}


RelTuple strip_function_symbols(SetTuple &sets) {
  return filter_function_symbols(sets,false);
}

RelTuple extract_function_symbols(SetTuple &sets) {
  return filter_function_symbols(sets,true);
}


std::string MMGenerateCode(RelTuple &T, SetTuple &old_IS, Relation &known, int effort) {
  Tuple<CG_outputRepr *> nameInfo;
  for (int stmt = 1; stmt <= T.size(); stmt++)
    nameInfo.append(new CG_stringRepr("s" + to_string(stmt)));

  CG_stringBuilder ocg;
  CG_stringRepr *sRepr = static_cast<CG_stringRepr *>(MMGenerateCode(&ocg, T, old_IS, nameInfo, known, effort));

  for (int i = 1; i <= nameInfo.size(); i++)
    delete nameInfo[i];
  if (sRepr != NULL)
    return GetString(sRepr);
  else
    return std::string();
}


//*****************************************************************************
// MMGenerateCode implementation, added by D people. Lei Zhou, Apr. 24, 96
//*****************************************************************************
CG_outputRepr* MMGenerateCode(CG_outputBuilder* ocg, RelTuple &T, SetTuple &old_IS, const Tuple<CG_outputRepr *> &stmt_content, Relation &known, int effort) {
  int stmts = T.size();
  if (stmts == 0)
    return ocg->CreateComment(1, "No statements found!");
  if (!known.is_null())
    known.simplify();
  
  // prepare iteration spaces by splitting disjoint conjunctions
  int maxStmt = 1;
  last_level = 0;
  for (int stmt = 1; stmt <= stmts; stmt++) {
    int old_dim = T[stmt].n_out();
    if (old_dim > last_level)
      last_level = old_dim;

    for (int i = 1; i <= old_IS[stmt].n_set(); i++)
      T[stmt].name_input_var(i, old_IS[stmt].set_var(i)->name());
    for (int i = 1; i <= old_dim; i++)
      T[stmt].name_output_var(i, std::string("t")+to_string(i));
    T[stmt].setup_names();
    
    Relation R = Range(Restrict_Domain(copy(T[stmt]), copy(old_IS[stmt])));
    R.simplify();
    while(R.is_upper_bound_satisfiable()) {
      new_IS.reallocate(maxStmt);
      transformations.reallocate(maxStmt);
      statementInfo.reallocate(maxStmt);
      DNF *dnf = R.query_DNF();
      DNF_Iterator c(dnf);
      Relation R2 = Relation(R, *c);
      R2.simplify();
      if (R2.is_inexact())
        throw codegen_error("unknown constraint in loop bounds");   
      if (known.is_null()) {
        new_IS[maxStmt] = R2;
        transformations[maxStmt] = T[stmt];
        statementInfo[maxStmt] = stmt_content[stmt];
        maxStmt++;
      }
      else {
        Relation R2_extended = copy(R2);
        Relation known_extended = copy(known);
        if (R2.n_set() > known.n_set())
          known_extended = Extend_Set(known_extended, R2.n_set()-known.n_set());
        else if (R2.n_set() < known.n_set())
          R2_extended = Extend_Set(R2_extended, known.n_set()-R2.n_set());
        if (Intersection(R2_extended, known_extended).is_upper_bound_satisfiable()) {
          new_IS[maxStmt] = R2;
          transformations[maxStmt] = T[stmt];
          statementInfo[maxStmt] = stmt_content[stmt];
          maxStmt++;
        }
      }
      c.next();
      if (!c.live()) 
        break;
      if(code_gen_debug) {
        fprintf(DebugFile, "splitting iteration space for disjoint form\n");
        fprintf(DebugFile, "Original iteration space: \n");
        R.print_with_subs(DebugFile);
        fprintf(DebugFile, "First conjunct: \n");
        R2.print_with_subs(DebugFile);
      }
      Relation remainder(R, *c);
      c.next();
      while (c.live()) {
        remainder = Union(remainder, Relation(R, *c));
        c.next();
      }
      R = Difference(remainder, copy(R2));
      R.simplify();
      if(code_gen_debug) {
        fprintf(DebugFile, "Remaining iteration space: \n");
        R.print_with_subs(DebugFile);
      }
    }
  }

  // reset number of statements
  stmts = maxStmt-1;
  if(stmts == 0)
    return ocg->CreateComment(1, "No points in any of the iteration spaces!");

  // entend iteration spaces to maximum dimension
  for (int stmt = 1; stmt <= stmts; stmt++) {
    int old_dim = new_IS[stmt].n_set();
    if (old_dim < last_level) {
      new_IS[stmt] = Extend_Set(new_IS[stmt], last_level-old_dim);
      F_And *f_root = new_IS[stmt].and_with_and();
      for (int i = old_dim+1; i <= last_level; i++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(new_IS[stmt].set_var(i), 1);
        h.update_const(posInfinity);
      }
    }   
  }
  
  // standarize the known condition
  if(known.is_null()) {
    known = Relation::True(last_level);
  }
  known = Extend_Set(known, last_level-known.n_set());
  for (int i = 1; i <= last_level; i++)
    known.name_set_var(i, std::string("t")+to_string(i));
  known.setup_names();
  
  // prepare projected subspaces for each loop level
  projected_nIS.clear();
  projected_nIS.reallocate(last_level);
  for(int i = 1; i <= last_level; i++ ) {
    projected_nIS[i].reallocate(stmts);
  }
  for (int stmt = 1; stmt <= stmts; stmt++) {
    if (last_level > 0)
      projected_nIS[last_level][stmt] = new_IS[stmt];
    for (int i = last_level-1; i >= 1; i--) {
      projected_nIS[i][stmt] = Project(copy(projected_nIS[i+1][stmt]), i+1, Set_Var);
      projected_nIS[i][stmt].simplify();
    }
  }

  // recursively generate AST
  IntTuple allStmts(stmts);
  for(int i = 1; i <= stmts; i++)
    allStmts[i] = 1;
  CG_result *cg = gen_recursive(1, allStmts); 

  // always force finite bounds
  cg = cg->recompute(known, known);
  cg = cg->force_finite_bounds();

  // loop overhead removal based on actual nesting depth -- by chun 09/17/2008
  for (int i = 1; i <= min(effort, cg->depth()); i++)
    cg = cg->liftOverhead(i);

  // merge adjacent if-conditions -- by chun 10/24/2006
  cg->hoistGuard();

  // really print out the loop
  //CG_outputRepr* sRepr = cg->printRepr(ocg, 1, std::vector<CG_outputRepr *>(last_level, NULL));
  CG_outputRepr* sRepr = cg->printRepr(ocg, 1, std::vector<CG_outputRepr *>(last_level));
  delete cg;
  cg = NULL;
  projected_nIS.clear();
  transformations.clear();
  new_IS.clear();

  return sRepr;
}

//protonu--overload the above MMGenerateCode to take into the CUDA-CHiLL
CG_outputRepr* MMGenerateCode(CG_outputBuilder* ocg, RelTuple &T, SetTuple &old_IS, 
		const Tuple<CG_outputRepr *> &stmt_content, Relation &known,
		Tuple< IntTuple >& smtNonSplitLevels_,
	       	std::vector< std::pair<int, std::string> > syncs_,
	       	const Tuple< Tuple<std::string> >& loopIdxNames_, 
	       	int effort) {
  int stmts = T.size();
  if (stmts == 0)
    return ocg->CreateComment(1, "No statements found!");
  if (!known.is_null())
    known.simplify();

  //protonu-- 
  //easier to handle this as a global
  smtNonSplitLevels = smtNonSplitLevels_;
  syncs = syncs_;
  loopIdxNames = loopIdxNames_; 
  //end-protonu



  // prepare iteration spaces by splitting disjoint conjunctions
  int maxStmt = 1;
  last_level = 0;
  for (int stmt = 1; stmt <= stmts; stmt++) {
    int old_dim = T[stmt].n_out();
    if (old_dim > last_level)
      last_level = old_dim;

    for (int i = 1; i <= old_IS[stmt].n_set(); i++)
      T[stmt].name_input_var(i, old_IS[stmt].set_var(i)->name());
    for (int i = 1; i <= old_dim; i++)
      T[stmt].name_output_var(i, std::string("t")+to_string(i));
    T[stmt].setup_names();
    
    Relation R = Range(Restrict_Domain(copy(T[stmt]), copy(old_IS[stmt])));
    R.simplify();
    while(R.is_upper_bound_satisfiable()) {
      new_IS.reallocate(maxStmt);
      transformations.reallocate(maxStmt);
      statementInfo.reallocate(maxStmt);

      //protonu--putting in fix provided by Mark Hall
      smtNonSplitLevels.reallocate(maxStmt);
      //end-protonu


      DNF *dnf = R.query_DNF();
      DNF_Iterator c(dnf);
      Relation R2 = Relation(R, *c);
      R2.simplify();
      if (R2.is_inexact())
        throw codegen_error("unknown constraint in loop bounds");   
      if (known.is_null()) {
        new_IS[maxStmt] = R2;
        transformations[maxStmt] = T[stmt];
        statementInfo[maxStmt] = stmt_content[stmt];
        maxStmt++;
      }
      else {
        Relation R2_extended = copy(R2);
        Relation known_extended = copy(known);
        if (R2.n_set() > known.n_set())
          known_extended = Extend_Set(known_extended, R2.n_set()-known.n_set());
        else if (R2.n_set() < known.n_set())
          R2_extended = Extend_Set(R2_extended, known.n_set()-R2.n_set());
        if (Intersection(R2_extended, known_extended).is_upper_bound_satisfiable()) {
          new_IS[maxStmt] = R2;
          transformations[maxStmt] = T[stmt];
          statementInfo[maxStmt] = stmt_content[stmt];
          maxStmt++;
        }
      }
      c.next();
      if (!c.live()) 
        break;
      if(code_gen_debug) {
        fprintf(DebugFile, "splitting iteration space for disjoint form\n");
        fprintf(DebugFile, "Original iteration space: \n");
        R.print_with_subs(DebugFile);
        fprintf(DebugFile, "First conjunct: \n");
        R2.print_with_subs(DebugFile);
      }
      Relation remainder(R, *c);
      c.next();
      while (c.live()) {
        remainder = Union(remainder, Relation(R, *c));
        c.next();
      }
      R = Difference(remainder, copy(R2));
      R.simplify();
      if(code_gen_debug) {
        fprintf(DebugFile, "Remaining iteration space: \n");
        R.print_with_subs(DebugFile);
      }
    }
  }

  // reset number of statements
  stmts = maxStmt-1;
  if(stmts == 0)
    return ocg->CreateComment(1, "No points in any of the iteration spaces!");

  // entend iteration spaces to maximum dimension
  for (int stmt = 1; stmt <= stmts; stmt++) {
    int old_dim = new_IS[stmt].n_set();
    if (old_dim < last_level) {
      new_IS[stmt] = Extend_Set(new_IS[stmt], last_level-old_dim);
      F_And *f_root = new_IS[stmt].and_with_and();
      for (int i = old_dim+1; i <= last_level; i++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(new_IS[stmt].set_var(i), 1);
        h.update_const(posInfinity);
      }
    }   
  }
  
  // standarize the known condition
  if(known.is_null()) {
    known = Relation::True(last_level);
  }
  known = Extend_Set(known, last_level-known.n_set());
  for (int i = 1; i <= last_level; i++)
    known.name_set_var(i, std::string("t")+to_string(i));
  known.setup_names();
  
  // prepare projected subspaces for each loop level
  projected_nIS.clear();
  projected_nIS.reallocate(last_level);
  for(int i = 1; i <= last_level; i++ ) {
    projected_nIS[i].reallocate(stmts);
  }
  for (int stmt = 1; stmt <= stmts; stmt++) {
    if (last_level > 0)
      projected_nIS[last_level][stmt] = new_IS[stmt];
    for (int i = last_level-1; i >= 1; i--) {
      projected_nIS[i][stmt] = Project(copy(projected_nIS[i+1][stmt]), i+1, Set_Var);
      projected_nIS[i][stmt].simplify();
    }
  }

  // recursively generate AST
  IntTuple allStmts(stmts);
  for(int i = 1; i <= stmts; i++)
    allStmts[i] = 1;
  CG_result *cg = gen_recursive(1, allStmts); 

  // always force finite bounds
  cg = cg->recompute(known, known);
  cg = cg->force_finite_bounds();

  // loop overhead removal based on actual nesting depth -- by chun 09/17/2008
  for (int i = 1; i <= min(effort, cg->depth()); i++)
    cg = cg->liftOverhead(i);

  // merge adjacent if-conditions -- by chun 10/24/2006
  cg->hoistGuard();

  // really print out the loop
  //CG_outputRepr* sRepr = cg->printRepr(ocg, 1, std::vector<CG_outputRepr *>(last_level, NULL));
  CG_outputRepr* sRepr = cg->printRepr(ocg, 1, std::vector<CG_outputRepr *>(last_level ));
  delete cg;
  cg = NULL;
  projected_nIS.clear();
  transformations.clear();
  new_IS.clear();

  return sRepr;
}

CG_result *gen_recursive(int level, IntTuple &isActive) {
  int stmts = isActive.size();

  Set<int> active;
  int s;
  for(s = 1; s <= stmts; s++)
    if(isActive[s]) active.insert(s);

  assert (active.size() >= 1);
  if(level > last_level) return new CG_leaf(isActive);

  if (active.size() == 1)
    return new CG_loop(isActive,level, gen_recursive(level+1,isActive));

  bool constantLevel = true;
   
  int test_rel_size;
  coef_t start,finish; 
  finish = -(posInfinity-1); // -(MAXINT-1);
  start = posInfinity;     // MAXINT;
  Tuple<coef_t> when(stmts);
  for(s=1; s<=stmts; s++) if (isActive[s]) {
      coef_t lb,ub;
      test_rel_size = projected_nIS[level][s].n_set();
      projected_nIS[level][s].single_conjunct()
        ->query_variable_bounds(
          projected_nIS[level][s].set_var(level),
          lb,ub);
      if(code_gen_debug) {
        fprintf(DebugFile, "IS%d:  " coef_fmt " <= t%d <= " coef_fmt "\n",s,
                lb,level,ub);
        projected_nIS[level][s].prefix_print(DebugFile);
      }
      if (lb != ub) {
        constantLevel = false;
        break;
      }
      else {
        set_max(finish,lb);
        set_min(start,lb);
        when[s] = lb;
      }
    }

 
  if (constantLevel && finish-start <= stmts) {
    IntTuple newActive(isActive.size());
    for(int i=1; i<=stmts; i++)  
      newActive[i] = isActive[i] && when[i] == start;
    CG_result *r  = new CG_loop(isActive,level, 
                                gen_recursive(level+1,newActive));
    for(coef_t time = start+1; time <= finish; time++) {
      int count = 0;
      for(int i=1; i<=stmts; i++)   {
        newActive[i] = isActive[i] && when[i] == time;
        if (newActive[i]) count++;
      }
      if (count) {
        Relation test_rel(test_rel_size);
        GEQ_Handle g = test_rel.and_with_GEQ(); 
        g.update_coef(test_rel.set_var(level),-1);
        g.update_const(time-1);
   
        r = new CG_split(isActive,level,test_rel,r,
                         new CG_loop(isActive,level, 
                                     gen_recursive(level+1,newActive)));
      }  
    }
    return r;
  }
  
// Since the Hull computation is approximate, we will get regions that
// have no stmts.  (since we will have split on constraints on the
// hull, and thus we are looking at a region outside the convex hull
// of all the iteration spaces.)
#if 1
  Relation hull = Hull(projected_nIS[level],isActive,1);
#else
  Relation hull = Hull(projected_nIS[level],isActive,0);
#endif

  if(code_gen_debug) {
    fprintf(DebugFile, "Hull (level %d) is:\n",level);
    hull.prefix_print(DebugFile);
  }

  
  IntTuple firstChunk(isActive);
  IntTuple secondChunk(isActive);

  //protonu-warn Chun about this change
  //This does some fancy splitting of statements into loops with the
  //fewest dimentions, but that's not necessarily what we want when
  //code-gening for CUDA. smtNonSplitLevels keeps track per-statment of
  //the levels that should not be split on.
  bool checkForSplits = true;
  for (int s = 1; s <= isActive.size(); s++){
    if (isActive[s]) {
      if(s < smtNonSplitLevels.size() && smtNonSplitLevels[s].index(level-2) != 0){
        checkForSplits = false;
        break;
      }
    }
  }

  //protonu-modifying the next for loop
  for (int s = 1; checkForSplits && s <= isActive.size(); s++)
    if (isActive[s]) {
      Relation gist = Gist(copy(projected_nIS[level][s]),copy(hull),1);
      if (gist.is_obvious_tautology()) break;
      gist.simplify();
      Conjunct *s_conj = gist.single_conjunct();
      for(GEQ_Iterator G(s_conj); G; G++) {
        Relation test_rel(gist.n_set());
        test_rel.and_with_GEQ(*G);
        Variable_ID v = set_var(level);
        coef_t sign = (*G).get_coef(v);
        if(sign > 0) test_rel = Complement(test_rel);
        if(code_gen_debug) {
          fprintf(DebugFile, "Considering split from stmt %d:\n",s);
          test_rel.prefix_print(DebugFile);
        }
  
        firstChunk[s] = sign <= 0;
        secondChunk[s] = sign > 0;
        int numberFirst = sign <= 0;
        int numberSecond = sign > 0;
        
        for (int s2 = 1; s2 <= isActive.size(); s2++)
          if (isActive[s2] && s2 != s) {
            if(code_gen_debug) 
              fprintf(DebugFile,"Consider stmt %d\n",s2);
            bool t = Intersection(copy(projected_nIS[level][s2]),
                                  copy(test_rel)).is_upper_bound_satisfiable();
            bool f = Difference(copy(projected_nIS[level][s2]),
                                copy(test_rel)).is_upper_bound_satisfiable();
            assert(t || f);
            if(code_gen_debug  && t&&f) 
              fprintf(DebugFile, "Slashes stmt %d\n",s2);
            if (t&&f) goto nextGEQ;
            if(code_gen_debug) {
              if (t)
                fprintf(DebugFile, "true for stmt %d\n",s2);
              else 
                fprintf(DebugFile, "false for stmt %d\n",s2);
            }
            if (t) numberFirst++;
            else numberSecond++;
            firstChunk[s2] = t;
            secondChunk[s2] = !t;
          }
            
        assert(numberFirst+numberSecond>1 && "Can't handle wildcard in iteration space");
        if(code_gen_debug) 
          fprintf(DebugFile, "%d true, %d false\n",
                  numberFirst,
                  numberSecond);
        if (numberFirst && numberSecond) {
          // Found a dividing constraint
          return new CG_split(isActive,level,test_rel,
                              gen_recursive(level,firstChunk),
                              gen_recursive(level,secondChunk));
        }
      nextGEQ: ;
      }
    }

  // No way found to divide stmts without splitting, generate loop

  return new CG_loop(isActive,level, gen_recursive(level+1,isActive));
}

}
