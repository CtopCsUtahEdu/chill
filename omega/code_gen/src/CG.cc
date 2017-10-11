
/*****************************************************************************

 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
 CG node classes, used to build AST tree from polyhedra scanning.

 Notes:
 Parameter "restriction" is always tighter than "known" since CG_split
 node does not correspond to any code for enforcement. This property is
 destroyed after hoistGuard since "restriction" is not used anymore.
 CG node's children are guaranteed not to be NULL, either NULL child is
 removed from the children or the parent node itself becomes NULL.

 History:
 04/20/96 printRepr added by D people. Lei Zhou
 10/24/06 hoistGuard added by chun
 08/03/10 collect CG classes into one place, by Chun Chen
 08/04/10 track dynamically substituted variables in printRepr, by chun
 04/02/11 rewrite the CG node classes, by chun
*****************************************************************************/

#include <typeinfo>
#include <assert.h>
#include <omega.h>
#include <code_gen/codegen.h>
#include <code_gen/CG.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_stringBuilder.h>
#include <code_gen/CG_utils.h>
#include <code_gen/codegen_error.h>
#include <stack>
#include <string.h>

namespace omega {
  
  extern std::vector<std::vector<int> > smtNonSplitLevels;
  extern std::vector<std::vector<std::string> > loopIdxNames; //per stmt
  extern std::vector<std::pair<int, std::string> > syncs;
  
  extern int checkLoopLevel;
  extern int stmtForLoopCheck;
  extern int upperBoundForLevel;
  extern int lowerBoundForLevel;
  extern bool fillInBounds;
  
  //-----------------------------------------------------------------------------
  // Class: CG_result
  //-----------------------------------------------------------------------------
  
  CG_outputRepr *CG_result::printRepr(CG_outputBuilder *ocg,
                                      const std::vector<CG_outputRepr *> &stmts,
                                      std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > uninterpreted_symbols, 
                                      bool printString) const {
    debug_fprintf(stderr, "\nCG_result::printRepr(ocg, stmts) \n"); 
    //Anand: making a tweak to allocate twice the original number of dynamically allocated variables
    //for use with Uninterpreted function symbols
    
    //Anand: adding support for Replacing substituted variables within
    //Uninterpreted function symbols or global variables with arity > 0 here
    //--begin
    
    // check for an error that happened once
    int num_unin = uninterpreted_symbols.size();
    int num_active =  active_.size();
    if (num_unin < num_active)
      throw std::runtime_error(std::string("CG_result::printRepr(), not enough uninterpreted symbols for active statement"));

    std::vector<std::pair<CG_outputRepr *, int> > aotf = std::vector<
      std::pair<CG_outputRepr *, int> >(2 * num_level(),
                                        std::make_pair(static_cast<CG_outputRepr *>(NULL), 0));

#define DYINGHERE 
#ifdef DYINGHERE
    int num_levels = num_level();
    
    for (int s = 0; s < active_.size(); s++) {
      debug_fprintf(stderr, "\ns %d\n", s); 
      std::vector<std::string> loop_vars;
      if (active_.get(s)) {
        
        Relation mapping = Inverse(
          copy((codegen_->xforms_[codegen_->remap_[s]])));
        
        mapping.simplify();
        mapping.setup_names();
        
        for (int i = 1; i <= mapping.n_out(); i++)
          loop_vars.push_back(mapping.output_var(i)->name());
        
        std::vector<CG_outputRepr *> subs_;
        for (int i = 1; i <= mapping.n_out(); i++) {
          Relation mapping1(mapping.n_out(), 1);
          F_And *f_root = mapping1.add_and();
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(mapping1.output_var(1), 1);
          h.update_coef(mapping1.input_var(i), -1);
          Relation r = Composition(mapping1, copy(mapping));
          r.simplify();
          
          //Relation r = copy(mapping);
          
          Variable_ID v = r.output_var(1);
          loop_vars.push_back(mapping.output_var(i)->name());
          
          std::pair<EQ_Handle, int> result = find_simplest_assignment(r,
                                                  v, aotf);
          
          std::string hand = result.first.print_to_string();
          //debug_fprintf(stderr, "result: %s, %d\n", hand.c_str(), result.second); 
          if (result.second < INT_MAX) {
            
            CG_outputRepr *subs = output_substitution_repr(ocg,
                                                           result.first, 
                                                           v, 
                                                           true, 
                                                           r, 
                                                           aotf,
                                                           uninterpreted_symbols[s]);
            
            subs_.push_back(subs->clone());
            
            aotf[num_levels + i - 1] = std::make_pair(subs, 999);
            
          } else {
            CG_outputRepr* repr = NULL;
            
            aotf[num_levels + i - 1] = std::make_pair(repr, 999);
          }
        }
        if(!printString)
          for (std::map<std::string, std::vector<CG_outputRepr *> >::iterator it =
                 uninterpreted_symbols[s].begin();
               it != uninterpreted_symbols[s].end(); it++) {
            std::vector<CG_outputRepr *> reprs_ = it->second;
            std::vector<CG_outputRepr *> reprs_2;
            
            for (int k = 0; k < reprs_.size(); k++) {
              std::vector<CG_outputRepr *> subs2;
              for (int l = 0; l < subs_.size(); l++) {
                
                subs2.push_back(subs_[l]->clone());
              }
              CG_outputRepr * temp =
                ocg->CreateSubstitutedStmt(0, reprs_[k]->clone(),
                                           loop_vars, subs2, false);
              
              if (temp != NULL)
                reprs_2.push_back(temp);
              //    reprs_2.push_back(subs_[k]->clone());
              
            }
            if(reprs_2.size() > 0)
              it->second = reprs_2;
          }
        
        //break;
      }
    }
    
//--end
    
#endif

    debug_fprintf(stderr, "\n\n\n\nprintRepr recursing ??? return printRepr( ... )\n"); 
    return printRepr(1, ocg, stmts, aotf, uninterpreted_symbols, printString);
  }
  


  std::string CG_result::printString(
    std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > uninterpreted_symbols) const {

    debug_fprintf(stderr, "CG.cc line 164, CG_result::printString()\n"); 
    CG_stringBuilder ocg;
    std::vector<CG_outputRepr *> stmts(codegen_->xforms_.size());

    debug_fprintf(stderr, "stmts.size() %d\n", stmts.size()); 
    for (int i = 0; i < stmts.size(); i++)
      stmts[i] = new CG_stringRepr("s" + to_string(i));

    CG_stringRepr *repr = static_cast<CG_stringRepr *>(printRepr(&ocg, 
                                                                 stmts,
                                                                 uninterpreted_symbols, 
                                                                 true));

    for (int i = 0; i < stmts.size(); i++)
      delete stmts[i];
    
    if (repr != NULL) {
      std::string s = repr->GetString();
      //debug_fprintf(stderr, "\nCG.cc L197 repr->GetString() = '%s'\n\n\n", s.c_str()); 
      delete repr;
      return s;
    } else
      return std::string();
  }
  
  int CG_result::num_level() const {
    return codegen_->num_level();
  }
  
  //-----------------------------------------------------------------------------
  // Class: CG_split
  //-----------------------------------------------------------------------------
  
  CG_result *CG_split::recompute(const BoolSet<> &parent_active,
                                 const Relation &known, const Relation &restriction) {
    active_ &= parent_active;
    if (active_.empty()) {
      delete this;
      return NULL;
    }
    
    
    int i = 0;
    while (i < restrictions_.size()) {
      Relation new_restriction = Intersection(copy(restrictions_[i]),
                                              copy(restriction));
      
      new_restriction.simplify(2, 4);
      //new_restriction.simplify();
      clauses_[i] = clauses_[i]->recompute(active_, copy(known),
                                           new_restriction);
      if (clauses_[i] == NULL) {
        restrictions_.erase(restrictions_.begin() + i);
        clauses_.erase(clauses_.begin() + i);
      } else
        i++;
    }
    
    
    if (restrictions_.size() == 0) {
      delete this;
      return NULL;
    } else
      return this;
  }
  
  int CG_split::populateDepth() {
    int max_depth = 0;
    for (int i = 0; i < clauses_.size(); i++) {
      int t = clauses_[i]->populateDepth();
      if (t > max_depth)
        max_depth = t;
    }
    return max_depth;
  }
  
  std::pair<CG_result *, Relation> CG_split::liftOverhead(int depth,
                                                          bool propagate_up) {
    for (int i = 0; i < clauses_.size();) {
      std::pair<CG_result *, Relation> result = clauses_[i]->liftOverhead(
                                                                          depth, propagate_up);
      if (result.first == NULL)
        clauses_.erase(clauses_.begin() + i);
      else {
        clauses_[i] = result.first;
        if (!result.second.is_obvious_tautology())
          return std::make_pair(this, result.second);
        i++;
      }
      
    }
    
    if (clauses_.size() == 0) {
      delete this;
      return std::make_pair(static_cast<CG_result *>(NULL),
                            Relation::True(num_level()));
    } else
      return std::make_pair(this, Relation::True(num_level()));
  }
  
  Relation CG_split::hoistGuard() {
    std::vector<Relation> guards;
    for (int i = 0; i < clauses_.size(); i++)
      guards.push_back(clauses_[i]->hoistGuard());
    
    return SimpleHull(guards, true, true);
  }
  
  void CG_split::removeGuard(const Relation &guard) {
    for (int i = 0; i < clauses_.size(); i++)
      clauses_[i]->removeGuard(guard);
  }
  
  std::vector<CG_result *> CG_split::findNextLevel() const {
    std::vector<CG_result *> result;
    for (int i = 0; i < clauses_.size(); i++) {
      CG_split *splt = dynamic_cast<CG_split *>(clauses_[i]);
      if (splt != NULL) {
        std::vector<CG_result *> t = splt->findNextLevel();
        result.insert(result.end(), t.begin(), t.end());
      } else
        result.push_back(clauses_[i]);
    }
    
    return result;
  }
  



  CG_outputRepr *CG_split::printRepr(int indent, 
                                     CG_outputBuilder *ocg,
                                     const std::vector<CG_outputRepr *> &stmts,
                                     const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                     std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, 
                                     bool printString) const {
    
    debug_fprintf(stderr, "CG_split::printRepr()\n"); 
    int numfly =  assigned_on_the_fly.size();
    //debug_fprintf(stderr, "assigned on the fly  %d\n", numfly );
    //for (int i=0; i<numfly; i++) { 
    //  debug_fprintf(stderr, "i %d\n", i); 
    //  std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
    //  CG_outputRepr *tr = NULL;
    //  if (p.first != NULL) tr = p.first->clone();
    //  int val = p.second;
    //  debug_fprintf(stderr, "0x%x   %d\n", tr, val);
    //} 
    
    CG_outputRepr *stmtList = NULL;
    std::vector<CG_result *> next_level = findNextLevel();
    
    std::vector<CG_loop *> cur_loops;
    for (int i = 0; i < next_level.size(); i++) {
      CG_loop *lp = dynamic_cast<CG_loop *>(next_level[i]);
      if (lp != NULL) {
        cur_loops.push_back(lp);
      } else {
        stmtList = ocg->StmtListAppend(stmtList,
                                       loop_print_repr(active_, 
                                                       cur_loops, 
                                                       0, 
                                                       cur_loops.size(),
                                                       Relation::True(num_level()), 
                                                       NULL, 
                                                       indent, 
                                                       codegen_->remap_, 
                                                       codegen_->xforms_, 
                                                       ocg,
                                                       stmts, 
                                                       assigned_on_the_fly,
                                                       unin));
        stmtList = ocg->StmtListAppend(stmtList,
                                       next_level[i]->printRepr(indent, 
                                                                ocg, 
                                                                stmts,
                                                                assigned_on_the_fly,
                                                                unin, 
                                                                printString));
        cur_loops.clear();
      }
    }
    
    stmtList = ocg->StmtListAppend(stmtList,
                                   loop_print_repr(active_,
                                                   cur_loops, 
                                                   0, 
                                                   cur_loops.size(),
                                                   Relation::True(num_level()), 
                                                   NULL, 
                                                   indent, 
                                                   codegen_->remap_,
                                                   codegen_->xforms_, 
                                                   ocg, 
                                                   stmts,
                                                   assigned_on_the_fly,
                                                   unin));
    return stmtList;
  }
  
  CG_result *CG_split::clone() const {
    //debug_fprintf(stderr, "CG_split::clone()\n"); 
    std::vector<CG_result *> clauses(clauses_.size());
    for (int i = 0; i < clauses_.size(); i++)
      clauses[i] = clauses_[i]->clone();
    return new CG_split(codegen_, active_, restrictions_, clauses);
  }
  
  void CG_split::dump(int indent) const {
    std::string prefix;
    for (int i = 0; i < indent; i++)
      prefix += "  ";
    std::cout << prefix << "SPLIT: " << active_ << std::endl;
    for (int i = 0; i < restrictions_.size(); i++) {
      std::cout << prefix << "restriction: ";
      const_cast<CG_split *>(this)->restrictions_[i].print();
      clauses_[i]->dump(indent + 1);
    }
    
  }
  
  //-----------------------------------------------------------------------------
  // Class: CG_loop
  //-----------------------------------------------------------------------------
  
  CG_result *CG_loop::recompute(const BoolSet<> &parent_active,
                                const Relation &known, const Relation &restriction) {
    known_ = copy(known);
    restriction_ = copy(restriction);
    active_ &= parent_active;
    
    std::vector<Relation> Rs;
    for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
      Relation r = Intersection(copy(restriction),
                                copy(codegen_->projected_IS_[level_ - 1][*i]));
      
      //r.simplify(2, 4);
      r.simplify();
      if (!r.is_upper_bound_satisfiable()) {
        active_.unset(*i);
        continue;
      }
      Rs.push_back(copy(r));
    }
    
    if (active_.empty()) {
      delete this;
      return NULL;
    }
    
    Relation hull = SimpleHull(Rs, true, true);
    
    //hull.simplify(2,4);
    //Anand:Variables for inspector constraint check
    bool has_insp = false;
    bool found_insp = false;
    
    //Global_Var_ID global_insp;
    //Argument_Tuple arg;
    // check if actual loop is needed
    std::pair<EQ_Handle, int> result = find_simplest_assignment(hull,
                                                                hull.set_var(level_));
    if (result.second < INT_MAX) {
      needLoop_ = false;
      
      bounds_ = Relation(hull.n_set());
      F_Exists *f_exists = bounds_.add_and()->add_exists();
      F_And *f_root = f_exists->add_and();
      std::map<Variable_ID, Variable_ID> exists_mapping;
      EQ_Handle h = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          h.update_coef(bounds_.input_var(v->get_position()),
                        cvi.curr_coef());
          break;
        case Wildcard_Var: {
          Variable_ID v2 = replicate_floor_definition(hull, v, bounds_,
                                                      f_exists, f_root, exists_mapping);
          h.update_coef(v2, cvi.curr_coef());
          break;
        }
        case Global_Var: {
          Global_Var_ID g = v->get_global_var();
          Variable_ID v2;
          if (g->arity() == 0)
            v2 = bounds_.get_local(g);
          else
            v2 = bounds_.get_local(g, v->function_of());
          h.update_coef(v2, cvi.curr_coef());
          break;
        }
        default:
          assert(false);
        }
      }
      h.update_const(result.first.get_const());
      bounds_.simplify();
    }
    // loop iterates more than once, extract bounds now
    else {
      debug_fprintf(stderr, "loop iterates more than once, extract bounds now\n"); 
      needLoop_ = true;
      
      bounds_ = Relation(hull.n_set());
      F_Exists *f_exists = bounds_.add_and()->add_exists();
      F_And *f_root = f_exists->add_and();
      std::map<Variable_ID, Variable_ID> exists_mapping;
      
      Relation b = Gist(copy(hull), copy(known), 1);
      bool has_unresolved_bound = false;
      
      std::set<Variable_ID> excluded_floor_vars;
      excluded_floor_vars.insert(b.set_var(level_));
      for (GEQ_Iterator e(b.single_conjunct()->GEQs()); e; e++)
        if ((*e).get_coef(b.set_var(level_)) != 0) {
          bool is_bound = true;
          for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
            std::pair<bool, GEQ_Handle> result = find_floor_definition(
                                                                       b, cvi.curr_var(), excluded_floor_vars);
            if (!result.first) {
              is_bound = false;
              has_unresolved_bound = true;
              //break;
            }
            
            if (!is_bound) {
              break;
            }
          }
          
          if (!is_bound)
            continue;
          
          GEQ_Handle h = f_root->add_GEQ();
          for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var:
              h.update_coef(bounds_.input_var(v->get_position()),
                            cvi.curr_coef());
              break;
            case Wildcard_Var: {
              Variable_ID v2 = replicate_floor_definition(b, v,
                                                          bounds_, f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            case Global_Var: {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = bounds_.get_local(g);
              else
                v2 = bounds_.get_local(g, v->function_of());
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            default:
              assert(false);
            }
          }
          h.update_const((*e).get_const());
        }
      
      if (has_unresolved_bound) {
        b = Approximate(b);
        b.simplify(2, 4);
        //Simplification of Hull
        hull = Approximate(hull);
        hull.simplify(2, 4);
        //end : Anand
        for (GEQ_Iterator e(b.single_conjunct()->GEQs()); e; e++)
          if ((*e).get_coef(b.set_var(level_)) != 0)
            f_root->add_GEQ(*e);
      }
      bounds_.simplify();
      hull.simplify(2,4);
      // Since current SimpleHull does not support max() upper bound or min() lower bound,
      // we have to forcefully split the loop when hull approximation does not return any bound.
      bool has_lb = false;
      bool has_ub = false;
      for (GEQ_Iterator e = bounds_.single_conjunct()->GEQs(); e; e++) {
        if ((*e).get_coef(bounds_.set_var(level_)) > 0)
          has_lb = true;
        else if ((*e).get_coef(bounds_.set_var(level_)) < 0)
          has_ub = true;
        if (has_lb && has_ub)
          break;
      }
      
      if (!has_lb) {
        for (int i = 0; i < Rs.size(); i++) {
          Relation r = Approximate(copy(Rs[i]));
          r.simplify(2, 4);
          for (GEQ_Iterator e = r.single_conjunct()->GEQs(); e; e++)
            if ((*e).get_coef(r.input_var(level_)) > 0) {
              Relation r2 = Relation::True(num_level());
              r2.and_with_GEQ(*e);
              r2.simplify();
              std::vector<Relation> restrictions(2);
              restrictions[0] = Complement(copy(r2));
              restrictions[0].simplify();
              restrictions[1] = r2;
              std::vector<CG_result *> clauses(2);
              clauses[0] = this;
              clauses[1] = this->clone();
              CG_result *cgr = new CG_split(codegen_, active_,
                                            restrictions, clauses);
              cgr = cgr->recompute(active_, copy(known),
                                   copy(restriction));
              return cgr;
            }
        }
        for (int i = 0; i < Rs.size(); i++) {
          Relation r = Approximate(copy(Rs[i]));
          r.simplify(2, 4);
          for (EQ_Iterator e = r.single_conjunct()->EQs(); e; e++)
            if ((*e).get_coef(r.input_var(level_)) != 0) {
              Relation r2 = Relation::True(num_level());
              r2.and_with_GEQ(*e);
              r2.simplify();
              std::vector<Relation> restrictions(2);
              if ((*e).get_coef(r.input_var(level_)) > 0) {
                restrictions[0] = Complement(copy(r2));
                restrictions[0].simplify();
                restrictions[1] = r2;
              } else {
                restrictions[0] = r2;
                restrictions[1] = Complement(copy(r2));
                restrictions[1].simplify();
              }
              std::vector<CG_result *> clauses(2);
              clauses[0] = this;
              clauses[1] = this->clone();
              CG_result *cgr = new CG_split(codegen_, active_,
                                            restrictions, clauses);
              cgr = cgr->recompute(active_, copy(known),
                                   copy(restriction));
              return cgr;
            }
        }
      } else if (!has_ub) {
        for (int i = 0; i < Rs.size(); i++) {
          Relation r = Approximate(copy(Rs[i]));
          r.simplify(2, 4);
          for (GEQ_Iterator e = r.single_conjunct()->GEQs(); e; e++)
            if ((*e).get_coef(r.input_var(level_)) < 0) {
              Relation r2 = Relation::True(num_level());
              r2.and_with_GEQ(*e);
              r2.simplify();
              std::vector<Relation> restrictions(2);
              restrictions[1] = Complement(copy(r2));
              restrictions[1].simplify();
              restrictions[0] = r2;
              std::vector<CG_result *> clauses(2);
              clauses[0] = this;
              clauses[1] = this->clone();
              CG_result *cgr = new CG_split(codegen_, active_,
                                            restrictions, clauses);
              cgr = cgr->recompute(active_, copy(known),
                                   copy(restriction));
              return cgr;
            }
        }
        for (int i = 0; i < Rs.size(); i++) {
          Relation r = Approximate(copy(Rs[i]));
          r.simplify(2, 4);
          for (EQ_Iterator e = r.single_conjunct()->EQs(); e; e++)
            if ((*e).get_coef(r.input_var(level_)) != 0) {
              Relation r2 = Relation::True(num_level());
              r2.and_with_GEQ(*e);
              r2.simplify();
              std::vector<Relation> restrictions(2);
              if ((*e).get_coef(r.input_var(level_)) > 0) {
                restrictions[0] = Complement(copy(r2));
                restrictions[0].simplify();
                restrictions[1] = r2;
              } else {
                restrictions[0] = r2;
                restrictions[1] = Complement(copy(r2));
                restrictions[1].simplify();
              }
              std::vector<CG_result *> clauses(2);
              clauses[0] = this;
              clauses[1] = this->clone();
              CG_result *cgr = new CG_split(codegen_, active_,
                                            restrictions, clauses);
              cgr = cgr->recompute(active_, copy(known),
                                   copy(restriction));
              return cgr;
            }
        }
      }
      
      if (!has_lb && !has_ub)
        throw codegen_error(
                            "can't find any bound at loop level " + to_string(level_));
      else if (!has_lb)
        throw codegen_error(
                            "can't find lower bound at loop level "
                            + to_string(level_));
      else if (!has_ub)
        throw codegen_error(
                            "can't find upper bound at loop level "
                            + to_string(level_));
    }
    bounds_.copy_names(hull);
    bounds_.setup_names();
    
    // additional guard/stride condition extraction
    if (needLoop_) {
      Relation cur_known = Intersection(copy(bounds_), copy(known_));
      cur_known.simplify();
      hull = Gist(hull, copy(cur_known), 1);
      
      std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(hull,
                                                                      hull.set_var(level_));
      if (result.second != NULL)
        if (abs(result.first.get_coef(hull.set_var(level_))) == 1) {
          F_Exists *f_exists = bounds_.and_with_and()->add_exists();
          F_And *f_root = f_exists->add_and();
          std::map<Variable_ID, Variable_ID> exists_mapping;
          EQ_Handle h = f_root->add_EQ();
          for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var:
              h.update_coef(bounds_.input_var(v->get_position()),
                            cvi.curr_coef());
              break;
            case Wildcard_Var: {
              Variable_ID v2;
              if (v == result.second)
                v2 = f_exists->declare();
              else
                v2 = replicate_floor_definition(hull, v, bounds_,
                                                f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            case Global_Var: {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = bounds_.get_local(g);
              else
                v2 = bounds_.get_local(g, v->function_of());
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            default:
              assert(false);
            }
          }
          h.update_const(result.first.get_const());
        } else {
          // since gist is not powerful enough on modular constraints for now,
          // make an educated guess
          coef_t stride = abs(result.first.get_coef(result.second))
            / gcd(abs(result.first.get_coef(result.second)),
                  abs(
                      result.first.get_coef(
                                            hull.set_var(level_))));
          
          Relation r1(hull.n_inp());
          F_Exists *f_exists = r1.add_and()->add_exists();
          F_And *f_root = f_exists->add_and();
          std::map<Variable_ID, Variable_ID> exists_mapping;
          EQ_Handle h = f_root->add_EQ();
          for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var:
              h.update_coef(r1.input_var(v->get_position()),
                            cvi.curr_coef());
              break;
            case Wildcard_Var: {
              Variable_ID v2;
              if (v == result.second)
                v2 = f_exists->declare();
              else
                v2 = replicate_floor_definition(hull, v, r1,
                                                f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            case Global_Var: {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = r1.get_local(g);
              else
                v2 = r1.get_local(g, v->function_of());
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            default:
              assert(false);
            }
          }
          h.update_const(result.first.get_const());
          r1.simplify();
          
          bool guess_success = false;
          for (GEQ_Iterator e(bounds_.single_conjunct()->GEQs()); e; e++)
            if ((*e).get_coef(bounds_.set_var(level_)) == 1) {
              Relation r2(hull.n_inp());
              F_Exists *f_exists = r2.add_and()->add_exists();
              F_And *f_root = f_exists->add_and();
              std::map<Variable_ID, Variable_ID> exists_mapping;
              EQ_Handle h = f_root->add_EQ();
              h.update_coef(f_exists->declare(), stride);
              for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
                Variable_ID v = cvi.curr_var();
                switch (v->kind()) {
                case Input_Var:
                  h.update_coef(r2.input_var(v->get_position()),
                                cvi.curr_coef());
                  break;
                case Wildcard_Var: {
                  Variable_ID v2 = replicate_floor_definition(
                                                              hull, v, r2, f_exists, f_root,
                                                              exists_mapping);
                  h.update_coef(v2, cvi.curr_coef());
                  break;
                }
                case Global_Var: {
                  Global_Var_ID g = v->get_global_var();
                  Variable_ID v2;
                  if (g->arity() == 0)
                    v2 = r2.get_local(g);
                  else
                    v2 = r2.get_local(g, v->function_of());
                  h.update_coef(v2, cvi.curr_coef());
                  break;
                }
                default:
                  assert(false);
                }
              }
              h.update_const((*e).get_const());
              r2.simplify();
              
              if (Gist(copy(r1),
                       Intersection(copy(cur_known), copy(r2)), 1).is_obvious_tautology()
                  && Gist(copy(r2),
                          Intersection(copy(cur_known), copy(r1)),
                          1).is_obvious_tautology()) {
                bounds_ = Intersection(bounds_, r2);
                bounds_.simplify();
                guess_success = true;
                break;
              }
            }
          
          // this is really a stride with non-unit coefficient for this loop variable
          if (!guess_success) {
            // TODO: for stride ax = b mod n it might be beneficial to
            //       generate modular linear equation solver code for
            //       runtime to get the starting position in printRepr,
            //       and stride would be n/gcd(|a|,n), thus this stride
            //       can be put into bounds_ too.
          }
          
        }
      
      hull = Project(hull, hull.set_var(level_));
      hull.simplify(2, 4);
      guard_ = Gist(hull, Intersection(copy(bounds_), copy(known_)), 1);
    }
    // don't generate guard for non-actual loop, postpone it. otherwise
    // redundant if-conditions might be generated since for-loop semantics
    // includes implicit comparison checking.  -- by chun 09/14/10
    else
      guard_ = Relation::True(num_level());
    guard_.copy_names(bounds_);
    guard_.setup_names();
    
    //guard_.simplify();  
    // recursively down the AST
    Relation new_known = Intersection(copy(known),
                                      Intersection(copy(bounds_), copy(guard_)));
    //Anand: IF inspector present in equality and GEQ add the equality constraint to known
    //So as to avoid unnecessary IF condition
    new_known.simplify(2, 4);
    Relation new_restriction = Intersection(copy(restriction),
                                            Intersection(copy(bounds_), copy(guard_)));
    new_restriction.simplify(2, 4);
    body_ = body_->recompute(active_, new_known, new_restriction);
    if (body_ == NULL) {
      delete this;
      return NULL;
    } else
      return this;
  }
  
  int CG_loop::populateDepth() {
    int depth = body_->populateDepth();
    if (needLoop_)
      depth_ = depth + 1;
    else
      depth_ = depth;
    return depth_;
  }
  
  std::pair<CG_result *, Relation> CG_loop::liftOverhead(int depth,
                                                         bool propagate_up) {
    if (depth_ > depth) {
      assert(propagate_up == false);
      std::pair<CG_result *, Relation> result = body_->liftOverhead(depth,
                                                                    false);
      body_ = result.first;
      return std::make_pair(this, Relation::True(num_level()));
    } else { // (depth_ <= depth)
      if (propagate_up) {
        Relation r = pick_one_guard(guard_, level_);
        if (!r.is_obvious_tautology())
          return std::make_pair(this, r);
      }
      
      std::pair<CG_result *, Relation> result;
      if (propagate_up || needLoop_)
        result = body_->liftOverhead(depth, true);
      else
        result = body_->liftOverhead(depth, false);
      body_ = result.first;
      if (result.second.is_obvious_tautology())
        return std::make_pair(this, result.second);
      
      // loop is an assignment, replace this loop variable in overhead condition
      if (!needLoop_) {
        result.second = Intersection(result.second, copy(bounds_));
        result.second = Project(result.second,
                                result.second.set_var(level_));
        result.second.simplify(2, 4);
      }
      
      int max_level = 0;
      bool has_wildcard = false;
      bool direction = true;
      for (EQ_Iterator e(result.second.single_conjunct()->EQs()); e; e++)
        if ((*e).has_wildcards()) {
          if (has_wildcard)
            assert(false);
          else
            has_wildcard = true;
          for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
            if (cvi.curr_var()->kind() == Input_Var
                && cvi.curr_var()->get_position() > max_level)
              max_level = cvi.curr_var()->get_position();
        } else
          assert(false);
      
      if (!has_wildcard) {
        int num_simple_geq = 0;
        for (GEQ_Iterator e(result.second.single_conjunct()->GEQs()); e;
             e++)
          if (!(*e).has_wildcards()) {
            num_simple_geq++;
            for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
              if (cvi.curr_var()->kind() == Input_Var
                  && cvi.curr_var()->get_position() > max_level) {
                max_level = cvi.curr_var()->get_position();
                direction = (cvi.curr_coef() < 0) ? true : false;
              }
          } else {
            has_wildcard = true;
            for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
              if (cvi.curr_var()->kind() == Input_Var
                  && cvi.curr_var()->get_position() > max_level) {
                max_level = cvi.curr_var()->get_position();
              }
          }
        assert(
               (has_wildcard && num_simple_geq == 0) 
               || (!has_wildcard && num_simple_geq == 1));
      }
      
      // check if this is the top loop level for splitting for this overhead
      if (!propagate_up || (has_wildcard && max_level == level_ - 1)
          || (!has_wildcard && max_level == level_)) {
        std::vector<Relation> restrictions(2);
        std::vector<CG_result *> clauses(2);
        int saved_num_level = num_level();
        if (has_wildcard || direction) {
          restrictions[1] = Complement(copy(result.second));
          restrictions[1].simplify();
          clauses[1] = this->clone();
          restrictions[0] = result.second;
          clauses[0] = this;
        } else {
          restrictions[0] = Complement(copy(result.second));
          restrictions[0].simplify();
          clauses[0] = this->clone();
          restrictions[1] = result.second;
          clauses[1] = this;
        }
        CG_result *cgr = new CG_split(codegen_, active_, restrictions,
                                      clauses);
        CG_result *new_cgr = cgr->recompute(active_, copy(known_),
                                            copy(restriction_));
        new_cgr->populateDepth();
        assert(new_cgr==cgr);
        if (static_cast<CG_split *>(new_cgr)->clauses_.size() == 1)
          // infinite recursion detected, bail out
          return std::make_pair(new_cgr, Relation::True(saved_num_level));
        else
          return cgr->liftOverhead(depth, propagate_up);
      } else
        return std::make_pair(this, result.second);
    }
  }
  
  
  
  Relation CG_loop::hoistGuard() {
    
    Relation r = body_->hoistGuard();
    
    // TODO: should bookkeep catched contraints in loop output as enforced and check if anything missing
    // if (!Gist(copy(b), copy(enforced)).is_obvious_tautology()) {
    //   debug_fprintf(stderr, "need to generate extra guard inside the loop\n");
    // }
    
    if (!needLoop_)
      r = Intersection(r, copy(bounds_));
    r = Project(r, r.set_var(level_));
    r = Gist(r, copy(known_), 1);
    
    Relation eliminate_existentials_r;
    Relation eliminate_existentials_known;
    
    eliminate_existentials_r = copy(r);
    if (!r.is_obvious_tautology()) {
      eliminate_existentials_r = Approximate(copy(r));
      eliminate_existentials_r.simplify(2,4);
      eliminate_existentials_known = Approximate(copy(known_));
      eliminate_existentials_known.simplify(2,4);
      
      eliminate_existentials_r = Gist(eliminate_existentials_r, 
                                      eliminate_existentials_known, 1);
    }
    
    
    if (!eliminate_existentials_r.is_obvious_tautology()) {
      // if (!r.is_obvious_tautology()) {
      body_->removeGuard(r);
      guard_ = Intersection(guard_, copy(r));
      guard_.simplify();
    }
    
    return guard_;
  }
  


  void CG_loop::removeGuard(const Relation &guard) {
    known_ = Intersection(known_, copy(guard));
    known_.simplify();
    
    guard_ = Gist(guard_, copy(known_), 1);
    guard_.copy_names(known_);
    guard_.setup_names();
  }
  



  CG_outputRepr *CG_loop::printRepr(int indent, 
                                    CG_outputBuilder *ocg,
                                    const std::vector<CG_outputRepr *> &stmts,
                                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                    std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString) const {
    
    debug_fprintf(stderr, "CG_loop::printRepr() w assigned_on_the_fly gonna call printRepr with more arguments\n"); 
    //int numfly =  assigned_on_the_fly.size();
    //debug_fprintf(stderr, "assigned on the fly  %d\n", numfly );
    //for (int i=0; i<numfly; i++) { 
    //  //debug_fprintf(stderr, "i %d\n", i); 
    //  std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
    //  CG_outputRepr *tr = NULL;
    //  if (p.first != NULL) tr = p.first->clone();
    //  int val = p.second;
    //  //debug_fprintf(stderr, "0x%x   %d\n", tr, val);
    //} 
    
    return printRepr(true, indent, ocg, stmts, assigned_on_the_fly, unin, printString);
  }




  
  CG_outputRepr *CG_loop::printRepr(bool do_print_guard, 
                                    int indent,
                                    CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts,
                                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                    std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString) const {
    debug_fprintf(stderr, "\n*** CG.cc  CG_loop printrepr with more arguments\n"); 
    
    
    // debugging output 
    int numfly =  assigned_on_the_fly.size();
    debug_fprintf(stderr, "assigned on the fly  %d\n", numfly ); // Anand makes twice as many
    for (int i=0; i<numfly; i++) { 
      //debug_fprintf(stderr, "i %d\n", i); 
      std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
      CG_outputRepr *tr = NULL;
      if (p.first != NULL) tr = p.first->clone();
      int val = p.second;
      //debug_fprintf(stderr, "0x%x   %d\n", tr, val);
    }
    
    //Anand: adding support for Replacing substituted variables within
    //Uninterpreted function symbols or global variables with arity > 0 here
    //--begin
    std::vector<std::pair<CG_outputRepr *, int> > aotf = assigned_on_the_fly;
    int stmt_num = -1;
    for (int s = 0; s < active_.size(); s++)
      if (active_.get(s))
        stmt_num = s;
    
    assert(stmt_num != -1);
    
    CG_outputRepr *guardRepr;
    if (do_print_guard)
      guardRepr = output_guard(ocg, guard_,  aotf, unin[stmt_num]);
    else
      guardRepr = NULL;
    
    debug_fprintf(stderr, "after guard assigned on the fly  %d\n", numfly );
    for (int i=0; i<numfly; i++) { 
      //debug_fprintf(stderr, "i %d\n", i); 
      std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
      CG_outputRepr *tr = NULL;
      if (p.first != NULL) tr = p.first->clone();
      int val = p.second;
      //debug_fprintf(stderr, "0x%x   %d\n", tr, val);
    }
    debug_fprintf(stderr, "done flying\n"); 

    Relation cur_known = Intersection(copy(known_), copy(guard_));
    
    cur_known.simplify();
    debug_fprintf(stderr, "checking needloop\n"); 
    if (needLoop_) {
      debug_fprintf(stderr, "needLoop_\n"); 
      
      if (checkLoopLevel)
        if (level_ == checkLoopLevel)
          if (active_.get(stmtForLoopCheck))
            fillInBounds = true;

      debug_fprintf(stderr, "ctrlRepr = output_loop()\n"); 
      CG_outputRepr *ctrlRepr = output_loop(ocg, bounds_, level_, cur_known,
                                            aotf, unin[stmt_num]);
      
      fillInBounds = false;
      
      debug_fprintf(stderr, "in needLoop_ bodyrepr = \n"); 
      int ind = (guardRepr == NULL) ? indent + 1 : indent + 2;
      CG_outputRepr *bodyRepr = body_->printRepr(ind,
                                                 ocg, 
                                                 stmts,
                                                 aotf, 
                                                 unin, 
                                                 printString);
      CG_outputRepr * loopRepr;
      
      if (guardRepr == NULL)
        loopRepr = ocg->CreateLoop(indent, ctrlRepr, bodyRepr);
      else
        loopRepr = ocg->CreateLoop(indent + 1, ctrlRepr, bodyRepr);
      
      
      if (!smtNonSplitLevels.empty()) {
        debug_fprintf(stderr, "!smtNonSplitLevels.empty()\n"); 
        bool blockLoop = false;
        bool threadLoop = false;
        bool sync = false;
        int firstActiveStmt = -1;
        for (int s = 0; s < active_.size(); s++) {
          if (active_.get(s)) {
            if (firstActiveStmt < 0)
              firstActiveStmt = s;
            //We assume smtNonSplitLevels is only used to mark the first of
            //the block or thread loops to be reduced in CUDA-CHiLL. Here we
            //place some comments to help with final code generation.
            //int idx = smtNonSplitLevels[s].index(level_);
            
            if (s < smtNonSplitLevels.size()) {
              if (smtNonSplitLevels[s].size() > 0)
                if (smtNonSplitLevels[s][0] == level_) {
                  blockLoop = true;
                }
              //Assume every stmt marked with a thread loop index also has a block loop idx
              if (smtNonSplitLevels[s].size() > 1)
                if (smtNonSplitLevels[s][1] == level_) {
                  threadLoop = true;
                }
            }
          }
        }
        if (blockLoop && threadLoop) { 
         debug_fprintf(stderr,
                  "Warning, have %d level more than once in smtNonSplitLevels\n",
                  level_);
          threadLoop = false;
        }
        std::string preferredIdx;

        debug_fprintf(stderr, "loopIdxNames.size() %d\n", loopIdxNames.size());
        for (int i=0; i<loopIdxNames.size(); i++) { 
          debug_fprintf(stderr, "\n"); 
          for (int j=0; j<loopIdxNames[i].size(); j++) { 
            debug_fprintf(stderr, "i %d   j %d %s\n", i, j,loopIdxNames[i][j].c_str() ); 
          }
        } 

        debug_fprintf(stderr, "firstActiveStmt %d\n", firstActiveStmt);
        debug_fprintf(stderr, "loopIdxNames[firstActiveStmt].size() %d\n", loopIdxNames[firstActiveStmt].size()); 
        debug_fprintf(stderr, "level_ %d   /2 %d\n", level_, level_/2); 

        if (loopIdxNames.size()
            && (level_ / 2) - 1 < loopIdxNames[firstActiveStmt].size()) {

          preferredIdx = loopIdxNames[firstActiveStmt][(level_ / 2) - 1];
        }
        for (int s = 0; s < active_.size(); s++) {
          if (active_.get(s)) {
            for (int ii = 0; ii < syncs.size(); ii++) {
              if (syncs[ii].first == s
                  && strcmp(syncs[ii].second.c_str(),
                            preferredIdx.c_str()) == 0) {
                sync = true;
                //printf("FOUND SYNC\n");
              }
              
            }
          }
        }

        if ( preferredIdx.length() != 0) {
          debug_fprintf(stderr, "CG.cc  preferredIdx %s\n", preferredIdx.c_str()); 
        } 

        if (threadLoop || blockLoop || preferredIdx.length() != 0) {
          char buf[1024];
          std::string loop;
          if (blockLoop)
            loop = "blockLoop ";
          if (threadLoop)
            loop = "threadLoop ";

          if ( preferredIdx.length() != 0) {
            debug_fprintf(stderr, "CG.cc adding comment with preferredIdx %s\n", preferredIdx.c_str());
          } 

          if (preferredIdx.length() != 0 && sync) {
            sprintf(buf, "~cuda~ %spreferredIdx: %s sync", loop.c_str(),
                    preferredIdx.c_str());
          } else if (preferredIdx.length() != 0) {
            sprintf(buf, "~cuda~ %spreferredIdx: %s", loop.c_str(),
                    preferredIdx.c_str());
          } else {
            sprintf(buf, "~cuda~ %s", loop.c_str());
          }
          
          
          loopRepr = ocg->CreateAttribute(loopRepr, buf);
        }
        
      }
      if (guardRepr == NULL)
        return loopRepr;
      else
        return ocg->CreateIf(indent, guardRepr, loopRepr, NULL);
    } 
    else {
      debug_fprintf(stderr, "NOT needloop_\n");
      
      std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> > result =
        output_assignment(ocg, bounds_, level_, cur_known, aotf, unin[stmt_num]);
      
      //debug_fprintf(stderr, "RESULT  0x%x  0x%x  %d\n", result.first, result.second.first, result.second.second ); 
      
      
      guardRepr = ocg->CreateAnd(guardRepr, result.first);
      //debug_fprintf(stderr, "RESULT  0x%x  0x%x  %d\n", result.first, result.second.first, result.second.second ); 
      
      //debug_fprintf(stderr, "after guardRepr assigned on the fly  %d\n", numfly );
      for (int i=0; i<numfly; i++) { 
        //debug_fprintf(stderr, "i %d\n", i); 
        std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
        CG_outputRepr *tr = NULL;
        if (p.first != NULL) tr = p.first->clone();
        int val = p.second;
        //debug_fprintf(stderr, "0x%x   %d\n", tr, val);
      } 
      
      
      if (result.second.second < CodeGen::var_substitution_threshold) {
        //debug_fprintf(stderr, "var_substitution_threshold  %d < %d    level_ = %d\n", result.second.second, CodeGen::var_substitution_threshold, level_); 
        std::vector<std::pair<CG_outputRepr *, int> > aotf =
          assigned_on_the_fly;
        aotf[level_ - 1] = result.second;
        //debug_fprintf(stderr, "RESULT  0x%x  second 0x%x  %d\n", result.first, result.second.first, result.second.second ); 
        
        if(!printString) {
          for (std::map<std::string, std::vector<CG_outputRepr *> >::iterator i =
                 unin[stmt_num].begin(); i != unin[stmt_num].end(); i++) {
            
            
            std::vector<CG_outputRepr *> to_push;
            for (int j = 0; j < i->second.size(); j++) {
              std::string index =
                const_cast<CG_loop *>(this)->bounds_.set_var(level_)->name();
              std::vector<std::string> loop_vars;
              loop_vars.push_back(index);
              std::vector<CG_outputRepr *> subs;
              subs.push_back(result.second.first->clone());
              CG_outputRepr * new_repr = ocg->CreateSubstitutedStmt(0,
                                                                    i->second[j]->clone(), loop_vars, subs);
              to_push.push_back(new_repr);
            }
            i->second = to_push;
          } // for 
        } // if 
        
        //debug_fprintf(stderr, "aotf !!\n"); 
        for (int i=0; i<numfly; i++) { 
          //debug_fprintf(stderr, "i %d\n", i); 
          std::pair<CG_outputRepr *, int>p = aotf[i];
          CG_outputRepr *tr = NULL;
          if (p.first != NULL) { tr = p.first->clone();  }
          int val = p.second;
        }
        
        //debug_fprintf(stderr, "\nbodyRepr =\n"); 
        //body_->dump(); // this dies 
        int ind =  (guardRepr == NULL) ? indent : indent + 1; 
        CG_outputRepr *bodyRepr = body_->printRepr(ind, ocg, stmts, aotf, unin,
                                                   printString);

        delete aotf[level_ - 1].first;
        if (guardRepr == NULL)
          return bodyRepr;
        else
          return ocg->CreateIf(indent, guardRepr, bodyRepr, NULL);
      } else {
        //debug_fprintf(stderr, "NOT var_substitution_threshold    gonna call output_ident()\n"); 
        int ind =  (guardRepr == NULL) ? indent : indent + 1;
        CG_outputRepr *assignRepr = ocg->CreateAssignment(
          ind,
          output_ident(ocg, bounds_,
                       const_cast<CG_loop *>(this)->bounds_.set_var(
                         level_), aotf, unin[stmt_num]),
          result.second.first);

        CG_outputRepr *bodyRepr = body_->printRepr(ind, ocg, stmts, aotf, unin, 
                                                   printString);

        if (guardRepr == NULL)
          return ocg->StmtListAppend(assignRepr, bodyRepr);
        else
          return ocg->CreateIf(indent, guardRepr,
                               ocg->StmtListAppend(assignRepr, bodyRepr), NULL);

        /*  DEAD CODE
            std::pair<EQ_Handle, int> result_ = find_simplest_assignment(
            copy(bounds_), copy(bounds_).set_var(level_),
            assigned_on_the_fly);
            bool found_func = false;
            Variable_ID v2;
            int arity;
            for (Constr_Vars_Iter cvi(result_.first); cvi; cvi++)
            if (cvi.curr_var()->kind() == Global_Var) {
            Global_Var_ID g = cvi.curr_var()->get_global_var();
            if ((g->arity() > 0)) {
            
            found_func = true;
            arity = g->arity();
            //copy(R).print();
            v2 = copy(bounds_).get_local(g,
            cvi.curr_var()->function_of());
            
            break;
            }
            }
            
            bool is_array = false;
            if (found_func) {
            
            is_array = ocg->QueryInspectorType(
            v2->get_global_var()->base_name());
            
            }
            if (!found_func || !is_array) {
            
            CG_outputRepr *assignRepr = ocg->CreateAssignment(
            (guardRepr == NULL) ? indent : indent + 1,
            output_ident(ocg, bounds_,
            const_cast<CG_loop *>(this)->bounds_.set_var(
            level_), assigned_on_the_fly),
            result.second.first);
            
            CG_outputRepr *bodyRepr = body_->printRepr(
            (guardRepr == NULL) ? indent : indent + 1, ocg, stmts,
            assigned_on_the_fly);
            if (guardRepr == NULL)
            return ocg->StmtListAppend(assignRepr, bodyRepr);
            else
            return ocg->CreateIf(indent, guardRepr,
            ocg->StmtListAppend(assignRepr, bodyRepr), NULL);
            
            } else {
            
            std::vector<CG_outputRepr *> index_expr;
            
            CG_outputRepr* lb = ocg->CreateArrayRefExpression(
            v2->get_global_var()->base_name(),
            output_ident(ocg, bounds_,
            const_cast<CG_loop *>(this)->bounds_.set_var(2),
            assigned_on_the_fly));
            
            for (int i = arity; i > 1; i--) {
            
            index_expr.push_back(
            ocg->CreateArrayRefExpression(
            v2->get_global_var()->base_name(),
            output_ident(ocg, bounds_,
            const_cast<CG_loop *>(this)->bounds_.set_var(
            2 * i),
            assigned_on_the_fly)));
            
            //}
            
            }
            
            CG_outputRepr *ub;
            if (index_expr.size() > 1)
            ub = ocg->CreateInvoke("max", index_expr);
            else
            ub = index_expr[0];
            CG_outputRepr *le = ocg->CreateMinus(ub, ocg->CreateInt(1));
            CG_outputRepr *inductive = ocg->CreateInductive(
            output_ident(ocg, bounds_,
            const_cast<CG_loop *>(this)->bounds_.set_var(
            level_), assigned_on_the_fly), lb, le,
            NULL);
            
            CG_outputRepr *bodyRepr = body_->printRepr(
            (guardRepr == NULL) ? indent : indent + 1, ocg, stmts,
            assigned_on_the_fly);
            
            if (guardRepr == NULL) {
            return ocg->CreateLoop(indent, inductive, bodyRepr);
            } else
            return ocg->CreateIf(indent, guardRepr,
            ocg->CreateLoop(indent + 1, inductive, bodyRepr),
            NULL);
            }
        */      
      }
    }
  }
  

/*
//Protonu--adding a helper function to get the ids of the nested loops
//to help with OpenMP code generation
  std::vector<std::string> get_idxs( CG_result * cgl, std::vector<std::string>& idxs)
  {
    CG_loop * _lp;
    CG_split *_sp;
    
    _lp = dynamic_cast<CG_loop *>(cgl);
    _sp = dynamic_cast<CG_split *>(cgl);
    
    if(_lp){
      if (_lp->needLoop_)
        idxs.push_back((copy(_lp->bounds_).set_var(_lp->level_))->name());
      if (_lp->depth() > 0 ){
        cgl = _lp->body_;
        get_idxs(cgl, idxs);
      }
      if (_lp->depth() == 0)
        return idxs;
    }
    if(_sp){
      for(int i=0; i<_sp->clauses_.size(); i++)
      {
        get_idxs(_sp->clauses_[i], idxs);
      }
    }
    
    return idxs;
    
  }
//end.
*/


  CG_result *CG_loop::clone() const {
    //debug_fprintf(stderr, "CG_loop::clone()\n"); 
    return new CG_loop(codegen_, active_, level_, body_->clone());
  }
  
  void CG_loop::dump(int indent) const {
    std::string prefix;
    for (int i = 0; i < indent; i++)
      prefix += "  ";
    std::cout << prefix << "LOOP (level " << level_ << "): " << active_
              << std::endl;
    std::cout << prefix << "known: ";
    const_cast<CG_loop *>(this)->known_.print();
    std::cout << prefix << "restriction: ";
    const_cast<CG_loop *>(this)->restriction_.print();
    std::cout << prefix << "bounds: ";
    const_cast<CG_loop *>(this)->bounds_.print();
    std::cout << prefix << "guard: ";
    const_cast<CG_loop *>(this)->guard_.print();
    body_->dump(indent + 1);
  }
  
  //-----------------------------------------------------------------------------
  // Class: CG_leaf
  //-----------------------------------------------------------------------------
  
  CG_result* CG_leaf::recompute(const BoolSet<> &parent_active,
                                const Relation &known, const Relation &restriction) {
    active_ &= parent_active;
    known_ = copy(known);
    
    guards_.clear();
    for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
      Relation r = Intersection(
                                copy(codegen_->projected_IS_[num_level() - 1][*i]),
                                copy(restriction));
      r.simplify(2, 4);
      if (!r.is_upper_bound_satisfiable())
        active_.unset(*i);
      else {
        r = Gist(r, copy(known), 1);
        if (!r.is_obvious_tautology()) {
          guards_[*i] = r;
          guards_[*i].copy_names(known);
          guards_[*i].setup_names();
        }
      }
    }
    
    
    if (active_.empty()) {
      delete this;
      return NULL;
    } else
      return this;
  }
  
  std::pair<CG_result *, Relation> CG_leaf::liftOverhead(int depth, bool) {
    if (depth == 0)
      return std::make_pair(this, Relation::True(num_level()));
    
    for (std::map<int, Relation>::iterator i = guards_.begin();
         i != guards_.end(); i++) {
      Relation r = pick_one_guard(i->second);
      if (!r.is_obvious_tautology()) {
        bool has_wildcard = false;
        int max_level = 0;
        for (EQ_Iterator e(r.single_conjunct()->EQs()); e; e++) {
          if ((*e).has_wildcards())
            has_wildcard = true;
          for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
            if (cvi.curr_var()->kind() == Input_Var
                && cvi.curr_var()->get_position() > max_level)
              max_level = cvi.curr_var()->get_position();
        }
        for (GEQ_Iterator e(r.single_conjunct()->GEQs()); e; e++) {
          if ((*e).has_wildcards())
            has_wildcard = true;
          for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
            if (cvi.curr_var()->kind() == Input_Var
                && cvi.curr_var()->get_position() > max_level)
              max_level = cvi.curr_var()->get_position();
        }
        
        if (!(has_wildcard && max_level == codegen_->num_level()))
          return std::make_pair(this, r);
      }
    }
    
    return std::make_pair(this, Relation::True(num_level()));
  }
  
  Relation CG_leaf::hoistGuard() {
    std::vector<Relation> guards;
    for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
      std::map<int, Relation>::iterator j = guards_.find(*i);
      if (j == guards_.end()) {
        Relation r = Relation::True(num_level());
        r.copy_names(known_);
        r.setup_names();
        return r;
      } else {
        guards.push_back(j->second);
      }
    }
    
    return SimpleHull(guards, true, true);
  }
  
  void CG_leaf::removeGuard(const Relation &guard) {
    known_ = Intersection(known_, copy(guard));
    known_.simplify();
    
    std::map<int, Relation>::iterator i = guards_.begin();
    while (i != guards_.end()) {
      i->second = Gist(i->second, copy(known_), 1);
      if (i->second.is_obvious_tautology())
        guards_.erase(i++);
      else
        ++i;
    }
  }
  
  CG_outputRepr *CG_leaf::printRepr(int indent, CG_outputBuilder *ocg,
                                    const std::vector<CG_outputRepr *> &stmts,
                                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                    std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, 
                                    bool printString) const {
    debug_fprintf(stderr, "CG_leaf::printRepr()\n"); 
    int numfly =  assigned_on_the_fly.size();
    //debug_fprintf(stderr, "assigned on the fly  %d\n", numfly );
    for (int i=0; i<numfly; i++) { 
      //debug_fprintf(stderr, "i %d\n", i); 
      std::pair<CG_outputRepr *, int>p = assigned_on_the_fly[i];
      CG_outputRepr *tr = NULL;
      if (p.first != NULL) tr = p.first->clone();
      int val = p.second;
      //debug_fprintf(stderr, "0x%x   %d\n", tr, val);
    }

    return leaf_print_repr(active_, guards_, NULL, known_, indent, ocg,
                           codegen_->remap_, codegen_->xforms_, stmts, 
                           assigned_on_the_fly, unin);
  }
  


  CG_result *CG_leaf::clone() const {
    //debug_fprintf(stderr, "CG_leaf::clone()\n"); 
    return new CG_leaf(codegen_, active_);
  }
  
  void CG_leaf::dump(int indent) const {
    
    std::string prefix;
    for (int i = 0; i < indent; i++)
      prefix += "  ";
    std::cout << prefix << "LEAF: " << active_ << std::endl;
    std::cout << prefix << "known: ";
    std::cout.flush();
    const_cast<CG_leaf *>(this)->known_.print();
    for (std::map<int, Relation>::const_iterator i = guards_.begin();
         i != guards_.end(); i++) {
      std::cout << prefix << "guard #" << i->first << ":";
      const_cast<Relation &>(i->second).print();
    }
  }
  
}
