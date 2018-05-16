/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   Utility functions for processing CG tree.

 Notes:
 
 History:
   07/19/07 when generating output variable substitutions for a mapping
            relation, map it to each output to get correct equality, -chun
   07/29/10 when translating equality to assignment, generate appropriate
            if-condition when necesssary, -chun
*****************************************************************************/

#include <typeinfo>
#include <omega.h>
#include <code_gen/CG.h>
#include <code_gen/CG_utils.h>
#include <code_gen/codegen_error.h>
#include <math.h>
#include <stack>

namespace omega {
  
  int checkLoopLevel = 0;
  int stmtForLoopCheck;
  int upperBoundForLevel;
  int lowerBoundForLevel;
  bool fillInBounds = false;

  bool bound_must_hit_stride(const GEQ_Handle &inequality, 
                             Variable_ID v, 
                             const EQ_Handle &stride_eq, 
                             Variable_ID wc, 
                             const Relation &bounds, 
                             const Relation &known) {
    assert(inequality.get_coef(v) != 0 
           && abs(stride_eq.get_coef(v)) == 1 
           && wc->kind() == Wildcard_Var 
           && abs(stride_eq.get_coef(wc)) > 1);
    
    // if bound expression uses floor operation, bail out for now
    // TODO: in the future, handle this
    if (abs(inequality.get_coef(v)) != 1)
      return false;
    
    coef_t stride = abs(stride_eq.get_coef(wc));
    
    Relation r1(known.n_set());
    F_Exists *f_exists1 = r1.add_and()->add_exists();
    F_And *f_root1 = f_exists1->add_and();
    std::map<Variable_ID, Variable_ID> exists_mapping1;
    EQ_Handle h1 = f_root1->add_EQ();
    Relation r2(known.n_set());
    F_Exists *f_exists2 = r2.add_and()->add_exists();
    F_And *f_root2 = f_exists2->add_and();
    std::map<Variable_ID, Variable_ID> exists_mapping2;
    EQ_Handle h2 = f_root2->add_EQ();
    for (Constr_Vars_Iter cvi(inequality); cvi; cvi++) {
      Variable_ID v = cvi.curr_var();
      switch (v->kind()) {
      case Input_Var: 
        h1.update_coef(r1.input_var(v->get_position()), cvi.curr_coef());
        h2.update_coef(r2.input_var(v->get_position()), cvi.curr_coef());
        break;
      case Global_Var: {      
        Global_Var_ID g = v->get_global_var();
        Variable_ID v1, v2;
        if (g->arity() == 0) {
          v1 = r1.get_local(g);
          v2 = r2.get_local(g);
        }
        else {
          v1 = r1.get_local(g, v->function_of());
          v2 = r2.get_local(g, v->function_of());
        }
        h1.update_coef(v1, cvi.curr_coef());
        h2.update_coef(v2, cvi.curr_coef());
        break;
      }
      case Wildcard_Var: {
        Variable_ID v1 = replicate_floor_definition(bounds, 
                                                    v, 
                                                    r1, 
                                                    f_exists1, 
                                                    f_root1, 
                                                    exists_mapping1);
        Variable_ID v2 = replicate_floor_definition(bounds, 
                                                    v, 
                                                    r2, 
                                                    f_exists2, 
                                                    f_root2, 
                                                    exists_mapping2);
        h1.update_coef(v1, cvi.curr_coef());
        h2.update_coef(v2, cvi.curr_coef());
        break;
      }
      default:
        assert(false);
      }
    }
    h1.update_const(inequality.get_const());
    h1.update_coef(f_exists1->declare(), stride);
    h2.update_const(inequality.get_const());
    r1.simplify();
    r2.simplify();

    Relation all_known = Intersection(copy(bounds), copy(known));
    all_known.simplify();

    if (Gist(r1, copy(all_known), 1).is_obvious_tautology()) {
      Relation r3(known.n_set());
      F_Exists *f_exists3 = r3.add_and()->add_exists();
      F_And *f_root3 = f_exists3->add_and();
      std::map<Variable_ID, Variable_ID> exists_mapping3;
      EQ_Handle h3 = f_root3->add_EQ();
      for (Constr_Vars_Iter cvi(stride_eq); cvi; cvi++) {
        Variable_ID v= cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          h3.update_coef(r3.input_var(v->get_position()), cvi.curr_coef());
          break;
        case Global_Var: {
          Global_Var_ID g = v->get_global_var();
          Variable_ID v3;
          if (g->arity() == 0)
            v3 = r3.get_local(g);
          else
            v3 = r3.get_local(g, v->function_of());
          h3.update_coef(v3, cvi.curr_coef());
          break;
        }
        case Wildcard_Var:
          if (v == wc)
            h3.update_coef(f_exists3->declare(), cvi.curr_coef());
          else {
            Variable_ID v3 = replicate_floor_definition(bounds, 
                                                        v, 
                                                        r3, 
                                                        f_exists3, 
                                                        f_root3, 
                                                        exists_mapping3);
            h3.update_coef(v3, cvi.curr_coef());
          }
          break;
        default:
          assert(false);
        }
      }
      h3.update_const(stride_eq.get_const());
      r3.simplify();

      return Gist(r3, Intersection(r2, all_known), 1).is_obvious_tautology();
    } else
      return false;
  }
  
  
  CG_outputRepr *output_ident(CG_outputBuilder *ocg,
                              const Relation &R, 
                              Variable_ID v, 
                              const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, 
                              const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    debug_fprintf(stderr, "output_ident( %s )\n", v->name().c_str()); 
    
    const_cast<Relation &>(R).setup_names(); // hack
    
    if (v->kind() == Input_Var) {
      int pos = v->get_position();
      if (assigned_on_the_fly[pos - 1].first != NULL)
        return assigned_on_the_fly[pos-1].first->clone();
      else
        return ocg->CreateIdent(v->name());
    } else if (v->kind() == Global_Var) {
      if (v->get_global_var()->arity() == 0)
        return ocg->CreateIdent(v->name());
      else {
        /* TODO This should be improved to take into account the possible elimination
           of the set variables. */
        int arity = v->get_global_var()->arity();
        std::vector<CG_outputRepr *> argList;
        if (unin.find(v->get_global_var()->base_name()) != unin.end()) {
          auto &argList_ = unin.find(v->get_global_var()->base_name())->second;
          for (auto &l: argList_)
            argList.push_back(l->clone());
        } else {
          for (int i = 1; i <= arity; i++) {
            if (assigned_on_the_fly.size() > (R.n_inp() + i - 1)) {
              if (assigned_on_the_fly[R.n_inp() + i - 1].first != NULL)
                argList.push_back(
                  assigned_on_the_fly[R.n_inp() + i - 1].first->clone());
              else
                argList.push_back(
                  ocg->CreateIdent(
                    const_cast<Relation &>(R).input_var(
                      2 * i)->name()));
            } else
              argList.push_back(
                ocg->CreateIdent(
                  const_cast<Relation &>(R).input_var(
                    2 * i)->name()));
            
            /*if (assigned_on_the_fly[2*i - 1].first == NULL)
              argList.push_back(
              ocg->CreateIdent(
              const_cast<Relation &>(R).input_var(2 * i)->name()));
              else
              argList.push_back(assigned_on_the_fly[2*i - 1].first->clone());
            */
          }
        }
        CG_outputRepr *call = ocg->CreateInvoke(
          v->get_global_var()->base_name(), argList);
        return call;
      }
    }
    else
      assert(false);
  }
  
  
  std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> > output_assignment(
    CG_outputBuilder *ocg, const Relation &R, int level, 
    const Relation &known, 
    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
    const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    
    Variable_ID v = const_cast<Relation &>(R).set_var(level);
    Conjunct *c = const_cast<Relation &>(R).single_conjunct();
    
    std::pair<EQ_Handle, int> result = find_simplest_assignment(R, v, assigned_on_the_fly);
    
    if (result.second == INT_MAX)
      return std::make_pair(static_cast<CG_outputRepr *>(nullptr), std::make_pair(static_cast<CG_outputRepr *>(nullptr), INT_MAX));
    
    CG_outputRepr *if_repr = nullptr;
    CG_outputRepr *assign_repr = nullptr;
    // check whether to generate if-conditions from equality constraints
    if (abs(result.first.get_coef(v)) != 1) {
      Relation r = Project(R, v);
      r.simplify();
      if (!Gist(r, copy(known), 1).is_obvious_tautology()) {
        CG_outputRepr *lhs = output_substitution_repr(ocg, result.first, v, 
                                                      false, R, assigned_on_the_fly, unin);  
        if_repr = ocg->CreateEQ(ocg->CreateIntegerMod(lhs->clone(), 
                                                      ocg->CreateInt(abs(result.first.get_coef(v)))),
                                ocg->CreateInt(0));
        assign_repr = ocg->CreateDivide(lhs, ocg->CreateInt(abs(result.first.get_coef(v))));
      }
      else
        assign_repr = output_substitution_repr(ocg, result.first, v, true, R,
                                               assigned_on_the_fly, unin);
    }
    else
      assign_repr = output_substitution_repr(ocg, result.first, v, true, R,
                                             assigned_on_the_fly, unin);
    if (assign_repr == nullptr)
      assign_repr = ocg->CreateInt(0);

    return std::make_pair(if_repr, std::make_pair(assign_repr, result.second));
  }
 
  
  CG_outputRepr *output_substitution_repr(CG_outputBuilder *ocg,
                                          const EQ_Handle &equality, 
                                          Variable_ID v, 
                                          bool apply_v_coef, 
                                          const Relation &R, 
                                          const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                          const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    const_cast<Relation &>(R).setup_names(); // hack
    
    coef_t a = equality.get_coef(v);
    assert(a != 0);
    
    CG_outputRepr *repr = nullptr;
    for (Constr_Vars_Iter cvi(equality); cvi; cvi++)
      if (cvi.curr_var() != v) {
        CG_outputRepr *t;
        if (cvi.curr_var()->kind() == Wildcard_Var) {
          std::pair<bool, GEQ_Handle> result = find_floor_definition(R, cvi.curr_var());
          if (!result.first) {
            delete repr;
            throw codegen_error("can't output non floor defined wildcard");
          }
          t = output_inequality_repr(ocg, result.second, cvi.curr_var(), R, assigned_on_the_fly, unin);
        }
        else
          t = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly, unin);
        coef_t coef = cvi.curr_coef();
        coef_t absc = abs(coef);

        if (signbit(a) ^ signbit(coef)) { // a * c <= 0
          if (absc == 1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr, ocg->CreateTimes(ocg->CreateInt(absc), t));
        } else {
          if (absc == 1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr, ocg->CreateTimes(ocg->CreateInt(absc), t));
        }
      }
    
    coef_t c = equality.get_const();
    coef_t absc = abs(c);
    if (c) {
      if (signbit(a) ^ signbit(c)) // a * c <= 0
        repr = ocg->CreatePlus(repr, ocg->CreateInt(absc));
      else
        repr = ocg->CreateMinus(repr, ocg->CreateInt(absc));
    }

    if (apply_v_coef && abs(a) != 1)
      repr = ocg->CreateDivide(repr, ocg->CreateInt(abs(a)));

    return repr;
  }
  
  
//
// original Substitutions class from omega can't handle integer
// division, this is new way.
//
  std::vector<CG_outputRepr*> output_substitutions(CG_outputBuilder *ocg, 
                                                   const Relation &R, 
                                                   const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, 
                                                   const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    std::vector<CG_outputRepr *> subs;
    
    debug_fprintf(stderr, "CG_utils.cc  output_substitutions()\n"); 
    
    for (int i = 1; i <= R.n_out(); i++) {
      Relation mapping(R.n_out(), 1);
      F_And *f_root = mapping.add_and();
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(1), 1);
      h.update_coef(mapping.input_var(i), -1);
      Relation r = Composition(mapping, copy(R));
      r.simplify();
      
      Variable_ID v = r.output_var(1);
      CG_outputRepr *repr = NULL;
      
      std::pair<EQ_Handle, int> result = find_simplest_assignment(r, v, assigned_on_the_fly);
      if (result.second < INT_MAX)
        repr = output_substitution_repr(ocg, result.first, v, true, r,
                                        assigned_on_the_fly, unin);
      else {
        auto result = find_floor_definition(R, v);
        if (result.first)
          try {
            repr = output_inequality_repr(ocg, result.second, v, R,
                                          assigned_on_the_fly, unin);
          }
          catch (const codegen_error &) {
          }
      }

      if (repr != NULL) {
        subs.push_back(repr);
      }
      else { 
        //Anand:Add additional checks for variables that have been mapped to inspector/global variables
        for (int k = 1; k <= R.n_out(); k++) {
          //Relation mapping1(mapping.n_out(), 1);
          //F_And *f_root = mapping1.add_and();
          //EQ_Handle h = f_root->add_EQ();
          //h.update_coef(mapping1.output_var(1), 1);
          //h.update_coef(mapping1.input_var(i), -1);
          //Relation r = Composition(mapping1, copy(mapping));
          //r.simplify();
          
          Relation r = copy(R);
          
          Variable_ID v = r.output_var(k);
          CG_outputRepr* inspector = NULL;
          bool has_insp = false;
          Global_Var_ID global_insp;
          
          std::pair<EQ_Handle, int> result = find_simplest_assignment(
            copy(r), v, assigned_on_the_fly);
          CG_outputRepr *repr_ = NULL;
          if (result.second < INT_MAX) {
            /*std::vector<Variable_ID> variables;
            //Anand: commenting this out as well :  05/29/2013
            variables.push_back(r.input_var(2 * k));
            for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
            Variable_ID v1 = cvi.curr_var();
            if (v1->kind() == Input_Var)
            variables.push_back(v1);
            
            }
            */
            repr_ = output_substitution_repr(ocg, result.first, v, true,
                                             copy(r), assigned_on_the_fly, unin);
            
            for (DNF_Iterator di(copy(r).query_DNF()); di; di++)
              for (GEQ_Iterator ci = (*di)->GEQs(); ci; ci++) {
                
                //Anand: temporarily commenting this out as it causes problems 5/28/2013
                /*int j;
                  for (j = 0; j < variables.size(); j++)
                  if ((*ci).get_coef(variables[j]) == 0)
                  break;
                  if (j != variables.size())
                  continue;
                */
                for (Constr_Vars_Iter cvi(*ci); cvi; cvi++) {
                  Variable_ID v1 = cvi.curr_var();
                  if (v1->kind() == Global_Var)
                    if (v1->get_global_var()->arity() > 0) {
                      
                      if (i
                          <= v1->get_global_var()->arity()) {
                        
                        /*  std::pair<EQ_Handle, int> result =
                            find_simplest_assignment(
                            copy(r), v,
                            assigned_on_the_fly);
                            
                            if (result.second < INT_MAX) {
                            CG_outputRepr *repr =
                            output_substitution_repr(
                            ocg,
                            result.first, v,
                            true, copy(r),
                            assigned_on_the_fly);
                            
                            inspector =
                            ocg->CreateArrayRefExpression(
                            ocg->CreateDotExpression(
                            ocg->ObtainInspectorData(
                            v1->get_global_var()->base_name()),
                            ocg->CreateIdent(
                            copy(
                            R).output_var(
                            i)->name())),
                            repr);
                            } else
                        */
                        inspector =
                          /*  ocg->CreateArrayRefExpression(
                              ocg->CreateDotExpression(
                              ocg->ObtainInspectorData(
                              v1->get_global_var()->base_name()),
                              ocg->CreateIdent(
                              copy(
                              R).output_var(
                              i)->name())),
                              repr);*/
                          
                          ocg->CreateArrayRefExpression(
                            
                            ocg->ObtainInspectorData(
                              v1->get_global_var()->base_name(),
                              
                              copy(R).output_var(
                                i)->name()),
                            repr_);
                        //ocg->CreateIdent(
                        //    v->name()));
                        
                      }
                    }
                  
                }
              }
          }
          
          if (inspector != NULL) {
            subs.push_back(inspector);
            break;
          } else if (k == i && repr_ != NULL) {
            subs.push_back(repr_);
            
          }
          
          //std::pair<EQ_Handle, int> result = find_simplest_assignment(r,
          //    v, assigned_on_the_fly, &has_insp);
          
          /*if (has_insp) {
            
            bool found_insp = false;
            //Anand: check if inspector constraint present in equality
            for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            if (v->kind() == Global_Var)
            if (v->get_global_var()->arity() > 0) {
            global_insp = v->get_global_var();
            found_insp = true;
            }
            
            }
            if (found_insp)
            if (i <= global_insp->arity())
            inspector =
            ocg->CreateArrayRefExpression(
            ocg->CreateDotExpression(
            ocg->ObtainInspectorData(
            global_insp->base_name()),
            ocg->CreateIdent(
            copy(R).output_var(
            i)->name())),
            ocg->CreateIdent(v->name()));
            
            
            }
          */
        } // for int k 
      }
    }
    
    debug_fprintf(stderr, "CG_utils.cc  output_substitutions()          DONE\n\n"); 
    return subs;
  }
  
  
  std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R,
                                                     Variable_ID v, 
                                                     const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                                     bool *found_inspector_global) {
    Conjunct *c = const_cast<Relation &>(R).single_conjunct();

    int min_cost = INT_MAX;
    EQ_Handle eq;
    for (EQ_Iterator e(c->EQs()); e; e++)
      if ((*e).get_coef(v) != 0) {
        bool is_assignment = true;
        for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
          std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v);
          if (!result.first) {
            is_assignment = false;
            break;
          }
        }
        if (!is_assignment)
          continue;

        int cost = 0;

        if (abs((*e).get_coef(v)) != 1)
          cost += 4;  // divide cost

        int num_var = 0;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
          if (cvi.curr_var() != v) {
            num_var++;
            if (abs(cvi.curr_coef()) != 1)
              cost += 2;  // multiply cost
            if (cvi.curr_var()->kind() == Wildcard_Var)
              cost += 10; // floor operation cost
            else if (cvi.curr_var()->kind() == Global_Var && cvi.curr_var()->get_global_var()->arity() > 0) {
              //Anand  --start: Check for Global Variable not being involved in any GEQ
              Conjunct *d = const_cast<Relation &>(R).single_conjunct();
              bool found = false;
              for (GEQ_Iterator g(d->GEQs()); g; g++)
                if ((*g).get_coef(v) != 0) {
                  found = true;
                  break;
                }
              if (!found)
                cost += 20;  // function cost
              else {
                cost = INT_MAX;
                eq = *e;
                if (found_inspector_global != NULL)
                  *found_inspector_global = true;
                return std::make_pair(eq, cost);
              } //Anand    --end: Check for Global Variable not being involved in any GEQ
            }
            else if (cvi.curr_var()->kind() == Input_Var &&
                     assigned_on_the_fly.size() >= cvi.curr_var()->get_position() &&
                     assigned_on_the_fly[cvi.curr_var()->get_position()-1].first != NULL) {
              cost += assigned_on_the_fly[cvi.curr_var()->get_position()-1].second;  // substitution cost on record
            }
          }
        if ((*e).get_const() != 0)
          num_var++;
        if (num_var > 1)
          cost += num_var - 1; // addition cost

        if (cost < min_cost) {
          min_cost = cost;
          eq = *e;
          if (found_inspector_global != NULL)
            *found_inspector_global = true;
        }
      }
    return std::make_pair(eq, min_cost);

  }
  
  
//
// find floor definition for variable v, e.g. m-c <= 4v <= m, (c is
// constant and 0 <= c < 4). this translates to v = floor(m, 4) and
// return 4v<=m in this case. All wildcards in such inequality are
// also floor defined.
//
  std::pair<bool, GEQ_Handle> find_floor_definition(const Relation &R, 
                                                    Variable_ID v, 
                                                    std::set<Variable_ID> excluded_floor_vars) {
    Conjunct *c = const_cast<Relation &>(R).single_conjunct();

    excluded_floor_vars.insert(v);
    for (GEQ_Iterator e = c->GEQs(); e; e++) {
      coef_t a = (*e).get_coef(v);
      if (a >= -1)
        continue;
      a = -a;
      
      bool interested = true;
      for (std::set<Variable_ID>::const_iterator i = excluded_floor_vars.begin(); 
           i != excluded_floor_vars.end(); 
           i++) {
        if ((*i) != v && (*e).get_coef(*i) != 0) {
          interested = false;
          break;
        }
      }
      if (!interested)
        continue;
      
      // check if any wildcard is floor defined
      bool has_undefined_wc = false;
      for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) 
        if (excluded_floor_vars.find(cvi.curr_var()) == excluded_floor_vars.end()) {
          std::pair<bool, GEQ_Handle> result = find_floor_definition(R, 
                                                                     cvi.curr_var(), 
                                                                     excluded_floor_vars);
          if (!result.first) {
            has_undefined_wc = true;
            break;
          }
        }
      if (has_undefined_wc)
        continue;
      
      // find the matching upper bound for floor definition
      for (GEQ_Iterator e2 = c->GEQs(); e2; e2++)
        if ((*e2).get_coef(v) == a && (*e).get_const() + (*e2).get_const() < a) {
          bool match = true;
          for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
            if ((*e2).get_coef(cvi.curr_var()) != -cvi.curr_coef()) {
              match = false;
              break;
            }
          if (!match)
            continue;
          for (Constr_Vars_Iter cvi(*e2); cvi; cvi++)
            if ((*e).get_coef(cvi.curr_var()) != -cvi.curr_coef()) {
              match = false;
              break;
            }
          if (match)
            return std::make_pair(true, *e);
        }
    }
    
    return std::make_pair(false, GEQ_Handle());
  }
  
//
// find the stride involving the specified variable, the stride
// equality can have other wildcards as long as they are defined as
// floor variables.
//
  std::pair<EQ_Handle, Variable_ID> find_simplest_stride(const Relation &R, 
                                                         Variable_ID v) {
    int best_num_var = INT_MAX;
    coef_t best_coef;
    EQ_Handle best_eq;
    Variable_ID best_stride_wc;
    for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); e; e++)
      if ((*e).has_wildcards() && (*e).get_coef(v) != 0) {
        bool is_stride = true;
        bool found_free = false;
        int num_var = 0;
        Variable_ID stride_wc;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
          switch (cvi.curr_var()->kind()) {
          case Wildcard_Var: {
            bool is_free = true;
            for (GEQ_Iterator e2(const_cast<Relation &>(R).single_conjunct()->GEQs()); e2; e2++)
              if ((*e2).get_coef(cvi.curr_var()) != 0) {
                is_free = false;
                break;
              }
            if (is_free) {
              if (found_free)
                is_stride = false;
              else {
                found_free = true;
                stride_wc = cvi.curr_var();
              }
            }
            else {
              std::pair<bool, GEQ_Handle> result = find_floor_definition(R, cvi.curr_var());
              if (!result.first)
                is_stride = false;
            }
            break;
          }
          case Input_Var:
            num_var++;
            break;
          default:
            ;
          }
          
          if (!is_stride)
            break;
        }
        
        if (is_stride) {
          coef_t coef = abs((*e).get_coef(v));
          if (best_num_var == INT_MAX || coef < best_coef ||
              (coef == best_coef && num_var < best_num_var)) {
            best_coef = coef;
            best_num_var = num_var;
            best_eq = *e;
            best_stride_wc = stride_wc;
          }
        }
      }
    
    if (best_num_var != INT_MAX)
      return std::make_pair(best_eq, best_stride_wc);
    else
      return std::make_pair(EQ_Handle(), static_cast<Variable_ID>(NULL));
  }


  CG_outputRepr *output_guard(CG_outputBuilder *ocg,
                              const Relation &R, 
                              const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                              const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    debug_fprintf(stderr, "output_guard()\n");
    assert(R.n_out()==0);
    
    CG_outputRepr *result = NULL;
    Conjunct *c = const_cast<Relation &>(R).single_conjunct();

    // e.g. 4i=5*j
    for (EQ_Iterator e = c->EQs(); e; e++)
      if (!(*e).has_wildcards()) {
        //debug_fprintf(stderr, "EQ\n"); 
        CG_outputRepr *lhs = NULL;
        CG_outputRepr *rhs = NULL;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
          CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly, unin);
          coef_t coef = cvi.curr_coef();
          if (coef > 0) {
            if (coef == 1)
              lhs = ocg->CreatePlus(lhs, v);
            else
              lhs = ocg->CreatePlus(lhs, 
                                    ocg->CreateTimes(ocg->CreateInt(coef), v));
          }
          else { // coef < 0
            if (coef == -1)
              rhs = ocg->CreatePlus(rhs, v);
            else
              rhs = ocg->CreatePlus(rhs, 
                                    ocg->CreateTimes(ocg->CreateInt(-coef), v));
          }
        }
        coef_t c = (*e).get_const();
        
        CG_outputRepr *term;
        if (lhs == NULL) 
          term = ocg->CreateEQ(rhs, ocg->CreateInt(c));
        else {
          if (c > 0)
            rhs = ocg->CreateMinus(rhs, ocg->CreateInt(c));
          else if (c < 0)
            rhs = ocg->CreatePlus(rhs, ocg->CreateInt(-c));
          else if (rhs == NULL)
            rhs = ocg->CreateInt(0);
          term = ocg->CreateEQ(lhs, rhs);
        }
        result = ocg->CreateAnd(result, term);
      }
    
    // e.g. i>5j
    for (GEQ_Iterator e = c->GEQs(); e; e++)
      if (!(*e).has_wildcards()) {
        //debug_fprintf(stderr, "GEQ\n"); 
        CG_outputRepr *lhs = NULL;
        CG_outputRepr *rhs = NULL;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
          CG_outputRepr *v = output_ident(ocg, 
                                          R, 
                                          cvi.curr_var(), 
                                          assigned_on_the_fly, 
                                          unin);
          coef_t coef = cvi.curr_coef();
          if (coef > 0) {
            if (coef == 1)
              lhs = ocg->CreatePlus(lhs, v);
            else
              lhs = ocg->CreatePlus(lhs, 
                                    ocg->CreateTimes(ocg->CreateInt(coef), v));
          }
          else { // coef < 0
            if (coef == -1)
              rhs = ocg->CreatePlus(rhs, v);
            else
              rhs = ocg->CreatePlus(rhs, 
                                    ocg->CreateTimes(ocg->CreateInt(-coef), v));
          }
        }
        coef_t c = (*e).get_const();
        
        CG_outputRepr *term;
        if (lhs == NULL)
          term = ocg->CreateLE(rhs, ocg->CreateInt(c));
        else {
          if (c > 0)
            rhs = ocg->CreateMinus(rhs, ocg->CreateInt(c));
          else if (c < 0)
            rhs = ocg->CreatePlus(rhs, ocg->CreateInt(-c));
          else if (rhs == NULL)
            rhs = ocg->CreateInt(0);
          term = ocg->CreateGE(lhs, rhs);
        }
        //debug_fprintf(stderr, "result =  ocg->CreateAnd(result, term);\n"); 
        result = ocg->CreateAnd(result, term);
      }
    
    // e.g. 4i=5j+4alpha
    for (EQ_Iterator e = c->EQs(); e; e++)
      if ((*e).has_wildcards()) {
        //debug_fprintf(stderr, "EQ ??\n"); 
        Variable_ID wc;
        int num_wildcard = 0;
        int num_positive = 0;
        int num_negative = 0;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
          if (cvi.curr_var()->kind() == Wildcard_Var) {
            num_wildcard++;
            wc = cvi.curr_var();
          }
          else {
            if (cvi.curr_coef() > 0)
              num_positive++;
            else
              num_negative++;
          }
        }
        
        if (num_wildcard > 1) {
          delete result;
          throw codegen_error(
            "Can't generate equality condition with multiple wildcards");
        }
        int sign = (num_positive>=num_negative)?1:-1;
        
        CG_outputRepr *lhs = NULL;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
          if (cvi.curr_var() != wc) {
            CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), 
                                            assigned_on_the_fly, unin);
            coef_t coef = cvi.curr_coef();
            if (sign == 1) {
              if (coef > 0) {
                if (coef == 1)
                  lhs = ocg->CreatePlus(lhs, v);
                else
                  lhs = ocg->CreatePlus(lhs, 
                                        ocg->CreateTimes(ocg->CreateInt(coef), v));
              }
              else { // coef < 0
                if (coef == -1)
                  lhs = ocg->CreateMinus(lhs, v);
                else
                  lhs = ocg->CreateMinus(lhs, 
                                         ocg->CreateTimes(ocg->CreateInt(-coef), v));
              }
            }
            else {
              if (coef > 0) {
                if (coef == 1)
                  lhs = ocg->CreateMinus(lhs, v);
                else
                  lhs = ocg->CreateMinus(lhs, 
                                         ocg->CreateTimes(ocg->CreateInt(coef), v));
              }
              else { // coef < 0
                if (coef == -1)
                  lhs = ocg->CreatePlus(lhs, v);
                else
                  lhs = ocg->CreatePlus(lhs, 
                                        ocg->CreateTimes(ocg->CreateInt(-coef), v));
              }
            }            
          }
        }
        coef_t c = (*e).get_const();
        if (sign == 1) {
          if (c > 0)
            lhs = ocg->CreatePlus(lhs, ocg->CreateInt(c));
          else if (c < 0)
            lhs = ocg->CreateMinus(lhs, ocg->CreateInt(-c));
        }
        else {
          if (c > 0)
            lhs = ocg->CreateMinus(lhs, ocg->CreateInt(c));
          else if (c < 0)
            lhs = ocg->CreatePlus(lhs, ocg->CreateInt(-c));
        }
        
        lhs = ocg->CreateIntegerMod(lhs, 
                                    ocg->CreateInt(abs((*e).get_coef(wc))));
        CG_outputRepr *term = ocg->CreateEQ(lhs, ocg->CreateInt(0));
        result = ocg->CreateAnd(result, term);
      }
    
    // e.g. 4alpha<=i<=5alpha
    for (GEQ_Iterator e = c->GEQs(); e; e++)
      if ((*e).has_wildcards()) {
        //debug_fprintf(stderr, "GEQ ??\n"); 
        Variable_ID wc;
        int num_wildcard = 0;
        for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++)
          if (num_wildcard == 0) {
            wc = cvi.curr_var();
            num_wildcard = 1;
          }
          else
            num_wildcard++;
        
        if (num_wildcard > 1) {
          delete result;
          // e.g.  c*alpha - x >= 0              (*)
          //       -d*alpha + y >= 0             (*)
          //       e1*alpha + f1*beta + g1 >= 0  (**)
          //       e2*alpha + f2*beta + g2 >= 0  (**)
          //       ...
          // TODO: should generate a testing loop for alpha using its lower and
          // upper bounds from (*) constraints and do the same if-condition test
          // for beta from each pair of opposite (**) constraints as above,
          // and exit the loop when if-condition satisfied.
          throw codegen_error(
            "Can't generate multiple wildcard GEQ guards right now");
        }
        
        coef_t c = (*e).get_coef(wc);
        int sign = (c>0)?1:-1;
        
        GEQ_Iterator e2 = e;
        e2++;
        for ( ; e2; e2++) {
          coef_t c2 = (*e2).get_coef(wc);
          if (c2 == 0)
            continue;
          int sign2 = (c2>0)?1:-1;
          if (sign != -sign2)
            continue;
          int num_wildcard2 = 0;
          for (Constr_Vars_Iter cvi(*e2, true); cvi; cvi++)
            num_wildcard2++;
          if (num_wildcard2 > 1)
            continue;
          
          GEQ_Handle lb, ub;
          if (sign == 1) {
            lb = (*e);
            ub = (*e2);
          }
          else {
            lb = (*e2);
            ub = (*e);
          }
          
          CG_outputRepr *lhs = NULL;
          for (Constr_Vars_Iter cvi(lb); cvi; cvi++)
            if (cvi.curr_var() != wc) {
              CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), 
                                              assigned_on_the_fly, unin);
              coef_t coef = cvi.curr_coef();
              if (coef > 0) {
                if (coef == 1)
                  lhs = ocg->CreateMinus(lhs, v);
                else
                  lhs = ocg->CreateMinus(lhs, 
                                         ocg->CreateTimes(ocg->CreateInt(coef), v));
              }
              else { // coef < 0
                if (coef == -1)
                  lhs = ocg->CreatePlus(lhs, v);
                else
                  lhs = ocg->CreatePlus(lhs, 
                                        ocg->CreateTimes(ocg->CreateInt(-coef), v));
              }
            }
          coef_t c = lb.get_const();
          if (c > 0)
            lhs = ocg->CreateMinus(lhs, ocg->CreateInt(c));
          else if (c < 0)
            lhs = ocg->CreatePlus(lhs, ocg->CreateInt(-c));
          
          CG_outputRepr *rhs = NULL;
          for (Constr_Vars_Iter cvi(ub); cvi; cvi++)
            if (cvi.curr_var() != wc) {
              CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), 
                                              assigned_on_the_fly, unin);
              coef_t coef = cvi.curr_coef();
              if (coef > 0) {
                if (coef == 1)
                  rhs = ocg->CreatePlus(rhs, v);
                else
                  rhs = ocg->CreatePlus(rhs, 
                                        ocg->CreateTimes(ocg->CreateInt(coef), v));
              }
              else { // coef < 0
                if (coef == -1)
                  rhs = ocg->CreateMinus(rhs, v);
                else
                  rhs = ocg->CreateMinus(rhs, 
                                         ocg->CreateTimes(ocg->CreateInt(-coef), v));
              }
            }
          c = ub.get_const();
          if (c > 0)
            rhs = ocg->CreatePlus(rhs, ocg->CreateInt(c));
          else if (c < 0)
            rhs = ocg->CreateMinus(rhs, ocg->CreateInt(-c));
          
          rhs = ocg->CreateIntegerFloor(rhs, ocg->CreateInt(-ub.get_coef(wc)));
          rhs = ocg->CreateTimes(ocg->CreateInt(lb.get_coef(wc)), rhs);
          CG_outputRepr *term = ocg->CreateLE(lhs, rhs);
          result = ocg->CreateAnd(result, term);
        }
      }
    
    //debug_fprintf(stderr, "output_guard returning at bottom 0x%x\n", result); 
    return result;
  }
  
  
  CG_outputRepr *output_inequality_repr(CG_outputBuilder *ocg,
                                        const GEQ_Handle &inequality, 
                                        Variable_ID v, 
                                        const Relation &R, 
                                        const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, 
                                        const std::map<std::string, std::vector<CG_outputRepr *> > &unin,
                                        std::set<Variable_ID> excluded_floor_vars) {
    debug_fprintf(stderr, "output_inequality_repr()  v %s\n", v->name().c_str()); 

    const_cast<Relation &>(R).setup_names(); // hack
    
    coef_t a = inequality.get_coef(v);
    assert(a != 0);
    excluded_floor_vars.insert(v);

    CG_outputRepr *repr = NULL;
    for (Constr_Vars_Iter cvi(inequality); cvi; cvi++)
      if (cvi.curr_var() != v) {
        CG_outputRepr *t;
        if (cvi.curr_var()->kind() == Wildcard_Var) {
          std::pair<bool, GEQ_Handle> result = find_floor_definition(R, 
                                                                     cvi.curr_var(), 
                                                                     excluded_floor_vars);
          if (!result.first) {
            delete repr;
            throw codegen_error("Can't generate bound expression with wildcard not involved in floor definition");
          }
          try {
            t = output_inequality_repr(ocg, 
                                       result.second, 
                                       cvi.curr_var(), 
                                       R, 
                                       assigned_on_the_fly, 
                                       unin, 
                                       excluded_floor_vars);
          }
          catch (const std::exception &e) {
            delete repr;
            throw e;
          }
        }
        else
          t = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly, unin);
        
        coef_t coef = cvi.curr_coef();
        if (a > 0) {
          if (coef > 0) {
            if (coef == 1)
              repr = ocg->CreateMinus(repr, t);
            else
              repr = ocg->CreateMinus(repr, 
                                      ocg->CreateTimes(ocg->CreateInt(coef),t));
          }
          else {
            if (coef == -1)
              repr = ocg->CreatePlus(repr, t);
            else
              repr = ocg->CreatePlus(repr, 
                                     ocg->CreateTimes(ocg->CreateInt(-coef),t));
          }
        }
        else {
          if (coef > 0) {
            if (coef == 1)
              repr = ocg->CreatePlus(repr, t);
            else
              repr = ocg->CreatePlus(repr, 
                                     ocg->CreateTimes(ocg->CreateInt(coef),t));
          }
          else {
            if (coef == -1)
              repr = ocg->CreateMinus(repr, t);
            else
              repr = ocg->CreateMinus(repr, 
                                      ocg->CreateTimes(ocg->CreateInt(-coef),t));
          }
        }
      }
    coef_t c = inequality.get_const();
    if (c > 0) {
      if (a > 0)
        repr = ocg->CreateMinus(repr, ocg->CreateInt(c));
      else 
        repr = ocg->CreatePlus(repr, ocg->CreateInt(c));
    }
    else if (c < 0) {
      if (a > 0)
        repr = ocg->CreatePlus(repr, ocg->CreateInt(-c));
      else
        repr = ocg->CreateMinus(repr, ocg->CreateInt(-c));
    }    
    
    if (abs(a) == 1)
      return repr;
    else if (a > 0)
      return ocg->CreateIntegerCeil(repr, ocg->CreateInt(a));
    else // a < 0
      return ocg->CreateIntegerFloor(repr, ocg->CreateInt(-a));  
  }
  
  
  CG_outputRepr *output_upper_bound_repr(CG_outputBuilder *ocg,
                                         const GEQ_Handle &inequality, 
                                         Variable_ID v, 
                                         const Relation &R, 
                                         const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                         const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    assert(inequality.get_coef(v) < 0);
    CG_outputRepr* zero_;
    
    zero_ = output_inequality_repr(ocg, inequality, v, R, assigned_on_the_fly, 
                                   unin);
    
    if(!zero_)
      zero_ = ocg->CreateInt(0);
    
    return zero_;
  }
  
  
  CG_outputRepr *output_lower_bound_repr(CG_outputBuilder *ocg,
                                         const GEQ_Handle &inequality, 
                                         Variable_ID v, 
                                         const EQ_Handle &stride_eq, 
                                         Variable_ID wc, 
                                         const Relation &R, 
                                         const Relation &known, 
                                         const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                         const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {

    debug_fprintf(stderr, "output_lower_bound_repr()\n"); 
    assert(inequality.get_coef(v) > 0);
    CG_outputRepr* zero_;
    if (wc == NULL 
        || bound_must_hit_stride(inequality, v, stride_eq, wc, R, known)){
      zero_ = output_inequality_repr(ocg, inequality, v, R, 
                                     assigned_on_the_fly, unin);
      if(!zero_)
        zero_ = ocg->CreateInt(0);
      
      return zero_;
    }
    CG_outputRepr *strideBoundRepr = NULL;
    int sign = (stride_eq.get_coef(v)>0)?1:-1;
    for (Constr_Vars_Iter cvi(stride_eq); cvi; cvi++) {
      Variable_ID v2 = cvi.curr_var();
      if (v2 == v || v2 == wc)
        continue;
      
      CG_outputRepr *v_repr;
      if (v2->kind() == Input_Var || v2->kind() == Global_Var)
        v_repr = output_ident(ocg, R, v2, assigned_on_the_fly,unin);
      else if (v2->kind() == Wildcard_Var) {
        std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v2);
        assert(result.first);
        v_repr = output_inequality_repr(ocg, result.second, v2, R,
                                        assigned_on_the_fly, unin);
      }
      
      coef_t coef = cvi.curr_coef();
      if (sign < 0) {
        if (coef > 0) {
          if (coef == 1)
            strideBoundRepr = ocg->CreatePlus(strideBoundRepr, v_repr);
          else
            strideBoundRepr = ocg->CreatePlus(strideBoundRepr, 
                                              ocg->CreateTimes(ocg->CreateInt(coef), v_repr));
        }
        else { // coef < 0
          if (coef == -1)
            strideBoundRepr = ocg->CreateMinus(strideBoundRepr, v_repr);
          else
            strideBoundRepr = ocg->CreateMinus(strideBoundRepr, 
                                               ocg->CreateTimes(ocg->CreateInt(-coef), v_repr));
        }
      }
      else {
        if (coef > 0) {
          if (coef == 1)
            strideBoundRepr = ocg->CreateMinus(strideBoundRepr, v_repr);
          else
            strideBoundRepr = ocg->CreateMinus(strideBoundRepr, 
                                               ocg->CreateTimes(ocg->CreateInt(coef), v_repr));
        }
        else { // coef < 0
          if (coef == -1)
            strideBoundRepr = ocg->CreatePlus(strideBoundRepr, v_repr);
          else
            strideBoundRepr = ocg->CreatePlus(strideBoundRepr, 
                                              ocg->CreateTimes(ocg->CreateInt(-coef), v_repr));
        }
      }
    }  
    coef_t c = stride_eq.get_const();
    debug_fprintf(stderr, "stride eq const %lld\n", c);
    debug_fprintf(stderr, "sign %d\n", sign ); 
    if (c > 0) {
      if (sign < 0)
        strideBoundRepr = ocg->CreatePlus(strideBoundRepr, ocg->CreateInt(c));
      else
        strideBoundRepr = ocg->CreateMinus(strideBoundRepr, ocg->CreateInt(c));
    }
    else if (c < 0) {
      if (sign < 0)
        strideBoundRepr = ocg->CreateMinus(strideBoundRepr, ocg->CreateInt(-c));
      else
        strideBoundRepr = ocg->CreatePlus(strideBoundRepr, ocg->CreateInt(-c));
    }
    
    CG_outputRepr *repr = output_inequality_repr(ocg, inequality, v, R, 
                                                 assigned_on_the_fly, unin);
    //debug_fprintf(stderr, "inequality repr %p\n", repr); 
    CG_outputRepr *repr2 = ocg->CreateCopy(repr);
    
    debug_fprintf(stderr, "stride_eq.get_coef(wc) %lld\n", stride_eq.get_coef(wc));
    debug_fprintf(stderr, "repr + mod( strideBoundRepr - repr, %lld )\n", stride_eq.get_coef(wc));

    repr = ocg->CreatePlus(repr2, 
                           ocg->CreateIntegerMod(ocg->CreateMinus(strideBoundRepr, repr), 
                                                 ocg->CreateInt(abs(stride_eq.get_coef(wc)))));
    
    return repr;
  }
  
  
  CG_outputRepr *output_loop(CG_outputBuilder *ocg,
                             const Relation &R, 
                             int level, 
                             const Relation &known, 
                             const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                             const std::map<std::string, std::vector<CG_outputRepr *> > &unin) {
    
    debug_fprintf(stderr, "CG_utils.cc output_loop()\n"); 
    std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(R, const_cast<Relation &>(R).set_var(level));
    debug_fprintf(stderr, "found stride\n"); 
    coef_t s = 1, c = 0;
    if (result.second != NULL) {
      assert(abs(result.first.get_coef(const_cast<Relation &>(R).set_var(level))) == 1);
      s = abs(result.first.get_coef(result.second));
      c = result.first.get_const();
    }
    
    std::vector<CG_outputRepr *> lbList, ubList;  
    try {
      
      coef_t const_lb = negInfinity, const_ub = posInfinity;
      
      for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); 
           e; 
           e++) {
        coef_t coef = (*e).get_coef(const_cast<Relation &>(R).set_var(level));
        if (coef > 0) {
          CG_outputRepr *repr = output_lower_bound_repr(ocg, *e, const_cast<Relation &>(R).set_var(level), result.first, result.second, R, known, assigned_on_the_fly, unin);
          if (repr == nullptr)
            repr = ocg->CreateInt(0);
          lbList.push_back(repr);
          if ((*e).is_const(const_cast<Relation &>(R).set_var(level))) {
            coef_t L, m;
            L = -((*e).get_const());
            m = (*e).get_coef(const_cast<Relation &>(R).set_var(level));
            L = L - c * m;
            m = m * s;
            if (L > 0)
              set_max(const_lb, c + s * (L + m - 1) / m);
            else
              set_max(const_lb, c + s * L / m);
          }
        }
        else if (coef < 0) {
          CG_outputRepr *repr = output_upper_bound_repr(ocg, 
                                                        *e, 
                                                        const_cast<Relation &>(R).set_var(level), 
                                                        R, 
                                                        assigned_on_the_fly, 
                                                        unin);
          if (repr == nullptr)
            repr = ocg->CreateInt(0);
          ubList.push_back(repr);
          
          if ((*e).is_const(const_cast<Relation &>(R).set_var(level))) {
            coef_t cst = (*e).get_const();
            coef_t cef = -(*e).get_coef(const_cast<Relation &>(R).set_var(level));
            if (cst >= 0)
              set_min(const_ub, cst/cef);
            else
              set_min(const_ub, (cst - cef + 1)/cef);
          }
        }
      }
      
      if(fillInBounds && lbList.size() == 1 && const_lb != negInfinity)
        lowerBoundForLevel = const_lb;
      
      if(fillInBounds && const_ub != posInfinity) {
        upperBoundForLevel = const_ub;
      }
      if (lbList.size() == 0)
        throw codegen_error(
          "missing lower bound at loop level " + to_string(level));
      if (ubList.size() == 0)
        throw codegen_error(
          "missing upper bound at loop level " + to_string(level));
    }
    catch (const std::exception &e) {
      for (int i = 0; i < lbList.size(); i++)
        delete lbList[i];
      for (int i = 0; i < ubList.size(); i++)
        delete ubList[i];
      throw e;
    }
    
    CG_outputRepr *lbRepr = NULL;
    if (lbList.size() > 1) {
      debug_fprintf(stderr, "CG_utils.cc output_loop() createInvoke( max )\n"); 
      lbRepr = ocg->CreateInvoke("max", lbList);
    }
    else { // (lbList.size() == 1)
      lbRepr = lbList[0];
    }

    CG_outputRepr *ubRepr = NULL;
    if (ubList.size() > 1) {
      debug_fprintf(stderr, "CG_utils.cc output_loop() createInvoke( min )\n"); 
      ubRepr = ocg->CreateInvoke("min", ubList);
    }
    else { // (ubList.size() == 1)
      ubRepr = ubList[0];
    }
    
    CG_outputRepr *stRepr;
    if (result.second == NULL)
      stRepr = ocg->CreateInt(1);
    else
      stRepr = ocg->CreateInt(abs(result.first.get_coef(result.second)));
    CG_outputRepr *indexRepr = output_ident(ocg, 
                                            R, 
                                            const_cast<Relation &>(R).set_var(level), 
                                            assigned_on_the_fly, 
                                            unin);

    return ocg->CreateInductive(indexRepr, lbRepr, ubRepr, stRepr);
  }
  
  
//
// parameter f_root is inside f_exists, not the other way around.
// return replicated variable in new relation, with all cascaded floor definitions
// using wildcards defined in the same way as in the original relation.
//
  Variable_ID replicate_floor_definition(const Relation &R, 
                                         const Variable_ID floor_var,
                                         Relation &r, 
                                         F_Exists *f_exists, 
                                         F_And *f_root,
                                         std::map<Variable_ID, Variable_ID> &exists_mapping) {
    //Anand: commenting the assertion out 8/4/2013
    //assert(R.n_out() == 0 && r.n_out() == 0 && R.n_inp() == r.n_inp());
    bool is_mapping = false;
    if (r.n_out() > 0)
      is_mapping = true;

    std::set<Variable_ID> excluded_floor_vars;
    std::stack<Variable_ID> to_fill;
    to_fill.push(floor_var);
    
    while (!to_fill.empty()) {
      Variable_ID v = to_fill.top();
      to_fill.pop();
      if (excluded_floor_vars.find(v) != excluded_floor_vars.end())
        continue;
      
      std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v,
                                                                 excluded_floor_vars);
      assert(result.first);
      excluded_floor_vars.insert(v);

      // The pair of added constraints
      GEQ_Handle h1 = f_root->add_GEQ(); // h1 => m >= av + x : result.second
      GEQ_Handle h2 = f_root->add_GEQ(); // h2 => m - (a - 1) <= av + x
      for (Constr_Vars_Iter cvi(result.second); cvi; cvi++) {
        Variable_ID v2 = cvi.curr_var();
        switch  (v2->kind()) {
        case Input_Var: {
          int pos = v2->get_position();
           if (!is_mapping) {
            h1.update_coef(r.input_var(pos), cvi.curr_coef());
            h2.update_coef(r.input_var(pos), -cvi.curr_coef());
          }
          else {
            h1.update_coef(r.output_var(pos), cvi.curr_coef());
            h2.update_coef(r.output_var(pos), -cvi.curr_coef());
          }
          break;
        }
        case Wildcard_Var: {
          std::map<Variable_ID, Variable_ID>::iterator p=exists_mapping.find(v2);
          Variable_ID v3;
          if (p == exists_mapping.end()) {
            v3 = f_exists->declare();
            exists_mapping[v2] = v3;
          }
          else
            v3 = p->second;
          h1.update_coef(v3, cvi.curr_coef());
          h2.update_coef(v3, -cvi.curr_coef());
          if (v2 != v)
            to_fill.push(v2);
          break;
        }
        case Global_Var: {
          Global_Var_ID g = v2->get_global_var();
          Variable_ID v3;
          if (g->arity() == 0)
            v3 = r.get_local(g);
          else
            v3 = r.get_local(g, v2->function_of());
          h1.update_coef(v3, cvi.curr_coef());
          h2.update_coef(v3, -cvi.curr_coef());
          break;
        }
        default:
          assert(false);
        }
      }
      h1.update_const(result.second.get_const());
      h2.update_const(
        -result.second.get_const()
        - result.second.get_coef(v) - 1);
    }

    if (floor_var->kind() == Input_Var)
      return r.input_var(floor_var->get_position());
    else if (floor_var->kind() == Wildcard_Var)
      return exists_mapping[floor_var];
    else
      assert(false);
  }
  
  
//
// pick one guard condition from relation. it can involve multiple
// constraints when involving wildcards, as long as its complement
// is a single conjunct.
//
  Relation pick_one_guard(const Relation &R, int level) {
    //debug_fprintf(stderr, "pick_one_guard()\n"); 
    assert(R.n_out()==0);
    
    Relation r = Relation::True(R.n_set());
    
    for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); 
         e; 
         e++)
      if (!(*e).has_wildcards()) {
        r.and_with_GEQ(*e);
        r.simplify();
        r.copy_names(R);
        r.setup_names();
        return r;
      }
    
    for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); 
         e; 
         e++)
      if (!(*e).has_wildcards()) {
        r.and_with_GEQ(*e);
        r.simplify();
        r.copy_names(R);
        r.setup_names();
        return r;
      }
    
    for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); 
         e; 
         e++)
      if ((*e).has_wildcards()) {
        int num_wildcard = 0;
        int max_level = 0;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
          switch (cvi.curr_var()->kind()) {
          case Wildcard_Var:
            num_wildcard++;
            break;
          case Input_Var:
            if (cvi.curr_var()->get_position() > max_level)
              max_level = cvi.curr_var()->get_position();
            break;
          default:
            ;
          }
        
        if (num_wildcard == 1 && max_level != level-1) {
          r.and_with_EQ(*e);
          r.simplify();
          r.copy_names(R);
          r.setup_names();
          return r;
        }
      }
    
    for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); 
         e; 
         e++)
      if ((*e).has_wildcards()) {
        int num_wildcard = 0;
        int max_level = 0;
        bool direction;
        Variable_ID wc;
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
          switch (cvi.curr_var()->kind()) {
          case Wildcard_Var:
            num_wildcard++;
            wc = cvi.curr_var();
            direction = cvi.curr_coef() > 0;
            break;
          case Input_Var:
            if (cvi.curr_var()->get_position() > max_level)
              max_level = cvi.curr_var()->get_position();
            break;
          default:
            ;
          }
        
        if (num_wildcard == 1 && max_level != level-1) {
          // find the pairing inequality
          GEQ_Iterator e2 = e;
          e2++;
          for ( ; e2; e2++) {
            int num_wildcard2 = 0;
            int max_level2 = 0;
            bool direction2;
            Variable_ID wc2;
            for (Constr_Vars_Iter cvi(*e2); cvi; cvi++)
              switch (cvi.curr_var()->kind()) {
              case Wildcard_Var:
                num_wildcard2++;
                wc2 = cvi.curr_var();
                direction2 = cvi.curr_coef() > 0;
                break;
              case Input_Var:
                if (cvi.curr_var()->get_position() > max_level2)
                  max_level2 = cvi.curr_var()->get_position();
                break;
              default:
                ;
              }
            
            if (num_wildcard2 == 1 
                && max_level2 != level-1 
                && wc2 == wc 
                && direction2 == not direction) {
              F_Exists *f_exists = r.and_with_and()->add_exists();
              Variable_ID wc3 = f_exists->declare();
              F_And *f_root = f_exists->add_and();
              GEQ_Handle h = f_root->add_GEQ();
              for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
                switch (cvi.curr_var()->kind()) {
                case Wildcard_Var:
                  h.update_coef(wc3, cvi.curr_coef());
                  break;
                case Input_Var:
                  h.update_coef(r.input_var(cvi.curr_var()->get_position()), 
                                cvi.curr_coef());
                  break;
                case Global_Var: {
                  Global_Var_ID g = cvi.curr_var()->get_global_var();
                  Variable_ID v;
                  if (g->arity() == 0)
                    v = r.get_local(g);
                  else
                    v = r.get_local(g, cvi.curr_var()->function_of());
                  h.update_coef(v, cvi.curr_coef());
                  break;
                }
                default:
                  assert(false);
                }
              }
              h.update_const((*e).get_const());
              
              h = f_root->add_GEQ();
              for (Constr_Vars_Iter cvi(*e2); cvi; cvi++) {
                switch (cvi.curr_var()->kind()) {
                case Wildcard_Var:
                  h.update_coef(wc3, cvi.curr_coef());
                  break;
                case Input_Var:
                  h.update_coef(r.input_var(cvi.curr_var()->get_position()), 
                                cvi.curr_coef());
                  break;
                case Global_Var: {
                  Global_Var_ID g = cvi.curr_var()->get_global_var();
                  Variable_ID v;
                  if (g->arity() == 0)
                    v = r.get_local(g);
                  else
                    v = r.get_local(g, cvi.curr_var()->function_of());
                  h.update_coef(v, cvi.curr_coef());
                  break;
                }
                default:
                  assert(false);
                }
              }
              h.update_const((*e2).get_const());
              
              r.simplify();
              r.copy_names(R);
              r.setup_names();
              return r;
            }
          }
        }
      }
  }
  
  

  



  std::string print_to_iegen_string(Relation &R) {
    std::string s = "";
    for (GEQ_Iterator e(R.single_conjunct()->GEQs()); e; e++) {
      if (s != "")
        s += " && ";
      s += (*e).print_to_string();
    }
    
    for (EQ_Iterator e(R.single_conjunct()->EQs()); e; e++) {
      if (s != "")
        s += " && ";
      s += (*e).print_to_string();
    }
    return s;
  }



//
// Check if a set/input var is projected out of a inequality by a global variable with arity > 0
//
  Relation checkAndRestoreIfProjectedByGlobal(const Relation &R1,
                                              const Relation &R2, Variable_ID v) {
    
    Relation to_return(R2.n_set());
    
//1. detect if a variable is not involved in a GEQ.
    for (DNF_Iterator di(copy(R1).query_DNF()); di; di++)
      for (GEQ_Iterator ci = (*di)->GEQs(); ci; ci++)
        if ((*ci).get_coef(v) != 0)
          return copy(R2);
    
    bool found_global_eq = false;
    Global_Var_ID g1;
    EQ_Handle e;
//2. detect if its involved in an eq involving a global with arity >= 1
    for (DNF_Iterator di(copy(R1).query_DNF()); di; di++)
      for (EQ_Iterator ci = (*di)->EQs(); ci; ci++)
        if ((*ci).get_coef(v) != 0)
          for (Constr_Vars_Iter cvi(*ci); cvi; cvi++) {
            v = cvi.curr_var();
            if (v->kind() == Global_Var)
              if (v->get_global_var()->arity() > 0) {
                found_global_eq = true;
                g1 = v->get_global_var();
                e = (*ci);
              }
          }
    
    if (!found_global_eq)
      return copy(R2);
    
//3. check if the global is involved in a geq
    
    bool found_global_lb = false;
    bool found_global_ub = false;
    
    std::vector<GEQ_Handle> ub;
    std::vector<GEQ_Handle> lb;
    for (DNF_Iterator di(copy(R1).query_DNF()); di; di++)
      for (GEQ_Iterator ci = (*di)->GEQs(); ci; ci++)
        for (Constr_Vars_Iter cvi(*ci); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          if (v->kind() == Global_Var)
            if (v->get_global_var()->arity() > 0
                && v->get_global_var() == g1) {
              if (cvi.curr_coef() > 0) {
                found_global_lb = true;
                
                lb.push_back(*ci);
              } 
              else if (cvi.curr_coef() < 0) {
                found_global_ub = true;
                
                ub.push_back(*ci);
              }
              
            }
        }
    
    if (!found_global_lb || !found_global_ub)
      return copy(R2);
    
//
    
    F_And *root = to_return.add_and();
    
    for (int i = 0; i < lb.size(); i++) {
      GEQ_Handle lower = root->add_GEQ();
      for (Constr_Vars_Iter cvi(lb[i]); cvi; cvi++) {
        Variable_ID v2 = cvi.curr_var();
        
        if (v2->kind() == Wildcard_Var)
          return copy(R2);
        if (v2->kind() != Global_Var
            || (v2->kind() == Global_Var && v2->get_global_var() != g1)) {
          if (v2->kind() != Global_Var)
            lower.update_coef(v2, cvi.curr_coef());
          else {
            Variable_ID lb1 = to_return.get_local(v2->get_global_var(),
                                                  Input_Tuple);
            lower.update_coef(lb1, cvi.curr_coef());
          }
          
        } 
        else
          lower.update_coef(v, cvi.curr_coef());
      }
      
      lower.update_const(lb[i].get_const());
    }
    
    for (int i = 0; i < ub.size(); i++) {
      GEQ_Handle upper = root->add_GEQ();
      for (Constr_Vars_Iter cvi(ub[i]); cvi; cvi++) {
        Variable_ID v2 = cvi.curr_var();
        
        if (v2->kind() == Wildcard_Var)
          return copy(R2);
        if (v2->kind() != Global_Var
            || (v2->kind() == Global_Var && v2->get_global_var() != g1)) {
          if (v2->kind() != Global_Var)
            upper.update_coef(v2, cvi.curr_coef());
          else {
            Variable_ID ub1 = to_return.get_local(v2->get_global_var(),
                                                  Input_Tuple);
            upper.update_coef(ub1, cvi.curr_coef());
          }
          
        } 
        else
          upper.update_coef(v, cvi.curr_coef());
      }
      
      upper.update_const(ub[i].get_const());
    }
    
//Relation to_return2 = copy(R2);
    
    for (EQ_Iterator g(const_cast<Relation &>(R2).single_conjunct()->EQs()); g;
         g++)
      to_return.and_with_EQ(*g);
    
    for (GEQ_Iterator g(const_cast<Relation &>(R2).single_conjunct()->GEQs());
         g; g++) {
      bool found_glob = false;
      for (Constr_Vars_Iter cvi(*g); cvi; cvi++) {
        Variable_ID v2 = cvi.curr_var();
        if (v2->kind() == Global_Var && v2->get_global_var() == g1) {
          found_glob = true;
          break;
        }
      }
      if (!found_glob)
        to_return.and_with_GEQ(*g);
      
    }
//to_return.and_with_EQ(e);
    
    to_return.copy_names(copy(R2));
    to_return.setup_names();
    to_return.finalize();
    return to_return;
  }

}
