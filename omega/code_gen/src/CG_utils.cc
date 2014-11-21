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

int checkLoopLevel;
int stmtForLoopCheck;
int upperBoundForLevel;
int lowerBoundForLevel;
bool fillInBounds;

//trick to static init checkLoopLevel to 0
class JunkStaticInit{ public: JunkStaticInit(){ checkLoopLevel=0; fillInBounds=false;} };
static JunkStaticInit junkInitInstance__;




namespace {

Relation find_best_guard(const Relation &R, const BoolSet<> &active, const std::map<int, Relation> &guards) {
  std::pair<int, int> best_cost = std::make_pair(0, 0);
  Relation best_cond = Relation::True(R.n_set());
  
  Relation r = copy(R);
  int max_iter_count = 2*(r.single_conjunct()->n_EQs()) + r.single_conjunct()->n_GEQs();
  int iter_count = 0;
  while (!r.is_obvious_tautology()) {
    std::pair<int, int> cost = std::make_pair(0, 0);        
    Relation cond = pick_one_guard(r);
    Relation complement_cond = Complement(copy(cond));
    complement_cond.simplify();    
    for (BoolSet<>::const_iterator i = active.begin(); i != active.end(); i++) {
      std::map<int, Relation>::const_iterator j = guards.find(*i);
      if (j == guards.end())
        continue;
      if (Must_Be_Subset(copy(j->second), copy(cond)))
        cost.first++;
      else if (Must_Be_Subset(copy(j->second), copy(complement_cond)))
        cost.second++;
    }
    if (cost > best_cost) {
      best_cost = cost;
      best_cond = copy(cond);
    }
    r = Gist(r, cond, 1);

    if (iter_count > max_iter_count)
      throw codegen_error("guard condition too complex to handle");

    iter_count++;
  }

  return best_cond;
}


Relation find_best_guard(const Relation &R, const std::vector<CG_loop *> &loops, int start, int end) {
  std::pair<int, int> best_cost = std::make_pair(0, 0);
  Relation best_cond = Relation::True(R.n_set());
  
  Relation r = copy(R);
  int max_iter_count = 2*(r.single_conjunct()->n_EQs()) + r.single_conjunct()->n_GEQs();
  int iter_count = 0;
  while (!r.is_obvious_tautology()) {
    std::pair<int, int> cost = std::make_pair(0, 0);
    Relation cond = pick_one_guard(r);
    int i = start;
    for ( ; i < end; i++) {
      if (Must_Be_Subset(copy(loops[i]->guard_), copy(cond)))
        cost.first++;
      else
        break;
    }
    Relation complement_cond = Complement(copy(cond));
    complement_cond.simplify();
    for (int j = i; j < end; j++)
      if (Must_Be_Subset(copy(loops[j]->guard_), copy(complement_cond)))
        cost.second++;
      else
        break;
    
    if (cost > best_cost) {
      best_cost = cost;
      best_cond = copy(cond);
    }
    r = Gist(r, cond, 1);

    if (iter_count > max_iter_count)
      throw codegen_error("guard condition too complex to handle");

    iter_count++;
  }

  return best_cond;
}

}

bool bound_must_hit_stride(const GEQ_Handle &inequality, Variable_ID v, const EQ_Handle &stride_eq, Variable_ID wc, const Relation &bounds, const Relation &known) {
  assert(inequality.get_coef(v) != 0 && abs(stride_eq.get_coef(v)) == 1 && wc->kind() == Wildcard_Var && abs(stride_eq.get_coef(wc)) > 1);

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
      Variable_ID v1 = replicate_floor_definition(bounds, v, r1, f_exists1, f_root1, exists_mapping1);
      Variable_ID v2 = replicate_floor_definition(bounds, v, r2, f_exists2, f_root2, exists_mapping2);
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
          Variable_ID v3 = replicate_floor_definition(bounds, v, r3, f_exists3, f_root3, exists_mapping3);
          h3.update_coef(v3, cvi.curr_coef());
        }
        break;
      default:
        assert(false);
      }
    }
    h3.update_const(stride_eq.get_const());
    r3.simplify();
    
    if (Gist(r3, Intersection(r2, all_known), 1).is_obvious_tautology())
      return true;
    else
      return false;
  }
  else    
    return false;
}


//
// output variable by its name, however if this variable need to be substituted,
// return the substitution.
//
CG_outputRepr *output_ident(CG_outputBuilder *ocg, const Relation &R, Variable_ID v, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  const_cast<Relation &>(R).setup_names(); // hack
  
  if (v->kind() == Input_Var) {
    int pos = v->get_position();
    if (assigned_on_the_fly[pos-1].first != NULL)
      return assigned_on_the_fly[pos-1].first->clone();
    else
      return ocg->CreateIdent(v->name());
  }
  else if (v->kind() == Global_Var) {
    if (v->get_global_var()->arity() == 0)
      return ocg->CreateIdent(v->name());
    else {
      /* This should be improved to take into account the possible elimination
         of the set variables. */
      int arity = v->get_global_var()->arity();
      std::vector<CG_outputRepr *> argList;
      for(int i = 1; i <= arity; i++)
        argList.push_back(ocg->CreateIdent(const_cast<Relation &>(R).input_var(i)->name()));
      CG_outputRepr *call = ocg->CreateInvoke(v->get_global_var()->base_name(), argList);
      return call;
    }
  }
  else
    assert(false);
}


//
// return pair<if condition, <assignment rhs, assignment cost> >
//
std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> > output_assignment(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  Variable_ID v = const_cast<Relation &>(R).set_var(level);
  Conjunct *c = const_cast<Relation &>(R).single_conjunct();

  std::pair<EQ_Handle, int> result = find_simplest_assignment(R, v, assigned_on_the_fly);

  if (result.second == INT_MAX)
    return std::make_pair(static_cast<CG_outputRepr *>(NULL), std::make_pair(static_cast<CG_outputRepr *>(NULL), INT_MAX));
  
  CG_outputRepr *if_repr = NULL;
  CG_outputRepr *assign_repr = NULL;
  // check whether to generate if-conditions from equality constraints
  if (abs(result.first.get_coef(v)) != 1) {
    Relation r(R.n_set());
    F_Exists *f_exists = r.add_and()->add_exists();
    F_And *f_root = f_exists->add_and();
    std::map<Variable_ID, Variable_ID> exists_mapping;
    exists_mapping[v] = f_exists->declare();

    EQ_Handle h = f_root->add_EQ();
    for (Constr_Vars_Iter cvi(result.first); cvi; cvi++)
      switch (cvi.curr_var()->kind()) {
      case Input_Var: {
        if (cvi.curr_var() == v)
          h.update_coef(exists_mapping[v], cvi.curr_coef());
        else
          h.update_coef(r.set_var(cvi.curr_var()->get_position()), cvi.curr_coef());
        break;
      }
      case Global_Var: {            
        Global_Var_ID g = cvi.curr_var()->get_global_var();
        Variable_ID v2;
        if (g->arity() == 0)
          v2 = r.get_local(g);
        else
          v2 = r.get_local(g, cvi.curr_var()->function_of());
        h.update_coef(v2, cvi.curr_coef());
        break;
      }
      case Wildcard_Var: {
        std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find(cvi.curr_var());
        Variable_ID v2;
        if (p == exists_mapping.end()) {
          v2 = f_exists->declare();
          exists_mapping[cvi.curr_var()] = v2;
        }
        else
          v2 = p->second;
        h.update_coef(v2, cvi.curr_coef());
        break;
      }
      default:
        assert(0);
      }
    h.update_const(result.first.get_const());
      
    for (EQ_Iterator e(c->EQs()); e; e++)
      if (!((*e) == result.first)) {
        EQ_Handle h = f_root->add_EQ();
        for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
          switch (cvi.curr_var()->kind()) {
          case Input_Var: {
            assert(cvi.curr_var() != v);
            h.update_coef(r.set_var(cvi.curr_var()->get_position()), cvi.curr_coef());
            break;
          }
          case Global_Var: {            
            Global_Var_ID g = cvi.curr_var()->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0)
              v2 = r.get_local(g);
            else
              v2 = r.get_local(g, cvi.curr_var()->function_of());
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          case Wildcard_Var: {
            std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find(cvi.curr_var());
            Variable_ID v2;
            if (p == exists_mapping.end()) {
              v2 = f_exists->declare();
              exists_mapping[cvi.curr_var()] = v2;
            }
            else
              v2 = p->second;
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          default:
            assert(0);
          }
        h.update_const((*e).get_const());
      }
        
    for (GEQ_Iterator e(c->GEQs()); e; e++) {
      GEQ_Handle h = f_root->add_GEQ();
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
        switch (cvi.curr_var()->kind()) {
        case Input_Var: {
          h.update_coef(r.set_var(cvi.curr_var()->get_position()), cvi.curr_coef());
          break;
        }
        case Global_Var: {            
          Global_Var_ID g = cvi.curr_var()->get_global_var();
          Variable_ID v2;
          if (g->arity() == 0)
            v2 = r.get_local(g);
          else
            v2 = r.get_local(g, cvi.curr_var()->function_of());
          h.update_coef(v2, cvi.curr_coef());
          break;
        }
        case Wildcard_Var: {
          std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find(cvi.curr_var());
          Variable_ID v2;
          if (p == exists_mapping.end()) {
            v2 = f_exists->declare();
            exists_mapping[cvi.curr_var()] = v2;
          }
          else
            v2 = p->second;
          h.update_coef(v2, cvi.curr_coef());
          break;
        }
        default:
          assert(0);
        }
      h.update_const((*e).get_const());
    }
      
    r.simplify();
    if (!Gist(r, copy(known), 1).is_obvious_tautology()) {
      CG_outputRepr *lhs = output_substitution_repr(ocg, result.first, v, false, R, assigned_on_the_fly);  
      if_repr = ocg->CreateEQ(ocg->CreateIntegerMod(lhs->clone(), ocg->CreateInt(abs(result.first.get_coef(v)))), ocg->CreateInt(0));
      assign_repr = ocg->CreateDivide(lhs, ocg->CreateInt(abs(result.first.get_coef(v))));
    }
    else
      assign_repr = output_substitution_repr(ocg, result.first, v, true, R, assigned_on_the_fly);
  }
  else
    assign_repr = output_substitution_repr(ocg, result.first, v, true, R, assigned_on_the_fly);

  if (assign_repr == NULL)
    assign_repr = ocg->CreateInt(0);
  
  return std::make_pair(if_repr, std::make_pair(assign_repr, result.second));
}


//
// return NULL if 0
//
CG_outputRepr *output_substitution_repr(CG_outputBuilder *ocg, const EQ_Handle &equality, Variable_ID v, bool apply_v_coef, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  const_cast<Relation &>(R).setup_names(); // hack
  
  coef_t a = equality.get_coef(v);
  assert(a != 0);
  
  CG_outputRepr *repr = NULL;
  for (Constr_Vars_Iter cvi(equality); cvi; cvi++)
    if (cvi.curr_var() != v) {
      CG_outputRepr *t;
      if (cvi.curr_var()->kind() == Wildcard_Var) {
        std::pair<bool, GEQ_Handle> result = find_floor_definition(R, cvi.curr_var());
        if (!result.first) {
          delete repr;
          throw codegen_error("can't output non floor defined wildcard");
        }
        t = output_inequality_repr(ocg, result.second, cvi.curr_var(), R, assigned_on_the_fly);
      }
      else
        t = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
      coef_t coef = cvi.curr_coef();

      if (a > 0) {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr, ocg->CreateTimes(ocg->CreateInt(coef), t));
        }
        else { // coef < 0
          if (coef == -1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr, ocg->CreateTimes(ocg->CreateInt(-coef), t));
        }          
      }
      else {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr, ocg->CreateTimes(ocg->CreateInt(coef), t));
        }
        else { // coef < 0
          if (coef == -1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr, ocg->CreateTimes(ocg->CreateInt(-coef), t));
        }        
      }
    }
  
  int c = equality.get_const();
  if (a > 0) {
    if (c > 0)
      repr = ocg->CreateMinus(repr, ocg->CreateInt(c));
    else if (c < 0)
      repr = ocg->CreatePlus(repr, ocg->CreateInt(-c));
  }
  else {
    if (c > 0)
      repr = ocg->CreatePlus(repr, ocg->CreateInt(c));
    else if (c < 0)
      repr = ocg->CreateMinus(repr, ocg->CreateInt(-c));
  }
    
  if (apply_v_coef && abs(a) != 1)
    repr = ocg->CreateDivide(repr, ocg->CreateInt(abs(a)));

  return repr;
}


//
// original Substitutions class from omega can't handle integer
// division, this is new way.
//
std::vector<CG_outputRepr*> output_substitutions(CG_outputBuilder *ocg, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  std::vector<CG_outputRepr *> subs;
    
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
      repr = output_substitution_repr(ocg, result.first, v, true, r, assigned_on_the_fly);
    else {
      std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v);
      if (result.first)
        try {
          repr = output_inequality_repr(ocg, result.second, v, R, assigned_on_the_fly);
        }
        catch (const codegen_error &) {
        }
    }

    subs.push_back(repr);
  }

  return subs;
}
    

//
// handle floor definition wildcards in equality, the second in returned pair
// is the cost.
//
std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R, Variable_ID v, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  Conjunct *c = const_cast<Relation &>(R).single_conjunct();

  int min_cost = INT_MAX;
  EQ_Handle eq;
  for (EQ_Iterator e(c->EQs()); e; e++)
    if (!(*e).has_wildcards() && (*e).get_coef(v) != 0) {
      int cost = 0;

      if (abs((*e).get_coef(v)) != 1)
        cost += 4;  // divide cost

      int num_var = 0;
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
        if (cvi.curr_var() != v) {
          num_var++;
          if (abs(cvi.curr_coef()) != 1)
            cost += 2;  // multiply cost
          if (cvi.curr_var()->kind() == Global_Var && cvi.curr_var()->get_global_var()->arity() > 0)
            cost += 10;  // function cost
          else if (cvi.curr_var()->kind() == Input_Var &&
                   assigned_on_the_fly.size() >= cvi.curr_var()->get_position() &&
                   assigned_on_the_fly[cvi.curr_var()->get_position()-1].first != NULL)
            cost += assigned_on_the_fly[cvi.curr_var()->get_position()-1].second;  // substitution cost on record
        }
      if ((*e).get_const() != 0)
        num_var++;
      if (num_var > 1)
        cost += num_var - 1; // addition cost

      if (cost < min_cost) {
        min_cost = cost;
        eq = *e;
      }
    }
    
  if (min_cost < INT_MAX)
    return std::make_pair(eq, min_cost);

  for (EQ_Iterator e(c->EQs()); e; e++)
    if ((*e).has_wildcards() && (*e).get_coef(v) != 0) {
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
          else if (cvi.curr_var()->kind() == Global_Var && cvi.curr_var()->get_global_var()->arity() > 0)
            cost += 20;  // function cost
          else if (cvi.curr_var()->kind() == Input_Var &&
                   assigned_on_the_fly.size() >= cvi.curr_var()->get_position() &&
                   assigned_on_the_fly[cvi.curr_var()->get_position()-1].first != NULL)
            cost += assigned_on_the_fly[cvi.curr_var()->get_position()-1].second;  // substitution cost on record
        }
      if ((*e).get_const() != 0)
        num_var++;
      if (num_var > 1)
        cost += num_var - 1; // addition cost

      if (cost < min_cost) {
        min_cost = cost;
        eq = *e;
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
std::pair<bool, GEQ_Handle> find_floor_definition(const Relation &R, Variable_ID v, std::set<Variable_ID> excluded_floor_vars) {
  Conjunct *c = const_cast<Relation &>(R).single_conjunct();

  excluded_floor_vars.insert(v);
  for (GEQ_Iterator e = c->GEQs(); e; e++) {
    coef_t a = (*e).get_coef(v);
    if (a >= -1)
      continue;
    a = -a;
    
    bool interested = true;
    for (std::set<Variable_ID>::const_iterator i = excluded_floor_vars.begin(); i != excluded_floor_vars.end(); i++)
      if ((*i) != v && (*e).get_coef(*i) != 0) {
        interested = false;
        break;
      }
    if (!interested)
      continue;

    // check if any wildcard is floor defined
    bool has_undefined_wc = false;
    for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) 
      if (excluded_floor_vars.find(cvi.curr_var()) == excluded_floor_vars.end()) {
        std::pair<bool, GEQ_Handle> result = find_floor_definition(R, cvi.curr_var(), excluded_floor_vars);
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
std::pair<EQ_Handle, Variable_ID> find_simplest_stride(const Relation &R, Variable_ID v) {
  int best_num_var = INT_MAX;
  coef_t best_coef;
  EQ_Handle best_eq;
  Variable_ID best_stride_wc;
  for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); e; e++)
    if ((*e).has_wildcards() && (*e).get_coef(v) != 0) {
      bool is_stride = true;
      bool found_free = false;
      int num_var = 0;
      int num_floor = 0;
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
            if (result.first)
              num_floor++;
            else 
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
            
//
// convert relation to if-condition
//
CG_outputRepr *output_guard(CG_outputBuilder *ocg, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  assert(R.n_out()==0);

  CG_outputRepr *result = NULL;
  Conjunct *c = const_cast<Relation &>(R).single_conjunct();

  // e.g. 4i=5*j
  for (EQ_Iterator e = c->EQs(); e; e++)
    if (!(*e).has_wildcards()) {
      CG_outputRepr *lhs = NULL;
      CG_outputRepr *rhs = NULL;
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
        CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
        coef_t coef = cvi.curr_coef();
        if (coef > 0) {
          if (coef == 1)
            lhs = ocg->CreatePlus(lhs, v);
          else
            lhs = ocg->CreatePlus(lhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
        }
        else { // coef < 0
          if (coef == -1)
            rhs = ocg->CreatePlus(rhs, v);
          else
            rhs = ocg->CreatePlus(rhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
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
      CG_outputRepr *lhs = NULL;
      CG_outputRepr *rhs = NULL;
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
        CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
        coef_t coef = cvi.curr_coef();
        if (coef > 0) {
          if (coef == 1)
            lhs = ocg->CreatePlus(lhs, v);
          else
            lhs = ocg->CreatePlus(lhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
        }
        else { // coef < 0
          if (coef == -1)
            rhs = ocg->CreatePlus(rhs, v);
          else
            rhs = ocg->CreatePlus(rhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
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
      result = ocg->CreateAnd(result, term);
    }

  // e.g. 4i=5j+4alpha
  for (EQ_Iterator e = c->EQs(); e; e++)
    if ((*e).has_wildcards()) {
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
        throw codegen_error("Can't generate equality condition with multiple wildcards");
      }
      int sign = (num_positive>=num_negative)?1:-1;
      
      CG_outputRepr *lhs = NULL;
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
        if (cvi.curr_var() != wc) {
          CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
          coef_t coef = cvi.curr_coef();
          if (sign == 1) {
            if (coef > 0) {
              if (coef == 1)
                lhs = ocg->CreatePlus(lhs, v);
              else
                lhs = ocg->CreatePlus(lhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
            }
            else { // coef < 0
              if (coef == -1)
                lhs = ocg->CreateMinus(lhs, v);
              else
                lhs = ocg->CreateMinus(lhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
            }
          }
          else {
            if (coef > 0) {
              if (coef == 1)
                lhs = ocg->CreateMinus(lhs, v);
              else
                lhs = ocg->CreateMinus(lhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
            }
            else { // coef < 0
              if (coef == -1)
                lhs = ocg->CreatePlus(lhs, v);
              else
                lhs = ocg->CreatePlus(lhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
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

      lhs = ocg->CreateIntegerMod(lhs, ocg->CreateInt(abs((*e).get_coef(wc))));
      CG_outputRepr *term = ocg->CreateEQ(lhs, ocg->CreateInt(0));
      result = ocg->CreateAnd(result, term);
    }

  // e.g. 4alpha<=i<=5alpha
  for (GEQ_Iterator e = c->GEQs(); e; e++)
    if ((*e).has_wildcards()) {
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
        throw codegen_error("Can't generate multiple wildcard GEQ guards right now");
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
            CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
            coef_t coef = cvi.curr_coef();
            if (coef > 0) {
              if (coef == 1)
                lhs = ocg->CreateMinus(lhs, v);
              else
                lhs = ocg->CreateMinus(lhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
            }
            else { // coef < 0
              if (coef == -1)
                lhs = ocg->CreatePlus(lhs, v);
              else
                lhs = ocg->CreatePlus(lhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
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
            CG_outputRepr *v = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);
            coef_t coef = cvi.curr_coef();
            if (coef > 0) {
              if (coef == 1)
                rhs = ocg->CreatePlus(rhs, v);
              else
                rhs = ocg->CreatePlus(rhs, ocg->CreateTimes(ocg->CreateInt(coef), v));
            }
            else { // coef < 0
              if (coef == -1)
                rhs = ocg->CreateMinus(rhs, v);
              else
                rhs = ocg->CreateMinus(rhs, ocg->CreateTimes(ocg->CreateInt(-coef), v));
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
  
  return result;
}


//
// return NULL if 0
//
CG_outputRepr *output_inequality_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, std::set<Variable_ID> excluded_floor_vars) {
  const_cast<Relation &>(R).setup_names(); // hack
  
  coef_t a = inequality.get_coef(v);
  assert(a != 0);
  excluded_floor_vars.insert(v);
  
  CG_outputRepr *repr = NULL;
  for (Constr_Vars_Iter cvi(inequality); cvi; cvi++)
    if (cvi.curr_var() != v) {
      CG_outputRepr *t;
      if (cvi.curr_var()->kind() == Wildcard_Var) {
        std::pair<bool, GEQ_Handle> result = find_floor_definition(R, cvi.curr_var(), excluded_floor_vars);
        if (!result.first) {
          delete repr;
          throw codegen_error("Can't generate bound expression with wildcard not involved in floor definition");
        }
        try {
          t = output_inequality_repr(ocg, result.second, cvi.curr_var(), R, assigned_on_the_fly, excluded_floor_vars);
        }
        catch (const std::exception &e) {
          delete repr;
          throw e;
        }
      }
      else
        t = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly);

      coef_t coef = cvi.curr_coef();
      if (a > 0) {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr, ocg->CreateTimes(ocg->CreateInt(coef), t));
        }
        else {
          if (coef == -1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr, ocg->CreateTimes(ocg->CreateInt(-coef), t));
        }
      }
      else {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr, ocg->CreateTimes(ocg->CreateInt(coef), t));
        }
        else {
          if (coef == -1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr, ocg->CreateTimes(ocg->CreateInt(-coef), t));
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


//
// nothing special, just an alias
//
CG_outputRepr *output_upper_bound_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const Relation &R, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  assert(inequality.get_coef(v) < 0);
  CG_outputRepr* zero_;

  zero_ =  output_inequality_repr(ocg, inequality, v, R, assigned_on_the_fly);

  if(!zero_)
         zero_ = ocg->CreateInt(0);

  return zero_;

}


//
// output lower bound with respect to lattice
//
CG_outputRepr *output_lower_bound_repr(CG_outputBuilder *ocg, const GEQ_Handle &inequality, Variable_ID v, const EQ_Handle &stride_eq, Variable_ID wc, const Relation &R, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  assert(inequality.get_coef(v) > 0);
  CG_outputRepr* zero_;
  if (wc == NULL || bound_must_hit_stride(inequality, v, stride_eq, wc, R, known)){
    zero_ =  output_inequality_repr(ocg, inequality, v, R, assigned_on_the_fly);
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
      v_repr = output_ident(ocg, R, v2, assigned_on_the_fly);
    else if (v2->kind() == Wildcard_Var) {
      std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v2);
      assert(result.first);
      v_repr = output_inequality_repr(ocg, result.second, v2, R, assigned_on_the_fly);
    }

    coef_t coef = cvi.curr_coef();
    if (sign < 0) {
      if (coef > 0) {
        if (coef == 1)
          strideBoundRepr = ocg->CreatePlus(strideBoundRepr, v_repr);
        else
          strideBoundRepr = ocg->CreatePlus(strideBoundRepr, ocg->CreateTimes(ocg->CreateInt(coef), v_repr));
      }
      else { // coef < 0
        if (coef == -1)
          strideBoundRepr = ocg->CreateMinus(strideBoundRepr, v_repr);
        else
          strideBoundRepr = ocg->CreateMinus(strideBoundRepr, ocg->CreateTimes(ocg->CreateInt(-coef), v_repr));
      }
    }
    else {
      if (coef > 0) {
        if (coef == 1)
          strideBoundRepr = ocg->CreateMinus(strideBoundRepr, v_repr);
        else
          strideBoundRepr = ocg->CreateMinus(strideBoundRepr, ocg->CreateTimes(ocg->CreateInt(coef), v_repr));
      }
      else { // coef < 0
        if (coef == -1)
          strideBoundRepr = ocg->CreatePlus(strideBoundRepr, v_repr);
        else
          strideBoundRepr = ocg->CreatePlus(strideBoundRepr, ocg->CreateTimes(ocg->CreateInt(-coef), v_repr));
      }
    }
  }  
  coef_t c = stride_eq.get_const();
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

  CG_outputRepr *repr = output_inequality_repr(ocg, inequality, v, R, assigned_on_the_fly);
  CG_outputRepr *repr2 = ocg->CreateCopy(repr);
  repr = ocg->CreatePlus(repr2, ocg->CreateIntegerMod(ocg->CreateMinus(strideBoundRepr, repr), ocg->CreateInt(abs(stride_eq.get_coef(wc)))));

  return repr;
}
  

//
// return loop control structure only
//
CG_outputRepr *output_loop(CG_outputBuilder *ocg, const Relation &R, int level, const Relation &known, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(R, const_cast<Relation &>(R).set_var(level));
  if (result.second != NULL)
    assert(abs(result.first.get_coef(const_cast<Relation &>(R).set_var(level))) == 1);

  std::vector<CG_outputRepr *> lbList, ubList;  
  try {

	coef_t const_lb = negInfinity, const_ub = posInfinity;

    for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); e; e++) {
      coef_t coef = (*e).get_coef(const_cast<Relation &>(R).set_var(level));

      if (coef > 0) {
        CG_outputRepr *repr = output_lower_bound_repr(ocg, *e, const_cast<Relation &>(R).set_var(level), result.first, result.second, R, known, assigned_on_the_fly);
          if (repr == NULL)
           repr = ocg->CreateInt(0);
        lbList.push_back(repr);

        if ((*e).is_const(const_cast<Relation &>(R).set_var(level))){
        	if(!result.second) {

                //no variables but v in constr
                coef_t L,m;
                L = -((*e).get_const());

                m = (*e).get_coef(const_cast<Relation &>(R).set_var(level));
                coef_t sb  =  (int) (ceil(((float) L) /m));
                set_max(const_lb, sb);
            }
        	else{

        		coef_t L,m,s,c;
        		        L = -((*e).get_const());
        		        m = (*e).get_coef(const_cast<Relation &>(R).set_var(level));
        		        s = abs(result.first.get_coef(result.second));
        		        c = result.first.get_const();
        		        coef_t sb  =  (s * (int) (ceil( (float) (L - (c * m)) /(s*m))))+ c;
        		        set_max(const_lb, sb);

        	}
        }

      }
      else if (coef < 0) {
        CG_outputRepr *repr = output_upper_bound_repr(ocg, *e, const_cast<Relation &>(R).set_var(level), R, assigned_on_the_fly);
        if (repr == NULL)
          repr = ocg->CreateInt(0);
        ubList.push_back(repr);

        if ((*e).is_const(const_cast<Relation &>(R).set_var(level))) {
                // no variables but v in constraint
                set_min(const_ub,-(*e).get_const()/(*e).get_coef(const_cast<Relation &>(R).set_var(level)));
        }

      }
    }
    
    if(fillInBounds && lbList.size() == 1 && const_lb != negInfinity)
       lowerBoundForLevel = const_lb;

    if(fillInBounds && const_ub != posInfinity)
       upperBoundForLevel = const_ub;
    if (lbList.size() == 0)
      throw codegen_error("missing lower bound at loop level " + to_string(level));
    if (ubList.size() == 0)
      throw codegen_error("missing upper bound at loop level " + to_string(level));
  }
  catch (const std::exception &e) {
    for (int i = 0; i < lbList.size(); i++)
      delete lbList[i];
    for (int i = 0; i < ubList.size(); i++)
      delete ubList[i];
    throw e;
  }

  CG_outputRepr *lbRepr = NULL;
  if (lbList.size() > 1)
    lbRepr = ocg->CreateInvoke("max", lbList);
  else // (lbList.size() == 1)
    lbRepr = lbList[0];

  CG_outputRepr *ubRepr = NULL;
  if (ubList.size() > 1)
    ubRepr = ocg->CreateInvoke("min", ubList);
  else // (ubList.size() == 1)
    ubRepr = ubList[0];

  CG_outputRepr *stRepr;
  if (result.second == NULL)
    stRepr = ocg->CreateInt(1);
  else
    stRepr = ocg->CreateInt(abs(result.first.get_coef(result.second)));
  CG_outputRepr *indexRepr = output_ident(ocg, R, const_cast<Relation &>(R).set_var(level), assigned_on_the_fly);
  return ocg->CreateInductive(indexRepr, lbRepr, ubRepr, stRepr);
}


//
// parameter f_root is inside f_exists, not the other way around.
// return replicated variable in new relation, with all cascaded floor definitions
// using wildcards defined in the same way as in the original relation.
//
Variable_ID replicate_floor_definition(const Relation &R, const Variable_ID floor_var,
                                       Relation &r, F_Exists *f_exists, F_And *f_root,
                                       std::map<Variable_ID, Variable_ID> &exists_mapping) {
  assert(R.n_out() == 0 && r.n_out() == 0 && R.n_inp() == r.n_inp());

  std::set<Variable_ID> excluded_floor_vars;
  std::stack<Variable_ID> to_fill;
  to_fill.push(floor_var);

  while (!to_fill.empty()) {
    Variable_ID v = to_fill.top();
    to_fill.pop();
    if (excluded_floor_vars.find(v) != excluded_floor_vars.end())
      continue;
    
    std::pair<bool, GEQ_Handle> result = find_floor_definition(R, v, excluded_floor_vars);
    assert(result.first);
    excluded_floor_vars.insert(v);
      
    GEQ_Handle h1 = f_root->add_GEQ();
    GEQ_Handle h2 = f_root->add_GEQ();
    for (Constr_Vars_Iter cvi(result.second); cvi; cvi++) {
      Variable_ID v2 = cvi.curr_var();
      switch  (v2->kind()) {
      case Input_Var: {
        int pos = v2->get_position();
        h1.update_coef(r.input_var(pos), cvi.curr_coef());
        h2.update_coef(r.input_var(pos), -cvi.curr_coef());
        break;
      }
      case Wildcard_Var: {
        std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find(v2);
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
    h2.update_const(-result.second.get_const()-result.second.get_coef(v)-1);
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
  assert(R.n_out()==0);
  
  Relation r = Relation::True(R.n_set());
  
  for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); e; e++)
    if (!(*e).has_wildcards()) {
      r.and_with_GEQ(*e);
      r.simplify();
      r.copy_names(R);
      r.setup_names();
      return r;
    }
  
  for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); e; e++)
    if (!(*e).has_wildcards()) {
      r.and_with_GEQ(*e);
      r.simplify();
      r.copy_names(R);
      r.setup_names();
      return r;
    }
  
  for (EQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->EQs()); e; e++)
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

  for (GEQ_Iterator e(const_cast<Relation &>(R).single_conjunct()->GEQs()); e; e++)
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
          direction = cvi.curr_coef()>0?true:false;
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
              direction2 = cvi.curr_coef()>0?true:false;
              break;
            case Input_Var:
              if (cvi.curr_var()->get_position() > max_level2)
                max_level2 = cvi.curr_var()->get_position();
              break;
            default:
              ;
            }

          if (num_wildcard2 == 1 && max_level2 != level-1 && wc2 == wc && direction2 == not direction) {
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
                h.update_coef(r.input_var(cvi.curr_var()->get_position()), cvi.curr_coef());
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
                h.update_coef(r.input_var(cvi.curr_var()->get_position()), cvi.curr_coef());
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


//
// heavy lifting for code output for one leaf node
//
CG_outputRepr *leaf_print_repr(BoolSet<> active, const std::map<int, Relation> &guards, 
                               CG_outputRepr *guard_repr, const Relation &known,
                               int indent, CG_outputBuilder *ocg, const std::vector<int> &remap,
                               const std::vector<Relation> &xforms, const std::vector<CG_outputRepr *> &stmts,
                               const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  if (active.num_elem() == 0)
    return NULL;
  
  CG_outputRepr *stmt_list = NULL;
  for (BoolSet<>::iterator i = active.begin(); i != active.end(); i++) {
    std::map<int, Relation>::const_iterator j = guards.find(*i);
    if (j == guards.end() || Must_Be_Subset(copy(known), copy(j->second))) {
      Relation mapping = Inverse(copy((xforms[remap[*i]])));
      mapping.simplify();
      mapping.setup_names();
      std::vector<std::string> loop_vars;
      for (int k = 1; k <= mapping.n_out(); k++) {
        loop_vars.push_back(mapping.output_var(k)->name());
//        std::cout << "CG_Utils:: " << k << ", " << mapping.output_var(k)->name().c_str() << "\n";
      }
      std::vector<CG_outputRepr *> sList = output_substitutions(ocg, mapping, assigned_on_the_fly);
      stmt_list = ocg->StmtListAppend(stmt_list, ocg->CreateSubstitutedStmt((guard_repr==NULL)?indent:indent+1, stmts[remap[*i]]->clone(), loop_vars, sList));
      active.unset(*i);
    }
  }

  if (stmt_list != NULL) {
    if (active.num_elem() != 0)
      stmt_list = ocg->StmtListAppend(stmt_list, leaf_print_repr(active, guards, NULL, known, (guard_repr==NULL)?indent:indent+1, ocg, remap, xforms, stmts, assigned_on_the_fly));
    if (guard_repr == NULL)
      return stmt_list;
    else
      return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
  }
  else {
    Relation then_cond = find_best_guard(const_cast<std::map<int, Relation> &>(guards)[*(active.begin())], active, guards);
    assert(!then_cond.is_obvious_tautology());
    Relation new_then_known = Intersection(copy(known), copy(then_cond));
    new_then_known.simplify();
    Relation else_cond = Complement(copy(then_cond));
    else_cond.simplify();
    Relation new_else_known = Intersection(copy(known), copy(else_cond));
    new_else_known.simplify();
    
    BoolSet<> then_active(active.size()), else_active(active.size()), indep_active(active.size());
    std::map<int, Relation> then_guards, else_guards;
    for (BoolSet<>::iterator i = active.begin(); i != active.end(); i++) {
      Relation &r = const_cast<std::map<int, Relation> &>(guards)[*i];
      if (Must_Be_Subset(copy(r), copy(then_cond))) {
        Relation r2 = Gist(copy(r), copy(then_cond), 1);
        if (!r2.is_obvious_tautology())
          then_guards[*i] = r2;
        then_active.set(*i);
      }
      else if (Must_Be_Subset(copy(r), copy(else_cond))) {
        Relation r2 = Gist(copy(r), copy(else_cond), 1);
        if (!r2.is_obvious_tautology())
          else_guards[*i] = r2;
        else_active.set(*i);
      }
      else
        indep_active.set(*i);
    }
    assert(!then_active.empty());
    
    CG_outputRepr *new_guard_repr = output_guard(ocg, then_cond, assigned_on_the_fly);
    if (else_active.empty() && indep_active.empty()) {      
      guard_repr = ocg->CreateAnd(guard_repr, new_guard_repr);
      return leaf_print_repr(then_active, then_guards, guard_repr, new_then_known, indent, ocg, remap, xforms, stmts, assigned_on_the_fly);
    }
    else if (else_active.empty() && !indep_active.empty()) {
      int new_indent = (guard_repr==NULL)?indent:indent+1;
      stmt_list = leaf_print_repr(then_active, then_guards, new_guard_repr, new_then_known, new_indent, ocg, remap, xforms, stmts, assigned_on_the_fly);
      stmt_list = ocg->StmtListAppend(stmt_list, leaf_print_repr(indep_active, guards, NULL, known, new_indent, ocg, remap, xforms, stmts, assigned_on_the_fly));
      if (guard_repr == NULL)
        return stmt_list;
      else
        return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
    }
    else { // (!else_active.empty())
      int new_indent = (guard_repr==NULL)?indent:indent+1;
      CG_outputRepr *then_stmt_list = leaf_print_repr(then_active, then_guards, NULL, new_then_known, new_indent+1, ocg, remap, xforms, stmts, assigned_on_the_fly);
      CG_outputRepr *else_stmt_list = leaf_print_repr(else_active, else_guards, NULL, new_else_known, new_indent+1, ocg, remap, xforms, stmts, assigned_on_the_fly);
      stmt_list = ocg->CreateIf(new_indent, new_guard_repr, then_stmt_list, else_stmt_list);
      if (!indep_active.empty())
        stmt_list = ocg->StmtListAppend(stmt_list, leaf_print_repr(indep_active, guards, NULL, known, new_indent, ocg, remap, xforms, stmts, assigned_on_the_fly));
      if (guard_repr == NULL)
        return stmt_list;
      else
        return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
    }
  }
}


//
// heavy lifting for code output for one level of loop nodes
//
CG_outputRepr *loop_print_repr(const std::vector<CG_loop *> &loops, int start, int end,
                               const Relation &guard, CG_outputRepr *guard_repr,
                               int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts,
                               const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) {
  if (start >= end)
    return NULL;

  Relation R = Gist(copy(loops[start]->guard_), copy(guard), 1);
  if (Must_Be_Subset(Intersection(copy(loops[start]->known_), copy(guard)), copy(R))) {
    int new_indent = (guard_repr==NULL)?indent:indent+1;
    int i = start+1;
    for ( ; i < end; i++)
      if (!Gist(copy(loops[i]->guard_), copy(guard), 1).is_obvious_tautology())
        break;
    CG_outputRepr *stmt_list = NULL;
    for (int j = start; j < i; j++)
      stmt_list = ocg->StmtListAppend(stmt_list, loops[j]->printRepr(false, new_indent, ocg, stmts, assigned_on_the_fly));
    stmt_list = ocg->StmtListAppend(stmt_list, loop_print_repr(loops, i, end, guard, NULL, new_indent, ocg, stmts, assigned_on_the_fly));
    if (guard_repr == NULL)
      return stmt_list;
    else
      return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
  }

  Relation then_cond = find_best_guard(R, loops, start, end);
  assert(!then_cond.is_obvious_tautology());
  Relation else_cond = Complement(copy(then_cond));
  else_cond.simplify();
  
  std::vector<CG_loop *> then_loops, else_loops, indep_loops;
  int i = start;
  for ( ; i < end; i++)
    if (!Must_Be_Subset(copy(loops[i]->guard_), copy(then_cond)))
      break;
  int j = i;
  for ( ; j < end; j++)
    if (!Must_Be_Subset(copy(loops[j]->guard_), copy(else_cond)))
      break;
  assert(i>start);

  CG_outputRepr *new_guard_repr = output_guard(ocg, then_cond, assigned_on_the_fly);
  if (j == i && end == j) {
    guard_repr = ocg->CreateAnd(guard_repr, new_guard_repr);
    Relation new_guard = Intersection(copy(guard), copy(then_cond));
    new_guard.simplify();
    return loop_print_repr(loops, start, end, new_guard, guard_repr, indent, ocg, stmts, assigned_on_the_fly);
  }
  else if (j == i && end > j) {
    int new_indent = (guard_repr==NULL)?indent:indent+1;
    Relation new_guard = Intersection(copy(guard), copy(then_cond));
    new_guard.simplify();
    CG_outputRepr *stmt_list = loop_print_repr(loops, start, i, new_guard, new_guard_repr, new_indent, ocg, stmts, assigned_on_the_fly);
    stmt_list = ocg->StmtListAppend(stmt_list, loop_print_repr(loops, j, end, guard, NULL, new_indent, ocg, stmts, assigned_on_the_fly));
    if (guard_repr == NULL)
      return stmt_list;
    else
      return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
  }
  else { // (j > i)
    int new_indent = (guard_repr==NULL)?indent:indent+1;
    Relation then_new_guard = Intersection(copy(guard), copy(then_cond));
    then_new_guard.simplify();
    CG_outputRepr *then_stmt_list = loop_print_repr(loops, start, i, then_new_guard, NULL, new_indent+1, ocg, stmts, assigned_on_the_fly);
    Relation else_new_guard = Intersection(copy(guard), copy(else_cond));
    else_new_guard.simplify();
    CG_outputRepr *else_stmt_list = loop_print_repr(loops, i, j, else_new_guard, NULL, new_indent+1, ocg, stmts, assigned_on_the_fly);
    CG_outputRepr *stmt_list = ocg->CreateIf(new_indent, new_guard_repr, then_stmt_list, else_stmt_list);
    stmt_list = ocg->StmtListAppend(stmt_list, loop_print_repr(loops, j, end, guard, NULL, new_indent, ocg, stmts, assigned_on_the_fly));
    if (guard_repr == NULL)
      return stmt_list;
    else
      return ocg->CreateIf(indent, guard_repr, stmt_list, NULL);
  }
}

}
