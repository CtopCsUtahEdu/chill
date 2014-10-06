/*****************************************************************************
 Copyright (C) 1994-2000 University of Maryland
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   utility functions for outputing CG_outputReprs

 Notes:
     
 History:
   07/30/10 collect various code outputing into one place, by Chun Chen
*****************************************************************************/

#include <omega.h>
#include <code_gen/CG_stringBuilder.h>
#include <code_gen/output_repr.h>
#include <basic/omega_error.h>
#include <math.h>
#include <stack>
#include <typeinfo>

namespace omega {

extern Tuple<Tuple<Relation> > projected_nIS;
int var_substitution_threshold = 100;
//protonu.
extern int upperBoundForLevel;
extern int lowerBoundForLevel;
extern bool fillInBounds;
//end--protonu.

}


namespace omega {

std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R_, Variable_ID v, const std::vector<CG_outputRepr *> &assigned_on_the_fly);


namespace {




void get_stride(const Constraint_Handle &h, Variable_ID &wc, coef_t &step){
  wc = 0;
  for(Constr_Vars_Iter i(h,true); i; i++) {
    assert(wc == 0);
    wc = (*i).var;
    step = ((*i).coef);
  }
}

}

CG_outputRepr* outputIdent(CG_outputBuilder* ocg, const Relation &R_, Variable_ID v, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &R = const_cast<Relation &>(R_);
  
  switch (v->kind()) {
  case Set_Var: {
    int pos = v->get_position();
    if (assigned_on_the_fly[pos-1] != NULL)
      return assigned_on_the_fly[pos-1]->clone();
    else
      return ocg->CreateIdent(v->name());
    break;
  }
  case Global_Var: {
    if (v->get_global_var()->arity() == 0)
      return ocg->CreateIdent(v->name());
    else {
      /* This should be improved to take into account the possible elimination
         of the set variables. */
      int arity = v->get_global_var()->arity();
      //assert(arity <= last_level);
      Tuple<CG_outputRepr *> argList;
      // Relation R = Relation::True(arity);
      
      // name_codegen_vars(R); // easy way to make sure the names are correct.
      for(int i = 1; i <= arity; i++)
        argList.append(ocg->CreateIdent(R.set_var(i)->name()));
      CG_outputRepr *call = ocg->CreateInvoke(v->get_global_var()->base_name(), argList);
      return call;
    }
    break;
  }
  default:
    throw std::invalid_argument("wrong variable type");
  }
}


//----------------------------------------------------------------------------
// Translate equality constraints to if-condition and assignment.
// return.first is right-hand-side of assignment, return.second
// is true if assignment is required.
//   -- by chun 07/29/2010
// ----------------------------------------------------------------------------
std::pair<CG_outputRepr *, bool> outputAssignment(CG_outputBuilder *ocg, const Relation &R_,  Variable_ID v, Relation &enforced, CG_outputRepr *&if_repr, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &R = const_cast<Relation &>(R_);

  Conjunct *c = R.query_DNF()->single_conjunct();

  // check whether to generate if-conditions from equality constraints
  for (EQ_Iterator ei(c); ei; ei++)
    if (!(*ei).has_wildcards() && abs((*ei).get_coef(v)) > 1) {
      Relation r(R.n_set());
      F_And *f_super_root = r.add_and();
      F_Exists *fe = f_super_root->add_exists();
      Variable_ID e = fe->declare();
      F_And *f_root = fe->add_and();
      EQ_Handle h = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
        switch ((*cvi).var->kind()) {
        case Input_Var: {
          if ((*cvi).var == v)
            h.update_coef(e, (*cvi).coef);
          else
            h.update_coef(r.set_var((*cvi).var->get_position()), (*cvi).coef);
          break;
        }
        case Global_Var: {            
          Global_Var_ID g = (*cvi).var->get_global_var();
          Variable_ID v2;
          if (g->arity() == 0)
            v2 = r.get_local(g);
          else
            v2 = r.get_local(g, (*cvi).var->function_of());
          h.update_coef(v2, (*cvi).coef);
          break;
        }
        default:
          assert(0);
        }
      h.update_const((*ei).get_const());

      r.copy_names(R);
      r.setup_names();
        
      // need if-condition to make sure this loop variable has integer value
      if (!Gist(r, copy(enforced), 1).is_obvious_tautology()) {
        coef_t coef = (*ei).get_coef(v);
        coef_t sign = -((coef>0)?1:-1);
        coef = abs(coef);

        CG_outputRepr *term = NULL;
        for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
          if ((*cvi).var != v) {
            CG_outputRepr *varRepr = outputIdent(ocg, R, (*cvi).var, assigned_on_the_fly);
            coef_t t = sign*(*cvi).coef;
            if (t == 1)
              term = ocg->CreatePlus(term, varRepr);
            else if (t == -1)
              term = ocg->CreateMinus(term, varRepr);
            else if (t > 0)
              term = ocg->CreatePlus(term, ocg->CreateTimes(ocg->CreateInt(t), varRepr));
            else if (t < 0)
              term = ocg->CreateMinus(term, ocg->CreateTimes(ocg->CreateInt(-t), varRepr));
          }
        coef_t t = sign*(*ei).get_const();
        if (t > 0)
          term = ocg->CreatePlus(term, ocg->CreateInt(t));
        else if (t < 0)
          term = ocg->CreateMinus(term, ocg->CreateInt(-t));

        term = ocg->CreateIntegerMod(term, ocg->CreateInt(coef));
        term = ocg->CreateEQ(term, ocg->CreateInt(0));

        if_repr = ocg->CreateAnd(if_repr, term);
      }

      enforced.and_with_EQ(*ei);
      enforced.simplify();
    }
  
  // find the simplest assignment
  std::pair<EQ_Handle, int> a = find_simplest_assignment(R, v, assigned_on_the_fly);
    
  // now generate assignment
  if (a.second < INT_MAX) {
    EQ_Handle eq = a.first;
    CG_outputRepr *rop_repr = NULL;
    
    coef_t divider = eq.get_coef(v);
    int sign = 1;
    if (divider < 0) {
      divider = -divider;
      sign = -1;
    }
        
    for (Constr_Vars_Iter cvi(eq); cvi; cvi++)
      if ((*cvi).var != v) {
        CG_outputRepr *var_repr = outputIdent(ocg, R, (*cvi).var, assigned_on_the_fly);
        coef_t coef = (*cvi).coef;
        if (-sign * coef == -1)
          rop_repr = ocg->CreateMinus(rop_repr, var_repr);
        else if (-sign * coef < -1)
          rop_repr = ocg->CreateMinus(rop_repr, ocg->CreateTimes(ocg->CreateInt(sign * coef), var_repr));
        else if (-sign * coef == 1)
          rop_repr = ocg->CreatePlus(rop_repr, var_repr);
        else // -sign * coef > 1
          rop_repr = ocg->CreatePlus(rop_repr, ocg->CreateTimes(ocg->CreateInt(-sign * coef), var_repr));
      }
        
    coef_t c_term = -(eq.get_const() * sign);

    if (c_term > 0)
      rop_repr = ocg->CreatePlus(rop_repr, ocg->CreateInt(c_term));
    else if (c_term < 0)
      rop_repr = ocg->CreateMinus(rop_repr, ocg->CreateInt(-c_term));
    else if (rop_repr == NULL)
      rop_repr = ocg->CreateInt(0);

    if (divider != 1)
      rop_repr = ocg->CreateIntegerDivide(rop_repr, ocg->CreateInt(divider));

    enforced.and_with_EQ(eq);
    enforced.simplify();

    if (a.second > var_substitution_threshold)
      return std::make_pair(rop_repr, true);
    else
      return std::make_pair(rop_repr, false);
  }
  else
    return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);
}


//----------------------------------------------------------------------------
// Don't use Substitutions class since it can't handle integer
// division.  Instead, use relation mapping to a single output
// variable to get substitution.  -- by chun, 07/19/2007
//----------------------------------------------------------------------------
Tuple<CG_outputRepr*> outputSubstitution(CG_outputBuilder* ocg, const Relation &R_, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &R = const_cast<Relation &>(R_);
  
  const int n = R.n_out();
  Tuple<CG_outputRepr*> oReprList;
       
  // Find substitution for each output variable
  for (int i = 1; i <= n; i++) {
    Relation mapping(n, 1);
    F_And *f_root = mapping.add_and();
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(1), 1);
    h.update_coef(mapping.input_var(i), -1);

    Relation S = Composition(mapping, copy(R));
  
    std::pair<EQ_Handle, int> a = find_simplest_assignment(S, S.output_var(1), assigned_on_the_fly);
    
    if (a.second < INT_MAX) {
      while (a.second > 0) {
        EQ_Handle eq = a.first;
        std::set<int> candidates;
        for (Constr_Vars_Iter cvi(eq); cvi; cvi++)
          if ((*cvi).var->kind() == Input_Var)
            candidates.insert((*cvi).var->get_position());

        bool changed = false;
        for (std::set<int>::iterator j = candidates.begin(); j != candidates.end(); j++) {
          Relation S2 = Project(copy(S), *j, Input_Var);
          std::pair<EQ_Handle, int> a2 = find_simplest_assignment(S2, S2.output_var(1), assigned_on_the_fly);
          if (a2.second <= a.second) {
            S = S2;
            a = a2;
            changed = true;
            break;
          }
        }
        if (!changed)
          break;
      }
    }

    if (a.second < INT_MAX) {
      CG_outputRepr *repr = NULL;
      EQ_Handle eq = a.first;
      Variable_ID v = S.output_var(1);

      for (int j = 1; j <= S.n_inp(); j++)
        S.name_input_var(j, R.input_var(j)->name());
      S.setup_names();
      
      int d = eq.get_coef(v);
      assert(d != 0);
      int sign = (d>0)?-1:1;
      d = -sign * d;
      for (Constr_Vars_Iter cvi(eq); cvi; cvi++)
        if ((*cvi).var != v) {
          int coef = sign * (*cvi).coef;
          CG_outputRepr *op = outputIdent(ocg, S, (*cvi).var, assigned_on_the_fly);
          if (coef > 1)
            op = ocg->CreateTimes(ocg->CreateInt(coef), op);
          else if (coef < -1)
            op = ocg->CreateTimes(ocg->CreateInt(-coef), op);              
          if (coef > 0)
            repr = ocg->CreatePlus(repr, op);
          else if (coef < 0)
            repr = ocg->CreateMinus(repr, op);
        }

      int c = sign * eq.get_const();
      if (c > 0)
        repr = ocg->CreatePlus(repr, ocg->CreateInt(c));
      else if (c < 0)
        repr = ocg->CreateMinus(repr, ocg->CreateInt(-c));
      else if (repr == NULL)
        repr = ocg->CreateInt(0);
        
      if (d != 1)
        repr = ocg->CreateIntegerDivide(repr, ocg->CreateInt(d));
        
      oReprList.append(repr);
    }
    else
      oReprList.append(NULL);
  }

  return oReprList;
}

namespace {

Relation create_stride_on_bound(int n, const std::map<Variable_ID, coef_t> &lb, coef_t stride) {
  Relation result(n);
  F_And *f_root = result.add_and();
  EQ_Handle h = f_root->add_stride(stride);
  
  for (std::map<Variable_ID, coef_t>::const_iterator i = lb.begin(); i != lb.end(); i++) {
    if (i->first == NULL)
      h.update_const(i->second);
    else {
      switch(i->first->kind()) {
      case Input_Var: {
        int pos = i->first->get_position();
        h.update_coef(result.set_var(pos), i->second);
        break;
      }
      case Global_Var: {
        Global_Var_ID g = i->first->get_global_var();
        Variable_ID v;
        if (g->arity() == 0)
          v = result.get_local(g);
        else
          v = result.get_local(g, i->first->function_of());
        h.update_coef(v, i->second);
        break;
      }
      default:
        assert(0);
      }
    }
  }

  return result;
}

}

//----------------------------------------------------------------------------
// Find the most restrictive common stride constraint for a set of
// relations. -- by chun, 05/20/09
// ----------------------------------------------------------------------------
Relation greatest_common_step(const Tuple<Relation> &I, const Tuple<int> &active, int level, const Relation &known) {
  assert(I.size() == active.size());
  int n = 0;

  std::vector<Relation> I1, I2;
  for (int i = 1; i <= I.size(); i++)
    if (active[i]) {
      if (n == 0)
        n = I[i].n_set();

      Relation r1;
      if (known.is_null())
        r1 = copy(I[i]);
      else {
        r1 = Intersection(copy(I[i]), copy(known));
        r1.simplify();
      }
      I1.push_back(r1);
      Relation r2 = Gist(copy(I[i]), copy(known));
      assert(r2.is_upper_bound_satisfiable());
      if (r2.is_obvious_tautology())
        return Relation::True(n);
      I2.push_back(r2);
    }
  
  std::vector<bool> is_exact(I2.size(), true);
  std::vector<coef_t> step(I2.size(), 0);
  std::vector<coef_t> messy_step(I2.size(), 0);
  Variable_ID t_col = set_var(level);
  std::map<Variable_ID, coef_t> lb;

  // first check all clean strides: t_col = ... (mod step)
  for (size_t i = 0; i < I2.size(); i++) {
    Conjunct *c = I2[i].query_DNF()->single_conjunct();

    bool is_degenerated = false;
    for (EQ_Iterator e = c->EQs(); e; e++) {
      coef_t coef = abs((*e).get_coef(t_col));
      if (coef != 0 && !(*e).has_wildcards()) {
        is_degenerated = true;
        break;
      }
    }
    if (is_degenerated)
      continue;

    for (EQ_Iterator e = c->EQs(); e; e++) {
      if ((*e).has_wildcards()) {
        coef_t coef = abs((*e).get_coef(t_col));
        if (coef == 0)
          continue;
        if (coef != 1) {
          is_exact[i] = false;
          continue;
        }

        coef_t this_step = abs(Constr_Vars_Iter(*e, true).curr_coef());
        assert(this_step != 1);

        if (lb.size() != 0) {
          Relation test = create_stride_on_bound(n, lb, this_step);
          if (Gist(test, copy(I1[i])).is_obvious_tautology()) {
            if (step[i] == 0)
              step[i] = this_step;
            else
              step[i] = lcm(step[i], this_step);
          }
          else
            is_exact[i] = false;
        }
        else {
          // try to find a lower bound that hits on stride
          Conjunct *c = I2[i].query_DNF()->single_conjunct();
          for (GEQ_Iterator ge = c->GEQs(); ge; ge++) {
            if ((*ge).has_wildcards() || (*ge).get_coef(t_col) != 1)
              continue;

            std::map<Variable_ID, coef_t> cur_lb;
            for (Constr_Vars_Iter cv(*ge); cv; cv++)
              cur_lb[cv.curr_var()] = cv.curr_coef();
            cur_lb[NULL] = (*ge).get_const();

            Relation test = create_stride_on_bound(n, cur_lb, this_step);        
            if (Gist(test, copy(I1[i])).is_obvious_tautology()) {
              if (step[i] == 0)
                step[i] = this_step;
              else
                step[i] = lcm(step[i], this_step);

              lb = cur_lb;
              break;
            }
          }

          // no clean lower bound, thus we use this modular constraint as is
          if (lb.size() == 0) {
            std::map<Variable_ID, coef_t> cur_lb;
            int wild_count = 0;
            for (Constr_Vars_Iter cv(*e); cv; cv++)
              if (cv.curr_var()->kind() == Wildcard_Var)
                wild_count++;
              else
                cur_lb[cv.curr_var()] = cv.curr_coef();
            cur_lb[NULL] = (*e).get_const();

            if (wild_count == 1) {
              lb = cur_lb;
              if (step[i] == 0)
                step[i] = this_step;
              else
                step[i] = lcm(step[i], this_step);
            }
          }

          if (lb.size() == 0)
            is_exact[i] = false;
        }
      }
    }
  }

  // aggregate all exact steps
  coef_t global_step = 0;
  for (size_t i = 0; i < is_exact.size(); i++)
    if (is_exact[i])
      global_step = gcd(global_step, step[i]);
  if (global_step == 1)
    return Relation::True(n);

  // now check all messy strides: a*t_col = ... (mod step)
  for (size_t i = 0; i < I2.size(); i++)
    if (!is_exact[i]) {      
      Conjunct *c = I2[i].query_DNF()->single_conjunct();
      for (EQ_Iterator e = c->EQs(); e; e++) {
        coef_t coef = abs((*e).get_coef(t_col));
        if (coef == 0 || coef == 1)
          continue;
        
        // make a guess for messy stride condition -- by chun 07/27/2007
        coef_t this_step = abs(Constr_Vars_Iter(*e, true).curr_coef());
        this_step /= gcd(this_step, coef);
        this_step = gcd(global_step, this_step);
        if (this_step == 1)
          continue;

        if (lb.size() != 0) {
          Relation test = create_stride_on_bound(n, lb, this_step);
          if (Gist(test, copy(I1[i])).is_obvious_tautology()) {
            if (step[i] == 0)
              step[i] = this_step;
            else
              step[i] = lcm(step[i], this_step);
          }
        }
        else {
          // try to find a lower bound that hits on stride
          Conjunct *c = I2[i].query_DNF()->single_conjunct();
          for (GEQ_Iterator ge = c->GEQs(); ge; ge++) {
            if ((*ge).has_wildcards() || (*ge).get_coef(t_col) != 1)
              continue;

            std::map<Variable_ID, coef_t> cur_lb;

            for (Constr_Vars_Iter cv(*ge); cv; cv++)
              cur_lb[cv.curr_var()] = cv.curr_coef();

            cur_lb[NULL] = (*ge).get_const();

            Relation test = create_stride_on_bound(n, cur_lb, this_step);        
            if (Gist(test, copy(I1[i])).is_obvious_tautology()) {
              if (step[i] == 0)
                step[i] = this_step;
              else
                step[i] = lcm(step[i], this_step);

              lb = cur_lb;
              break;
            }
          }
        }
      }
    }

  // aggregate all non-exact steps
  for (size_t i = 0; i < is_exact.size(); i++)
    if (!is_exact[i])
      global_step = gcd(global_step, step[i]);
  if (global_step == 1 || global_step == 0)
    return Relation::True(n);

  Relation result = create_stride_on_bound(n, lb, global_step);

  // check for statements that haven't been factored into global step
  for (size_t i = 0; i < I1.size(); i++)
    if (step[i] == 0) {
      if (!Gist(copy(result), copy(I1[i])).is_obvious_tautology())
        return Relation::True(n);
    }

  return result;
}


//-----------------------------------------------------------------------------
// Substitute variables in a statement according to mapping function.
//-----------------------------------------------------------------------------
CG_outputRepr* outputStatement(CG_outputBuilder *ocg, CG_outputRepr *stmt, int indent, const Relation &mapping_, const Relation &known_, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation mapping = copy(mapping_);
  Relation known = copy(known_);
  Tuple<std::string> loop_vars;

  for (int i = 1; i <= mapping.n_inp(); i++)
    loop_vars.append(mapping.input_var(i)->name());

  // discard non-existant variables from iteration spaces -- by chun 12/31/2008
  if (known.n_set() > mapping.n_out()) {
    Relation r(known.n_set(), mapping.n_out());
    F_And *f_root = r.add_and();
    for (int i = 1; i <= mapping.n_out(); i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(r.input_var(i), 1);
      h.update_coef(r.output_var(i), -1);
    }
    known = Range(Restrict_Domain(r, known));
    known.simplify();
  }
  
  // remove modular constraints from known to simplify mapping process -- by chun 11/10/2008
  Relation k(known.n_set());
  F_And *f_root = k.add_and();
  Conjunct *c = known.query_DNF()->single_conjunct();
  for (EQ_Iterator e = c->EQs(); e; e++) {
    if (!(*e).has_wildcards())
      f_root->add_EQ(*e);
  }
  k.simplify();
  
  // get variable substituion list
  Relation Inv_mapping = Restrict_Domain(Inverse(mapping), k);
  Tuple<CG_outputRepr*> sList = outputSubstitution(ocg, Inv_mapping, assigned_on_the_fly);
  
  return ocg->CreatePlaceHolder(indent, stmt, sList, loop_vars);
}


// find floor definition for variable such as m-3 <= 4v <= m
bool findFloorInequality(Relation &r, Variable_ID v, GEQ_Handle &h, Variable_ID excluded) {
  Conjunct *c = r.single_conjunct();

  std::set<Variable_ID> var_checked;
  std::stack<Variable_ID> var_checking;
  var_checking.push(v);

  while (!var_checking.empty()) {
    Variable_ID v2 = var_checking.top();
    var_checking.pop();

    bool is_floor = false;
    for (GEQ_Iterator gi(c); gi; gi++) {
      if (excluded != NULL && (*gi).get_coef(excluded) != 0)
        continue;
      
      coef_t a = (*gi).get_coef(v2);
      if (a < 0) {
        for (GEQ_Iterator gi2(c); gi2; gi2++) {
          coef_t b = (*gi2).get_coef(v2);
          if (b == -a && (*gi).get_const()+(*gi2).get_const() < -a) {
            bool match = true;
            for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
              if ((*gi2).get_coef((*cvi).var) != -(*cvi).coef) {
                match = false;
                break;
              }
            if (!match)
              continue;
            for (Constr_Vars_Iter cvi(*gi2); cvi; cvi++)
              if ((*gi).get_coef((*cvi).var) != -(*cvi).coef) {
                match = false;
                break;
              }
            if (match) {
              var_checked.insert(v2);
              is_floor = true;
              if (v == v2)
                h = *gi;

              for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
                if (((*cvi).var->kind() == Exists_Var || (*cvi).var->kind() == Wildcard_Var) &&
                    var_checked.find((*cvi).var) == var_checked.end())
                  var_checking.push((*cvi).var);
              
              break;
            }
          }
        }
        if (is_floor)
          break;
      }
    }
    if (!is_floor)
      return false;
  }
  return true;
}

          


//-----------------------------------------------------------------------------
// Output a reqular equality or inequality to conditions.
// e.g. (i=5*j)
//-----------------------------------------------------------------------------
CG_outputRepr* output_as_guard(CG_outputBuilder* ocg, const Relation &guards_in, Constraint_Handle e, bool is_equality, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &guards = const_cast<Relation &>(guards_in);
  if (e.has_wildcards())
    throw std::invalid_argument("constraint must not have wildcard");
  
  Variable_ID v = (*Constr_Vars_Iter(e)).var;

  coef_t saved_coef = ((e).get_coef(v));
  int sign = saved_coef < 0 ? -1 : 1;

  (e).update_coef_during_simplify(v, -saved_coef+sign);
  CG_outputRepr* rop = outputEasyBoundAsRepr(ocg, guards, e, v, false, 0, assigned_on_the_fly);
  (e).update_coef_during_simplify(v,saved_coef-sign);

  CG_outputRepr* lop = outputIdent(ocg, guards, v, assigned_on_the_fly);
  if (abs(saved_coef) != 1)
    lop = ocg->CreateTimes(ocg->CreateInt(abs(saved_coef)), lop);

  
  if (is_equality) {
    return ocg->CreateEQ(lop, rop);
  }
  else {
    if (saved_coef < 0)
      return ocg->CreateLE(lop, rop);
    else
      return ocg->CreateGE(lop, rop);
  }
}


//-----------------------------------------------------------------------------
// Output stride conditions from equalities.
// e.g. (exists alpha: i = 5*alpha)
//-----------------------------------------------------------------------------
CG_outputRepr *output_EQ_strides(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation guards = const_cast<Relation &>(guards_in);
  Conjunct *c = guards.single_conjunct();

  CG_outputRepr *eqRepr = NULL;

  for (EQ_Iterator ei(c->EQs()); ei; ei++) {
    Variable_ID wc = NULL;
    for (Constr_Vars_Iter cvi((*ei), true); cvi; cvi++) {
      if (wc != NULL)
        throw codegen_error("Can't generate equality condition with multiple wildcards");
      else
        wc = (*cvi).var;
    }
    if (wc == NULL)
      continue;
    
    coef_t step = (*ei).get_coef(wc);

    (*ei).update_coef_during_simplify(wc, 1-step);
    CG_outputRepr* lop = outputEasyBoundAsRepr(ocg, guards, (*ei), wc, false, 0, assigned_on_the_fly);
    (*ei).update_coef_during_simplify(wc, step-1);
    
    CG_outputRepr* rop = ocg->CreateInt(abs(step));
    CG_outputRepr* intMod = ocg->CreateIntegerMod(lop, rop);
    CG_outputRepr* eqNode = ocg->CreateEQ(intMod, ocg->CreateInt(0));
    
    eqRepr = ocg->CreateAnd(eqRepr, eqNode);
  }
    
  return eqRepr;
}



//-----------------------------------------------------------------------------
// Output hole conditions created by inequalities involving wildcards.
// e.g. (exists alpha: 4*alpha <= i <= 5*alpha)
// Collect wildcards
// For each whildcard
//   collect lower and upper bounds in which wildcard appears
//   For each lower bound
//     create constraint with each upper bound
//-----------------------------------------------------------------------------
CG_outputRepr *output_GEQ_strides(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation guards = const_cast<Relation &>(guards_in);
  Conjunct *c = guards.single_conjunct();
  
  CG_outputRepr* geqRepr = NULL;

  std::set<Variable_ID> non_orphan_wildcard;
  for (GEQ_Iterator gi(c); gi; gi++) {
    int num_wild = 0;
    Variable_ID first_one;
    for (Constr_Vars_Iter cvi(*gi, true); cvi; cvi++) {
      num_wild++;
      if (num_wild == 1)
        first_one = (*cvi).var;
      else
        non_orphan_wildcard.insert((*cvi).var);
    }
    if (num_wild > 1)
      non_orphan_wildcard.insert(first_one);
  }

  for (int i = 1; i <= (*(c->variables())).size(); i++) {
    Variable_ID wc = (*(c->variables()))[i];
    if (wc->kind() == Wildcard_Var && non_orphan_wildcard.find(wc) == non_orphan_wildcard.end()) {
      Tuple<GEQ_Handle> lower, upper;
      for (GEQ_Iterator gi(c); gi; gi++) {
        if((*gi).get_coef(wc) > 0) 
          lower.append(*gi); 
        else if((*gi).get_coef(wc) < 0)
          upper.append(*gi);
      }

      // low: c*alpha - x >= 0
      // up:  -d*alpha + y >= 0
      for (Tuple_Iterator<GEQ_Handle> low(lower); low; low++) {
        for (Tuple_Iterator<GEQ_Handle> up(upper); up; up++) {
          coef_t low_coef = (*low).get_coef(wc);
          coef_t up_coef = (*up).get_coef(wc);
          
          (*low).update_coef_during_simplify(wc, 1-low_coef);
          CG_outputRepr* lowExpr = outputEasyBoundAsRepr(ocg, guards, *low, wc, false, 0, assigned_on_the_fly);
          (*low).update_coef_during_simplify(wc, low_coef-1);
          
          (*up).update_coef_during_simplify(wc, -1-up_coef);
          CG_outputRepr* upExpr = outputEasyBoundAsRepr(ocg, guards, *up, wc, false, 0, assigned_on_the_fly);
          (*up).update_coef_during_simplify(wc, up_coef+1);
 
          CG_outputRepr* intDiv = ocg->CreateIntegerDivide(upExpr, ocg->CreateInt(-up_coef));
          CG_outputRepr* rop = ocg->CreateTimes(ocg->CreateInt(low_coef), intDiv);
          CG_outputRepr* geqNode = ocg->CreateLE(lowExpr, rop);
 
          geqRepr = ocg->CreateAnd(geqRepr, geqNode);
        }
      }
    }
  }

  if (non_orphan_wildcard.size() > 0) {
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

  return geqRepr;
}


//-----------------------------------------------------------------------------
// Translate all constraints in a relation to guard conditions.
//-----------------------------------------------------------------------------
CG_outputRepr *outputGuard(CG_outputBuilder* ocg, const Relation &guards_in, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &guards = const_cast<Relation &>(guards_in);
  if (guards.is_null() || guards.is_obvious_tautology())
    return NULL;
  
  CG_outputRepr* nodeRepr = NULL;

  CG_outputRepr *eqStrideRepr = output_EQ_strides(ocg, guards, assigned_on_the_fly);
  nodeRepr = ocg->CreateAnd(nodeRepr, eqStrideRepr);

  CG_outputRepr *geqStrideRepr = output_GEQ_strides(ocg, guards, assigned_on_the_fly);
  nodeRepr = ocg->CreateAnd(nodeRepr, geqStrideRepr);  

  Conjunct *c = guards.single_conjunct();
  for(EQ_Iterator ei(c->EQs()); ei; ei++)
    if (!(*ei).has_wildcards()) {
      CG_outputRepr *eqRepr = output_as_guard(ocg, guards, (*ei), true, assigned_on_the_fly);
      nodeRepr = ocg->CreateAnd(nodeRepr, eqRepr);
    }
  for(GEQ_Iterator gi(c->GEQs()); gi; gi++)
    if (!(*gi).has_wildcards()) {
      CG_outputRepr *geqRepr = output_as_guard(ocg, guards, (*gi), false, assigned_on_the_fly);
      nodeRepr = ocg->CreateAnd(nodeRepr, geqRepr);
    }

  return nodeRepr;
}


//-----------------------------------------------------------------------------
// one is 1 for LB
// this function is overloaded should replace the original one
//-----------------------------------------------------------------------------
CG_outputRepr *outputLBasRepr(CG_outputBuilder* ocg, const GEQ_Handle &g, 
                              Relation &bounds, Variable_ID v,
                              coef_t stride, const EQ_Handle &strideEQ,
                              Relation known, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
#if ! defined NDEBUG
  coef_t v_coef;
  assert((v_coef = g.get_coef(v)) > 0);
#endif

  std::string s;
  CG_outputRepr *lbRepr;
  if (stride == 1) {
    lbRepr = outputEasyBoundAsRepr(ocg, bounds, g, v, false, 1, assigned_on_the_fly);
  }
  else {
    if (!boundHitsStride(g,v,strideEQ,stride,known)) {
      bounds.setup_names(); // boundsHitsStride resets variable names

      CG_stringBuilder oscg;
      std::string c = GetString(outputEasyBoundAsRepr(&oscg, bounds, strideEQ, v, true, 0, assigned_on_the_fly));
      CG_outputRepr *cRepr = NULL;
      if (c != std::string("0"))
        cRepr = outputEasyBoundAsRepr(ocg, bounds, strideEQ, v, true, 0, assigned_on_the_fly);
      std::string LoverM = GetString(outputEasyBoundAsRepr(&oscg, bounds, g, v, false, 1, assigned_on_the_fly));
      CG_outputRepr *LoverMRepr = NULL;
      if (LoverM != std::string("0"))
        LoverMRepr = outputEasyBoundAsRepr(ocg, bounds, g, v, false, 1, assigned_on_the_fly); 

      if (code_gen_debug > 2) {
        fprintf(DebugFile,"::: LoverM is %s\n", LoverM.c_str());
        fprintf(DebugFile,"::: c is %s\n", c.c_str());
      }

      int complexity1 = 0, complexity2 = 0;
      for (size_t i = 0; i < c.length(); i++)
        if (c[i] == '+' || c[i] == '-' || c[i] == '*' || c[i] == '/')
          complexity1++;
        else if (c[i] == ',')
          complexity1 += 2;
      for (size_t i = 0; i < LoverM.length(); i++)
        if (LoverM[i] == '+' || LoverM[i] == '-' || LoverM[i] == '*' || LoverM[i] == '/')
          complexity2++;
        else if (LoverM[i] == ',')
          complexity2 += 2;
      
      if (complexity1 < complexity2) {
        CG_outputRepr *idUp = LoverMRepr;
        CG_outputRepr *c1Repr = ocg->CreateCopy(cRepr);
        idUp = ocg->CreateMinus(idUp, c1Repr);
        idUp = ocg->CreatePlus(idUp, ocg->CreateInt(stride-1));
        CG_outputRepr *idLow = ocg->CreateInt(stride);
        lbRepr = ocg->CreateTimes(ocg->CreateInt(stride),
                                  ocg->CreateIntegerDivide(idUp, idLow));
        lbRepr = ocg->CreatePlus(lbRepr, cRepr);
      }
      else {
        CG_outputRepr *LoverM1Repr = ocg->CreateCopy(LoverMRepr);
        CG_outputRepr *imUp = ocg->CreateMinus(cRepr, LoverM1Repr);
        CG_outputRepr *imLow = ocg->CreateInt(stride);
        CG_outputRepr *intMod = ocg->CreateIntegerMod(imUp, imLow);
        lbRepr = ocg->CreatePlus(LoverMRepr, intMod);
      }
    } 
    else {
      // boundsHitsStride resets variable names
      bounds.setup_names(); 
      lbRepr = outputEasyBoundAsRepr(ocg, bounds, g, v, false, 0, assigned_on_the_fly);
    }
  }

  return lbRepr;
}

//-----------------------------------------------------------------------------
// one is -1 for UB
// this function is overloaded should replace the original one
//-----------------------------------------------------------------------------
CG_outputRepr *outputUBasRepr(CG_outputBuilder* ocg, const GEQ_Handle &g, 
                              Relation & bounds,
                              Variable_ID v,
                              coef_t /*stride*/, // currently unused
                              const EQ_Handle &/*strideEQ*/, //currently unused 
                              const std::vector<CG_outputRepr *> &assigned_on_the_fly) { 
  assert(g.get_coef(v) < 0);
  CG_outputRepr* upRepr = outputEasyBoundAsRepr(ocg, bounds, g, v, false, 0, assigned_on_the_fly);
  return upRepr;
}

//-----------------------------------------------------------------------------
// Print the expression for the variable given as v.  Works for both 
// GEQ's and EQ's, but produces intDiv (not intMod) when v has a nonunit 
// coefficient.  So it is OK for loop bounds, but for checking stride
// constraints, you want to make sure the coef of v is 1, and insert the
// intMod yourself.
//
// original name is outputEasyBound
//-----------------------------------------------------------------------------
CG_outputRepr* outputEasyBoundAsRepr(CG_outputBuilder* ocg, Relation &bounds,
                                     const Constraint_Handle &g, Variable_ID v, 
                                     bool ignoreWC,
                                     int ceiling,
                                     const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  // assert ignoreWC => g is EQ
  // rewrite constraint as foo (== or <= or >=) v, return foo as string 

  CG_outputRepr* easyBoundRepr = NULL;

  coef_t v_coef = g.get_coef(v);
  int v_sign = v_coef > 0 ? 1 : -1;
  v_coef *= v_sign;
  assert(v_coef > 0);
  // foo is (-constraint)/v_sign/v_coef 

  int sign_adj = -v_sign;

  //----------------------------------------------------------------------
  // the following generates +- cf*varName
  //----------------------------------------------------------------------
  for(Constr_Vars_Iter c2(g, false); c2; c2++) {
    if ((*c2).var != v && (!ignoreWC || (*c2).var->kind()!=Wildcard_Var)) {

      coef_t cf = (*c2).coef*sign_adj;
      assert(cf != 0);

      CG_outputRepr *varName;
      if ((*c2).var->kind() == Wildcard_Var) {
        GEQ_Handle h;
        if (!findFloorInequality(bounds, (*c2).var, h, v)) {
          if (easyBoundRepr != NULL) {
            easyBoundRepr->clear();
            delete easyBoundRepr;
          }
          return NULL;
        }
        varName = outputEasyBoundAsRepr(ocg, bounds, h, (*c2).var, false, 0, assigned_on_the_fly);
      }
      else {
        varName = outputIdent(ocg, bounds, (*c2).var, assigned_on_the_fly);
      }
      CG_outputRepr *cfRepr = NULL;

      if (cf > 1) {
        cfRepr = ocg->CreateInt(cf);
        CG_outputRepr* rbRepr = ocg->CreateTimes(cfRepr, varName);
        easyBoundRepr = ocg->CreatePlus(easyBoundRepr, rbRepr);
      }
      else if (cf < -1) {
        cfRepr = ocg->CreateInt(-cf);
        CG_outputRepr* rbRepr = ocg->CreateTimes(cfRepr, varName);
        easyBoundRepr = ocg->CreateMinus(easyBoundRepr, rbRepr);
      }
      else if (cf == 1) {
        easyBoundRepr = ocg->CreatePlus(easyBoundRepr, varName);
      }
      else if (cf == -1) {
        easyBoundRepr = ocg->CreateMinus(easyBoundRepr, varName);
      }
    }
  }

  if (g.get_const()) {
    coef_t cf = g.get_const()*sign_adj;
    assert(cf != 0);
    if (cf > 0) {
      easyBoundRepr = ocg->CreatePlus(easyBoundRepr, ocg->CreateInt(cf));
    }
    else {
      easyBoundRepr = ocg->CreateMinus(easyBoundRepr, ocg->CreateInt(-cf));
    }
  }
  else {
    if(easyBoundRepr == NULL) {
      easyBoundRepr = ocg->CreateInt(0);
    }
  }

  if (v_coef > 1) {
    assert(ceiling >= 0);
    if (ceiling) {
      easyBoundRepr= ocg->CreatePlus(easyBoundRepr, ocg->CreateInt(v_coef-1));
    }
    easyBoundRepr = ocg->CreateIntegerDivide(easyBoundRepr, ocg->CreateInt(v_coef));
  }
  
  return easyBoundRepr;
}


//----------------------------------------------------------------------------
// Translate inequality constraints to loop or assignment.
// if return.second is true, return.first is loop structure,
// otherwise it is assignment.
// ----------------------------------------------------------------------------
std::pair<CG_outputRepr *, bool> outputBounds(CG_outputBuilder* ocg, const Relation &bounds, Variable_ID v, int indent, Relation &enforced, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation b = copy(bounds);
  Conjunct *c = b.query_DNF()->single_conjunct();

  // Elaborate stride simplification which is complementary to gist function
  // since we further target the specific loop variable.  -- by chun 08/07/2008
  Relation r1 = Relation::True(b.n_set()), r2 = Relation::True(b.n_set());
  for (EQ_Iterator ei(c); ei; ei++) {
    if ((*ei).get_coef(v) != 0 && (*ei).has_wildcards()) { // stride condition found
      coef_t sign;
      if ((*ei).get_coef(v) > 0)
        sign = 1;
      else
        sign = -1;

      coef_t stride = 0;
      for (Constr_Vars_Iter cvi(*ei, true); cvi; cvi++)
        if ((*cvi).var->kind() == Wildcard_Var) {
          stride = abs((*cvi).coef);
          break;
        }

      // check if stride hits lower bound
      bool found_match = false;
      if (abs((*ei).get_coef(v)) != 1) { // expensive matching for non-clean stride condition
        coef_t d = stride / gcd(abs((*ei).get_coef(v)), stride);
        Relation r3 = Relation::True(b.n_set());
        r3.and_with_EQ(*ei);
        
        for (GEQ_Iterator gi(c); gi; gi++) {
          if ((*gi).get_coef(v) == 1 && !(*gi).has_wildcards()) {
            Relation r4(b.n_set());
            F_And *f_root = r4.add_and();
            Stride_Handle h = f_root->add_stride(d);
            
            for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
              switch ((*cvi).var->kind()) {
              case Input_Var: {
                int pos = (*cvi).var->get_position();
                h.update_coef(r4.set_var(pos), (*cvi).coef);
                break;
              }
              case Global_Var: {
                Global_Var_ID g = (*cvi).var->get_global_var();
                Variable_ID v;
                if (g->arity() == 0)
                  v = r4.get_local(g);
                else
                  v = r4.get_local(g, (*cvi).var->function_of());
                h.update_coef(v, (*cvi).coef);
                break;
              }
              default:
                fprintf(DebugFile, "can't deal with the variable type in lower bound\n");
                return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);
              }
            h.update_const((*gi).get_const());

            Relation r5 = Gist(copy(r3), Intersection(copy(r4), copy(enforced)));

            // replace original stride condition with striding from this lower bound
            if (r5.is_obvious_tautology()) {
              r1 = Intersection(r1, r4);
              found_match = true;
              break;
            }
          }
        }
      }
      else {
        for (GEQ_Iterator gi(c); gi; gi++) {
          if ((*gi).get_coef(v) == abs((*ei).get_coef(v)) && !(*gi).has_wildcards()) { // potential matching lower bound found
            Relation r(b.n_set());
            Stride_Handle h = r.add_and()->add_stride(stride);

            for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
              switch ((*cvi).var->kind()) {
              case Input_Var: {
                int pos = (*cvi).var->get_position();
                if ((*cvi).var != v) {
                  int t1 = int_mod((*cvi).coef, stride);
                  if (t1 != 0) {
                    coef_t t2 = enforced.query_variable_mod(enforced.set_var(pos), stride);
                    if (t2 != posInfinity)
                      h.update_const(t1*t2);
                    else
                      h.update_coef(r.set_var(pos), t1);
                  }
                }
                else
                  h.update_coef(r.set_var(pos), (*cvi).coef);
                break;
              }
              case Global_Var: {
                Global_Var_ID g = (*cvi).var->get_global_var();
                Variable_ID v;
                if (g->arity() == 0)
                  v = enforced.get_local(g);
                else
                  v = enforced.get_local(g, (*cvi).var->function_of());
                coef_t t = enforced.query_variable_mod(v, stride);
                if (t != posInfinity)
                  h.update_const(t*(*cvi).coef);
                else {
                  Variable_ID v2;
                  if (g->arity() == 0)
                    v2 = r.get_local(g);
                  else
                    v2 = r.get_local(g, (*cvi).var->function_of());
                  h.update_coef(v2, (*cvi).coef);
                }
                break;
              }
              default:
                fprintf(DebugFile, "can't deal with the variable type in lower bound\n");
                return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);
              }
            h.update_const((*gi).get_const());

            bool t = true;
            {
              Conjunct *c2 = r.query_DNF()->single_conjunct();
              EQ_Handle h2;
              for (EQ_Iterator ei2(c2); ei2; ei2++) {
                h2 = *ei2;
                break;
              }
                        
              int sign;
              if (h2.get_coef(v) == (*ei).get_coef(v))
                sign = 1;
              else
                sign = -1;

              t = int_mod(h2.get_const() - sign * (*ei).get_const(), stride) == 0;

              if (t != false)
                for (Constr_Vars_Iter cvi(h2); cvi; cvi++)
                  if ((*cvi).var->kind() != Wildcard_Var &&
                      int_mod((*cvi).coef - sign * (*ei).get_coef((*cvi).var), stride) != 0) {
                    t = false;
                    break;
                  }
                  
              if (t != false)
                for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
                  if ((*cvi).var->kind() != Wildcard_Var &&
                      int_mod((*cvi).coef - sign * h2.get_coef((*cvi).var), stride) != 0) {
                    t = false;
                    break;
                  }
              
            }
            
            if (t) {
              // replace original stride condition with striding from this lower bound
              F_And *f_root = r1.and_with_and();
              Stride_Handle h = f_root->add_stride(stride);
              for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
                switch ((*cvi).var->kind()) {
                case Input_Var: {
                  h.update_coef(r1.set_var((*cvi).var->get_position()), (*cvi).coef);
                  break;
                }
                case Global_Var: {
                  Global_Var_ID g = (*cvi).var->get_global_var();
                  Variable_ID v;
                  if (g->arity() == 0)
                    v = r1.get_local(g);
                  else
                    v = r1.get_local(g, (*cvi).var->function_of());
                  h.update_coef(v, (*cvi).coef);
                  break;
                }
                default:
                  fprintf(DebugFile, "can't deal with the variable type in lower bound\n");
                  return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);
                }
              h.update_const((*gi).get_const());
            
              found_match = true;
              break;
            }
          }
        }
      }

      if (!found_match)
        r1.and_with_EQ(*ei);   
    }
    else if ((*ei).get_coef(v) == 0) {
        Relation r3 = Relation::True(b.n_set());
        r3.and_with_EQ(*ei);
        Relation r4 = Gist(r3, copy(enforced));
        if (!r4.is_obvious_tautology())
          r2.and_with_EQ(*ei);
    }
    else 
      r2.and_with_EQ(*ei);
  }
      
  // restore remaining inequalities
  {
    std::map<Variable_ID, Variable_ID> exists_mapping;
    F_Exists *fe = r2.and_with_and()->add_exists();
    F_And *f_root = fe->add_and();
    for (GEQ_Iterator gi(c); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var: {
          int pos = v->get_position();
          h.update_coef(r2.set_var(pos), cvi.curr_coef());
          break;
        }
        case Exists_Var:
        case Wildcard_Var: {
          std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find(v);
          Variable_ID e;
          if (p == exists_mapping.end()) {
            e = fe->declare();
            exists_mapping[v] = e;
          }
          else
            e = (*p).second;
          h.update_coef(e, cvi.curr_coef());
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
          assert(0);
        }
      }
      h.update_const((*gi).get_const());
    }
  }
          
  // overwrite original bounds
  {
    r1.simplify();
    r2.simplify();
    Relation b2 = Intersection(r1, r2);
    b2.simplify();
    for (int i = 1; i <= b.n_set(); i++)
      b2.name_set_var(i, b.set_var(i)->name());
    b2.setup_names();
    b = b2;
    c = b.query_DNF()->single_conjunct();
  }  
    

  // get loop strides
  EQ_Handle strideEQ;
  bool foundStride = false; // stride that can be translated to loop
  bool foundSimpleStride = false; // stride that starts from const value
  coef_t step = 1;
  int num_stride = 0;

  for (EQ_Iterator ei(c); ei; ei++) {
    if ((*ei).get_coef(v) != 0 && (*ei).has_wildcards()) {
      num_stride++;

      if (abs((*ei).get_coef(v)) != 1)
        continue;

      bool t = true;
      coef_t d = 1;
      for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
        if ((*cvi).var->kind() == Wildcard_Var) {
          assert(d==1);
          d = abs((*cvi).coef);
        }
        else if ((*cvi).var->kind() == Input_Var) {
          if ((*cvi).var != v)
            t = false;
        }
        else
          t = false;

      if (d > step) {
        step = d;
        foundSimpleStride = t;
        strideEQ = *ei;
        foundStride = true;
      }
    }
  }

  // More than one stride or complex stride found, we should move all
  // but strideEQ to body's guard condition. alas, not implemented.
  if (!(num_stride == 0 || (num_stride == 1 && foundStride)))
    return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);

  // get loop bounds
  int lower_bounds = 0, upper_bounds = 0;
  Tuple<CG_outputRepr *> lbList;
  Tuple<CG_outputRepr *> ubList;
  coef_t const_lb = negInfinity, const_ub = posInfinity;
  for (GEQ_Iterator g(c); g; g++) {
    coef_t coef = (*g).get_coef(v);
    if (coef == 0) 
      continue;
    else if (coef > 0) { // lower bound
      lower_bounds++;
      if ((*g).is_const(v) && !foundStride) { 
        //no variables but v in constr
        coef_t L,m;
        L = -((*g).get_const());
 
        m = (*g).get_coef(v);
        coef_t sb  =  (int) (ceil(((float) L) /m));
        set_max(const_lb, sb);
      }
      else if ((*g).is_const(v) && foundSimpleStride) { 
        // no variables but v in constr
        //make LB fit the stride constraint
        coef_t L,m,s,c;
        L = -((*g).get_const());
        m = (*g).get_coef(v);
        s = step;
        c = strideEQ.get_const();
        coef_t sb  =  (s * (int) (ceil( (float) (L - (c * m)) /(s*m))))+ c;
        set_max(const_lb, sb);
      } 
      else 
        lbList.append(outputLBasRepr(ocg, *g, b, v, step, strideEQ, enforced, assigned_on_the_fly));
    }
    else {  // upper bound
      upper_bounds++;
      if ((*g).is_const(v)) { 
        // no variables but v in constraint
        set_min(const_ub,-(*g).get_const()/(*g).get_coef(v));
      }
      else
        ubList.append(outputUBasRepr(ocg, *g, b, v, step, strideEQ, assigned_on_the_fly));
    }
  }

  CG_outputRepr *lbRepr = NULL;
  CG_outputRepr *ubRepr = NULL;
  if (const_lb != negInfinity)
    lbList.append(ocg->CreateInt(const_lb));    
  if (lbList.size() > 1)
    lbRepr = ocg->CreateInvoke("max", lbList);
  else if (lbList.size() == 1)
    lbRepr = lbList[1];

  //protonu
    if(fillInBounds && lbList.size() == 1 && const_lb != negInfinity)
    lowerBoundForLevel = const_lb;
  //end-protonu

  if (const_ub != posInfinity)
    ubList.append(ocg->CreateInt(const_ub));
  if (ubList.size() > 1)
    ubRepr = ocg->CreateInvoke("min", ubList);
  else if (ubList.size() == 1)
    ubRepr = ubList[1];

  //protonu
   if(fillInBounds && const_ub != posInfinity)
    upperBoundForLevel = const_ub;
 //end-protonu

  if (upper_bounds == 0 || lower_bounds == 0) {
    return std::make_pair(static_cast<CG_outputRepr *>(NULL), false);
  }
  else {
    // bookkeeping catched constraints in new_knwon
    F_Exists *fe = enforced.and_with_and()->add_exists();
    F_And *f_root = fe->add_and();
    std::map<Variable_ID, Variable_ID> exists_mapping;
    std::stack<std::pair<GEQ_Handle, Variable_ID> > floor_geq_stack;
    std::set<Variable_ID> floor_var_set;

    if (foundStride) {
      EQ_Handle h = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(strideEQ); cvi; cvi++)
        switch ((*cvi).var->kind()) {
        case Input_Var: {
          int pos = (*cvi).var->get_position();
          h.update_coef(enforced.set_var(pos), (*cvi).coef);
          break;
        }
        case Exists_Var:
        case Wildcard_Var: {
          std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find((*cvi).var);
          Variable_ID e;
          if (p == exists_mapping.end()) {
            e = fe->declare();
            exists_mapping[(*cvi).var] = e;
          }
          else
            e = (*p).second;
          h.update_coef(e, (*cvi).coef);
          break;
        }
        case Global_Var: {
          Global_Var_ID g = (*cvi).var->get_global_var();
          Variable_ID e;
          if (g->arity() == 0)
            e = enforced.get_local(g);
          else
            e = enforced.get_local(g, (*cvi).var->function_of());
          h.update_coef(e, (*cvi).coef);
          break;
        }
        default:
          assert(0);
        }
      h.update_const(strideEQ.get_const());
    }
    
    for (GEQ_Iterator gi(c); gi; gi++)
      if ((*gi).get_coef(v) != 0) {
        GEQ_Handle h = f_root->add_GEQ();
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++)
          switch ((*cvi).var->kind()) {
          case Input_Var: {
            int pos = (*cvi).var->get_position();
            h.update_coef(enforced.set_var(pos), (*cvi).coef);
            break;
          }
          case Exists_Var:
          case Wildcard_Var: {
            std::map<Variable_ID, Variable_ID>::iterator p = exists_mapping.find((*cvi).var);
            Variable_ID e;
            if (p == exists_mapping.end()) {
              e = fe->declare();
              exists_mapping[(*cvi).var] = e;
            }
            else
              e = (*p).second;
            h.update_coef(e, (*cvi).coef);

            if (floor_var_set.find((*cvi).var) == floor_var_set.end()) {
              GEQ_Handle h2;
              findFloorInequality(b, (*cvi).var, h2, v);
              floor_geq_stack.push(std::make_pair(h2, (*cvi).var));
              floor_var_set.insert((*cvi).var);
            }
            break;
          }
          case Global_Var: {
            Global_Var_ID g = (*cvi).var->get_global_var();
            Variable_ID e;
            if (g->arity() == 0)
              e = enforced.get_local(g);
            else
              e = enforced.get_local(g, (*cvi).var->function_of());
            h.update_coef(e, (*cvi).coef);
            break;
          }
          default:
            assert(0);
          }
        h.update_const((*gi).get_const());
      }

    // add floor definition involving variables appeared in bounds
    while (!floor_geq_stack.empty()) {
      std::pair<GEQ_Handle, Variable_ID> p = floor_geq_stack.top();
      floor_geq_stack.pop();

      GEQ_Handle h1 = f_root->add_GEQ();
      GEQ_Handle h2 = f_root->add_GEQ();
      for (Constr_Vars_Iter cvi(p.first); cvi; cvi++) {
        switch  ((*cvi).var->kind()) {
        case Input_Var: {
          int pos = (*cvi).var->get_position();
          h1.update_coef(enforced.input_var(pos), (*cvi).coef);
          h2.update_coef(enforced.input_var(pos), -(*cvi).coef);
          break;
        }
        case Exists_Var:
        case Wildcard_Var: {
          std::map<Variable_ID, Variable_ID>::iterator p2 = exists_mapping.find((*cvi).var);
          Variable_ID e;
          if (p2 == exists_mapping.end()) {
            e = fe->declare();
            exists_mapping[(*cvi).var] = e;
          }
          else
            e = (*p2).second;
          h1.update_coef(e, (*cvi).coef);
          h2.update_coef(e, -(*cvi).coef);

          if (floor_var_set.find((*cvi).var) == floor_var_set.end()) {
            GEQ_Handle h3;
            findFloorInequality(b, (*cvi).var, h3, v);
            floor_geq_stack.push(std::make_pair(h3, (*cvi).var));
            floor_var_set.insert((*cvi).var);
          }
          break;
        }
        case Global_Var: {
          Global_Var_ID g = (*cvi).var->get_global_var();
          Variable_ID e;
          if (g->arity() == 0)
            e = enforced.get_local(g);
          else
            e = enforced.get_local(g, (*cvi).var->function_of());
          h1.update_coef(e, (*cvi).coef);
          h2.update_coef(e, -(*cvi).coef);
          break;
        }
        default:
          assert(0);
        }
      }
      h1.update_const(p.first.get_const());
      h2.update_const(-p.first.get_const());
      h2.update_const(-p.first.get_coef(p.second)-1);
    }
    enforced.simplify();
    
    CG_outputRepr *stRepr = NULL;
    if (step != 1)
      stRepr = ocg->CreateInt(abs(step));
    CG_outputRepr *indexRepr = outputIdent(ocg, b, v, assigned_on_the_fly);
    CG_outputRepr *ctrlRepr = ocg->CreateInductive(indexRepr, lbRepr, ubRepr, stRepr);
    
    return std::make_pair(ctrlRepr, true);
  }
}


Relation project_onto_levels(Relation R, int last_level, bool wildcards) {
  assert(last_level >= 0 && R.is_set() && last_level <= R.n_set());
  if (last_level == R.n_set()) return R;

  int orig_vars = R.n_set();
  int num_projected = orig_vars - last_level;
  R = Extend_Set(R,num_projected
    );  // Project out vars numbered > last_level
  Mapping m1 = Mapping::Identity(R.n_set());  // now orig_vars+num_proj

  for(int i=last_level+1; i <= orig_vars; i++) {
    m1.set_map(Set_Var, i, Exists_Var, i);
    m1.set_map(Set_Var, i+num_projected, Set_Var, i);
  }

  MapRel1(R, m1, Comb_Id);
  R.finalize();
  R.simplify();
  if (!wildcards) 
    R = Approximate(R,1);
  assert(R.is_set());
  return R;
}


// Check if the lower bound already enforces the stride by
// (Where m is coef of v in g and L is the bound on m*v):
// Check if m divides L evenly and Check if this l.bound on v implies strideEQ 
bool boundHitsStride(const GEQ_Handle &g, Variable_ID v,
                            const EQ_Handle &strideEQ,
                            coef_t /*stride*/, // currently unused
                            Relation known) {
/* m = coef of v in g;
   L = bound on v part of g;
*/ 
  // Check if m divides L evenly
  coef_t m = g.get_coef(v);
  Relation test(known.n_set());
  F_Exists *e = test.add_exists();       // g is "L >= mv"
  Variable_ID alpha = e->declare();      // want: "l = m alpha"
  F_And *a = e->add_and();
  EQ_Handle h = a->add_EQ(); 
  for(Constr_Vars_Iter I(g,false); I; I++)
    if((*I).var != v) {
      if((*I).var->kind() != Global_Var)
        h.update_coef((*I).var, (*I).coef);
      else
        h.update_coef(test.get_local((*I).var->get_global_var()), (*I).coef);
    }

  h.update_const(g.get_const());
  h.update_coef(alpha,m);                // set alpha's coef to m
  if (!(Gist(test,copy(known)).is_obvious_tautology()))      
    return false;
  // Check if this lower bound on v implies the strideEQ 
  Relation boundRel = known;    // want: "known and l = m v"
  boundRel.and_with_EQ(g);      // add in l = mv
  Relation strideRel(known.n_set());
  strideRel.and_with_EQ(strideEQ);
  return Gist(strideRel, boundRel).is_obvious_tautology();
}


// // Return true if there are no variables in g except wildcards & v
bool isSimpleStride(const EQ_Handle &g, Variable_ID v) {
  EQ_Handle gg = g;  // should not be necessary, but iterators are
  // a bit brain-dammaged
  bool is_simple=true;
  for(Constr_Vars_Iter cvi(gg, false); cvi && is_simple; cvi++)
    is_simple = ((*cvi).coef == 0 || (*cvi).var == v 
                 || (*cvi).var->kind() == Wildcard_Var);
  return is_simple;
}


int countStrides(Conjunct *c, Variable_ID v, EQ_Handle &strideEQ, 
                 bool &simple) {
  int strides=0;
  for(EQ_Iterator G(c); G; G++)
    for(Constr_Vars_Iter I(*G, true); I; I++)
      if (((*I).coef != 0) && (*G).get_coef(v) != 0) {
        strides++;
        simple = isSimpleStride(*G,v);
        strideEQ = *G;
        break;
      }
  return strides;
}

namespace {

bool hasEQ(Relation r, int level) {
  r.simplify();
  Variable_ID v = set_var(level);
  Conjunct *s_conj = r.single_conjunct();
  for(EQ_Iterator G(s_conj); G; G++)
    if ((*G).get_coef(v))
      return true;
  return false;
}



static Relation pickEQ(Relation r, int level) {
  r.simplify();
  Variable_ID v = set_var(level);
  Conjunct *s_conj = r.single_conjunct();
  for(EQ_Iterator E(s_conj); E; E++)
    if ((*E).get_coef(v)) {
      Relation test_rel(r.n_set());
      test_rel.and_with_EQ(*E);
      return test_rel;
    }
  assert(0);
  return r;
}

/* pickBound will return an EQ as a GEQ if it finds one */
Relation pickBound(Relation r, int level, int UB) {
  r.simplify();
  Variable_ID v = set_var(level);
  Conjunct *s_conj = r.single_conjunct();
  for(GEQ_Iterator G(s_conj); G; G++) {
    if ((UB && (*G).get_coef(v) < 0)
        ||  (!UB && (*G).get_coef(v) > 0) ) {
      Relation test_rel(r.n_set());
      test_rel.and_with_GEQ(*G);
      return test_rel;
    }
  }
  for(EQ_Iterator E(s_conj); E; E++) {
    if ((*E).get_coef(v)) {
      Relation test_rel(r.n_set());
      test_rel.and_with_GEQ(*E);
      if ((UB && (*E).get_coef(v) > 0)
          ||  (!UB && (*E).get_coef(v) < 0) ) 
        test_rel = Complement(test_rel);
      return test_rel;
    }
  }
  assert(0);
  return r;
}

}

Relation pickOverhead(Relation r, int liftTo) {
  r.simplify();
  Conjunct *s_conj = r.single_conjunct();
  for(GEQ_Iterator G(s_conj); G; G++) {
    Relation test_rel(r.n_set());
    test_rel.and_with_GEQ(*G);
    Variable_ID v;
    coef_t pos = -1;
    coef_t c= 0;
    for(Constr_Vars_Iter cvi(*G, false); cvi; cvi++) 
      if ((*cvi).coef && (*cvi).var->kind() == Input_Var 
          && (*cvi).var->get_position() > pos) {
        v = (*cvi).var;
        pos = (*cvi).var->get_position();
        c = (*cvi).coef;
      }
#if 0
    fprintf(DebugFile,"Coef = %d, constraint = %s\n",
            c,(const char *)test_rel.print_formula_to_string());
#endif
    return test_rel;
  }
  for(EQ_Iterator E(s_conj); E; E++) {
    assert(liftTo >= 1);
    int pos = max((*E).max_tuple_pos(),max_fs_arity(*E)+1);
 
/* Pick stride constraints only when the variables with stride are outer
   loop variables */
    if ((*E).has_wildcards()  && pos < liftTo) {
      Relation test_rel(r.n_set());
      test_rel.and_with_EQ(*E);
      return test_rel;
    }
    else if (!(*E).has_wildcards()  && pos <= liftTo) {
      Relation test_rel(r.n_set());
      test_rel.and_with_EQ(*E);
      test_rel.simplify();
      test_rel = EQs_to_GEQs(test_rel,true);
      return pickOverhead(test_rel,liftTo);
    }
  }
  if (code_gen_debug>1) {
    fprintf(DebugFile,"Could not find overhead:\n");
    r.prefix_print(DebugFile);
  }
  return Relation::True(r.n_set());
}



bool hasBound(Relation r, int level, int UB) {
  r.simplify();
  Variable_ID v = set_var(level);
  Conjunct *s_conj = r.single_conjunct();
  for(GEQ_Iterator G(s_conj); G; G++) {
    if (UB && (*G).get_coef(v) < 0) return true;
    if (!UB && (*G).get_coef(v) > 0) return true;
  }
  for(EQ_Iterator E(s_conj); E; E++) {
    if ((*E).get_coef(v)) return true;
  }
  return false;
}

bool find_any_constraint(int s, int level, Relation &kr, int direction,
                         Relation &S, bool approx) {
  /* If we don't intersect I with restrictions, the combination 
     of S and restrictions can be unsatisfiable, which means that
     the new split node gets pruned away and we still don't have
     finite bounds -> infinite recursion. */

  Relation I = projected_nIS[level][s];
  I = Gist(I,copy(kr));
  if(approx) I = Approximate(I);
  if (hasBound(I,level,direction)) {
    Relation pickfrom;
    if(has_nonstride_EQ(I,level))
      pickfrom = pickEQ(I,level);
    else 
      pickfrom = pickBound(I,level,direction);
    S = pickOverhead(pickfrom,level);
    if(S.is_obvious_tautology()) S = Relation::Null();
    return !S.is_null();
  }
  return false;
}


bool has_nonstride_EQ(Relation r, int level) {
  r.simplify();
  Variable_ID v = set_var(level);
  Conjunct *s_conj = r.single_conjunct();
  for(EQ_Iterator G(s_conj); G; G++)
    if ((*G).get_coef(v) && !(*G).has_wildcards())
      return true;
  return false;
}


Relation minMaxOverhead(Relation r, int level) {
  r.finalize();
  r.simplify();
  Conjunct *s_conj = r.single_conjunct();
  GEQ_Handle LBs[50],UBs[50];
  int numLBs = 0;
  int numUBs = 0;
  Variable_ID v = set_var(level);
  for(GEQ_Iterator G(s_conj); G; G++) if ((*G).get_coef(v)) {
      GEQ_Handle g = *G;
      if (g.get_coef(v) > 0) LBs[numLBs++] = g;
      else UBs[numUBs++] = g;
    }
  if (numLBs <= 1 && numUBs <= 1) {
    return Relation::True(r.n_set());
  }
  Relation r1(r.n_set());
  Relation r2(r.n_set());
  if (numLBs > 1) {
    // remove a max in lower bound
    r1.and_with_GEQ(LBs[0]);
    r2.and_with_GEQ(LBs[1]);
    r1 = project_onto_levels(Difference(r1,r2),level-1,0);
  }
  else {
    // remove a min in upper bound
    r1.and_with_GEQ(UBs[0]);
    r2.and_with_GEQ(UBs[1]);
    r1 = project_onto_levels(Difference(r1,r2),level-1,0);
  }
#if 0
  fprintf(DebugFile,"Testing %s\n",(const char *)r1.print_formula_to_string());
  fprintf(DebugFile,"will removed overhead on bounds of t%d: %s\n",level,
          (const char *)r.print_formula_to_string());
#endif
   
  return pickOverhead(r1, -1);
}

std::pair<EQ_Handle, int> find_simplest_assignment(const Relation &R_, Variable_ID v, const std::vector<CG_outputRepr *> &assigned_on_the_fly) {
  Relation &R = const_cast<Relation &>(R_);
  Conjunct *c = R.single_conjunct();
  
  int min_cost = INT_MAX;
  EQ_Handle eq;
  for (EQ_Iterator ei(c->EQs()); ei; ei++)
    if (!(*ei).has_wildcards() && (*ei).get_coef(v) != 0) {
      int cost = 0;

      if (abs((*ei).get_coef(v)) != 1)
        cost += 4;  // divide cost

      int num_var = 0;
      for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
        if ((*cvi).var != v) {
          num_var++;
          if ((*cvi).var->kind() == Global_Var && (*cvi).var->get_global_var()->arity() > 0) {
            cost += 10;  // function cost
          }
          if (abs((*cvi).coef) != 1)
            cost += 2;  // multiply cost
          if ((*cvi).var->kind() == Input_Var && assigned_on_the_fly[(*cvi).var->get_position()-1] != NULL) {
            cost += 5;  // substituted variable cost
          }
        }
      if ((*ei).get_const() != 0)
        num_var++;
      if (num_var > 1)
        cost += num_var - 1; // addition cost

      if (cost < min_cost) {
        min_cost = cost;
        eq = *ei;
      }
    }

  return std::make_pair(eq, min_cost);
}

int max_fs_arity(const Constraint_Handle &c) {
  int max_arity=0;
  for(Constr_Vars_Iter cv(c); cv; cv++)
    if((*cv).var->kind() == Global_Var)
      max_arity = max(max_arity,(*cv).var->get_global_var()->arity());
  return max_arity;
}

}
