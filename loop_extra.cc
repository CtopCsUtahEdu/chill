/*****************************************************************************
 Copyright (C) 2010 University of Utah
 All Rights Reserved.

 Purpose:
   Additional loop transformations.

 Notes:

 History:
   07/31/10 Created by Chun Chen
*****************************************************************************/

#include <codegen.h>
#include <code_gen/CG_utils.h>
#include "loop.hh"
#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"

using namespace omega;


void Loop::shift_to(int stmt_num, int level, int absolute_position) {
  // combo
  tile(stmt_num, level, 1, level, CountedTile);
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> active = getStatements(lex, 2*level-2);
  shift(active, level, absolute_position);
  
  // remove unnecessary tiled loop since tile size is one
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].xform.n_out();
    Relation mapping(n, n-2);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= 2*level; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = 2*level+3; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j-2), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    stmt[*i].xform = Composition(mapping, stmt[*i].xform);
    stmt[*i].xform.simplify();
    
    for (int j = 0; j < stmt[*i].loop_level.size(); j++)
      if (j != level-1 &&
          stmt[*i].loop_level[j].type == LoopLevelTile &&
          stmt[*i].loop_level[j].payload >= level)
        stmt[*i].loop_level[j].payload--;
    
    stmt[*i].loop_level.erase(stmt[*i].loop_level.begin()+level-1);
  }
}


std::set<int> Loop::unroll_extra(int stmt_num, int level, int unroll_amount, int cleanup_split_level) {
  std::set<int> cleanup_stmts = unroll(stmt_num, level, unroll_amount,std::vector< std::vector<std::string> >(), cleanup_split_level);
  for (std::set<int>::iterator i = cleanup_stmts.begin(); i != cleanup_stmts.end(); i++)
    unroll(*i, level, 0);
  
  return cleanup_stmts;
}





void Loop::peel(int stmt_num, int level, int peel_amount) {
  debug_fprintf(stderr, "\n\nloop_extra.cc\n*** Loop::peel( stmt_num %d, level %d, amount %d)\n", stmt_num, level, peel_amount); 


  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));

  debug_fprintf(stderr, "peel amount %d\n", peel_amount);
  
  if (peel_amount == 0) {
    debug_fprintf(stderr, "peel amount is zero???\n\n\n\n\n\n\n\n"); 
    return;
  }

  std::set<int> subloop = getSubLoopNest(stmt_num, level);
  std::vector<Relation> Rs;
  int sl = 0; 
  for (std::set<int>::iterator i = subloop.begin(); i != subloop.end(); i++) {
    debug_fprintf(stderr, "\nSUBLOOP %d\n", sl);

    Relation r = getNewIS(*i);
    //r.print(); fflush(stdout); 

    Relation f(r.n_set(), level);
    //f.print(); fflush(stdout); 
    F_And *f_root = f.add_and();
    for (int j = 1; j <= level; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(f.input_var(2*j), 1);
      h.update_coef(f.output_var(j), -1);
    }
    //r.print(); fflush(stdout); 
    //f.print();  fflush(stdout); 

    //Anand composition will fail due to unintepreted function symbols introduced by flattening
		//r = Composition(f, r);
    r = omega::Range(Restrict_Domain(f, r));
    r.simplify();
    Rs.push_back(r);

    sl++; 
  }

  Relation hull = SimpleHull(Rs); 
    
  if (peel_amount > 0) {
    debug_fprintf(stderr, "\n*** peel from beginning of loop\n"); 
    GEQ_Handle bound_eq;
    bool found_bound = false;
    for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++) { 
      if (!(*e).has_wildcards() && (*e).get_coef(hull.set_var(level)) > 0) {
        bound_eq = *e;
        found_bound = true;
        break;
      }
    }

    if (found_bound) debug_fprintf(stderr, "beginning of loop, peel after first,  found bound\n");
    else  debug_fprintf(stderr, "beginning of loop, peel after first,  NOT found bound\n");
    
    if (!found_bound) { 
      for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++)
        if ((*e).has_wildcards() && (*e).get_coef(hull.set_var(level)) > 0) {
          bool is_bound = true;
          for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
            std::pair<bool, GEQ_Handle> result = find_floor_definition(hull, cvi.curr_var());
            if (!result.first) {
              is_bound = false;
              break;
            }
          }
          if (is_bound) {
            bound_eq = *e;
            found_bound = true;
            break;
          }
        }
    }
    
    if (found_bound) debug_fprintf(stderr, "beginning of loop, peel after second, found bound\n");
    else  debug_fprintf(stderr, "beginning of loop, peel after second, NOT found bound\n");
    

    if (!found_bound)
      throw loop_error("can't find lower bound for peeling at loop level " + to_string(level));
    
    for (int i = 1; i <= peel_amount; i++) {
      debug_fprintf(stderr, "peeling statement %d\n", i);
      
      Relation r(level);
      F_Exists *f_exists = r.add_and()->add_exists();
      F_And *f_root = f_exists->add_and();
      GEQ_Handle h = f_root->add_GEQ();
      std::map<Variable_ID, Variable_ID> exists_mapping;
      for (Constr_Vars_Iter cvi(bound_eq); cvi; cvi++)
        switch (cvi.curr_var()->kind()) {
        case Input_Var:
          h.update_coef(r.set_var(cvi.curr_var()->get_position()), cvi.curr_coef());
          break;
        case Wildcard_Var: {
          Variable_ID v = replicate_floor_definition(hull, cvi.curr_var(), r, f_exists, f_root, exists_mapping);
          h.update_coef(v, cvi.curr_coef());
          break;
        }
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
      h.update_const(bound_eq.get_const() - i);
      r.simplify();

      debug_fprintf(stderr, "loop_extra.cc peel() calling split()\n"); 
      split(stmt_num, level, r);
      debug_fprintf(stderr, "loop_extra.cc peel() DONE calling split()\n"); 
    }
  }
  else { // peel_amount < 0
    debug_fprintf(stderr, "\n*** peel from end of loop\n");
    //debug_fprintf(stderr, "*** NOT DOING THAT. SOMETHING ELSE IS DOING THE FRONT PEEL\n"); 

    GEQ_Handle bound_eq;
    bool found_bound = false;
    for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++) {
      if (!(*e).has_wildcards() && (*e).get_coef(hull.set_var(level)) < 0) {
        bound_eq = *e;
        found_bound = true;
        break;
      }
    }

    if (found_bound) debug_fprintf(stderr, "end of loop, peel after first,  found bound\n");
    else  debug_fprintf(stderr, "end of loop, peel after first,  NOT found bound  (will try again) \n");
       
    if (!found_bound) { 
      for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++)
        if ((*e).has_wildcards() && (*e).get_coef(hull.set_var(level)) < 0) {
          bool is_bound = true;
          for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
            std::pair<bool, GEQ_Handle> result = find_floor_definition(hull, cvi.curr_var());
            if (!result.first) {
              is_bound = false;
              break;
            }
          }
          if (is_bound) {
            bound_eq = *e;
            found_bound = true;
            break;
          }
        }
    }

    if (found_bound) debug_fprintf(stderr, "end of loop, peel after second, found bound\n");
    else  debug_fprintf(stderr, "end of loop, peel after second, NOT found bound\n");
    
   
    if (!found_bound)
      throw loop_error("can't find upper bound for peeling at loop level " + to_string(level));
    
    for (int i = 1; i <= -peel_amount; i++) {
      debug_fprintf(stderr, "\npeel i %d\n", i);
      
      Relation r(level);
      F_Exists *f_exists = r.add_and()->add_exists();
      F_And *f_root = f_exists->add_and();
      GEQ_Handle h = f_root->add_GEQ();
      std::map<Variable_ID, Variable_ID> exists_mapping;
      for (Constr_Vars_Iter cvi(bound_eq); cvi; cvi++)
        switch (cvi.curr_var()->kind()) {
        case Input_Var:
          h.update_coef(r.set_var(cvi.curr_var()->get_position()), cvi.curr_coef());
          break;
        case Wildcard_Var: {
          Variable_ID v = replicate_floor_definition(hull, cvi.curr_var(), r, f_exists, f_root, exists_mapping);
          h.update_coef(v, cvi.curr_coef());
          break;
        }
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
      h.update_const(bound_eq.get_const() - i);
      r.simplify();
      
      split(stmt_num, level, r);
    }

  }


  // we just made a change to the code. invalidate the previous generated code
  invalidateCodeGen();

  debug_fprintf(stderr, "loop_extra.cc peel() DONE\n\n\n");

  
}

