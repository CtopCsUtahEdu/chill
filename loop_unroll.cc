/*
 * loop_unroll.cc
 *
 *  Created on: Nov 12, 2012
 *      Author: anand
 */

#include <codegen.h>
#include <code_gen/CG_utils.h>
#include "loop.hh"
#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"
#include <math.h>

using namespace omega;


std::set<int> Loop::unroll(int stmt_num, int level, int unroll_amount,
                           std::vector<std::vector<std::string> > idxNames,
                           int cleanup_split_level) {
  // check for sanity of parameters
  // check for sanity of parameters
  if (unroll_amount < 0)
    throw std::invalid_argument(
      "invalid unroll amount " + to_string(unroll_amount));
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  if (cleanup_split_level == 0)
    cleanup_split_level = level;
  if (cleanup_split_level > level)
    throw std::invalid_argument(
      "cleanup code must be split at or outside the unrolled loop level "
      + to_string(level));
  if (cleanup_split_level <= 0)
    throw std::invalid_argument(
      "invalid split loop level " + to_string(cleanup_split_level));
  
  // invalidate saved codegen computation
  invalidateCodeGen();

  int dim = 2 * level - 1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  // nothing to do
  if (unroll_amount == 1)
    return std::set<int>();
  
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end();
       i++) {
    std::vector<std::pair<int, DependenceVector> > D;
    int n = stmt[*i].xform.n_out();
    for (DependenceGraph::EdgeList::iterator j =
           dep.vertex[*i].second.begin(); j != dep.vertex[*i].second.end();
         j++) {
      if (same_loop.find(j->first) != same_loop.end())
        for (int k = 0; k < j->second.size(); k++) {
          DependenceVector dv = j->second[k];
          int dim2 = level - 1;
          if (dv.type != DEP_CONTROL) {
            
            while (stmt[*i].loop_level[dim2].type == LoopLevelTile) {
              dim2 = stmt[*i].loop_level[dim2].payload - 1;
            }
            dim2 = stmt[*i].loop_level[dim2].payload;
            
            /*if (dv.isCarried(dim2)
              && (dv.hasNegative(dim2) && !dv.quasi))
              throw loop_error(
              "loop error: Unrolling is illegal, dependence violation!");
              
              if (dv.isCarried(dim2)
              && (dv.hasPositive(dim2) && dv.quasi))
              throw loop_error(
              "loop error: Unrolling is illegal, dependence violation!");
            */
            bool safe = false;
            
            if (dv.isCarried(dim2) && dv.hasPositive(dim2)) {
              if (dv.quasi)
                throw loop_error(
                  "loop error: a quasi dependence with a positive carried distance");
              if (!dv.quasi) {
                if (dv.lbounds[dim2] != posInfinity) {
                  //if (dv.lbounds[dim2] != negInfinity)
                  if (dv.lbounds[dim2] > unroll_amount)
                    safe = true;
                } else
                  safe = true;
              }/* else {
                  if (dv.ubounds[dim2] != negInfinity) {
                  if (dv.ubounds[dim2] != posInfinity)
                  if ((-(dv.ubounds[dim2])) > unroll_amount)
                  safe = true;
                  } else
                  safe = true;
                  }*/
              
              if (!safe) {
                for (int l = level + 1; l <= (n - 1) / 2; l++) {
                  int dim3 = l - 1;
                  
                  if (stmt[*i].loop_level[dim3].type
                      != LoopLevelTile)
                    dim3 =
                      stmt[*i].loop_level[dim3].payload;
                  else {
                    while (stmt[*i].loop_level[dim3].type
                           == LoopLevelTile) {
                      dim3 =
                        stmt[*i].loop_level[dim3].payload
                        - 1;
                    }
                    dim3 =
                      stmt[*i].loop_level[dim3].payload;
                  }
                  
                  if (dim3 > dim2) {
                    
                    if (dv.hasPositive(dim3))
                      break;
                    else if (dv.hasNegative(dim3))
                      throw loop_error(
                        "loop error: Unrolling is illegal, dependence violation!");
                  }
                }
              }
            }
          }
        }
    }
  }
  // extract the intersection of the iteration space to be considered
  Relation hull = Relation::True(level);
  apply_xform(same_loop);
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end();
       i++) {
    if (stmt[*i].IS.is_upper_bound_satisfiable()) {
      Relation mapping(stmt[*i].IS.n_set(), level);
      F_And *f_root = mapping.add_and();
      for (int j = 1; j <= level; j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(j), 1);
        h.update_coef(mapping.output_var(j), -1);
      }
      hull = Intersection(hull,
                          omega::Range(Restrict_Domain(mapping, copy(stmt[*i].IS))));
      hull.simplify(2, 4);
      
    }
  }
  for (int i = 1; i <= level; i++) {
    std::string name = tmp_loop_var_name_prefix + to_string(i);
    hull.name_set_var(i, name);
  }
  hull.setup_names();
  
  // extract the exact loop bound of the dimension to be unrolled
  if (is_single_loop_iteration(hull, level, this->known))
    return std::set<int>();
  Relation bound = get_loop_bound(hull, level, this->known);
  if (!bound.has_single_conjunct() || !bound.is_satisfiable()
      || bound.is_tautology())
    throw loop_error("unable to extract loop bound for unrolling");
  
  // extract the loop stride
  coef_t stride;
  std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(bound,
                                                                  bound.set_var(level));
  if (result.second == NULL)
    stride = 1;
  else
    stride = abs(result.first.get_coef(result.second))
      / gcd(abs(result.first.get_coef(result.second)),
            abs(result.first.get_coef(bound.set_var(level))));
  
  // separate lower and upper bounds
  std::vector<GEQ_Handle> lb_list, ub_list;
  {
    Conjunct *c = bound.query_DNF()->single_conjunct();
    for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
      int coef = (*gi).get_coef(bound.set_var(level));
      if (coef < 0)
        ub_list.push_back(*gi);
      else if (coef > 0)
        lb_list.push_back(*gi);
    }
  }
  
  // simplify overflow expression for each pair of upper and lower bounds
  std::vector<std::vector<std::map<Variable_ID, int> > > overflow_table(
    lb_list.size(),
    std::vector<std::map<Variable_ID, int> >(ub_list.size(),
                                             std::map<Variable_ID, int>()));
  bool is_overflow_simplifiable = true;
  for (int i = 0; i < lb_list.size(); i++) {
    if (!is_overflow_simplifiable)
      break;
    
    for (int j = 0; j < ub_list.size(); j++) {
      // lower bound or upper bound has non-unit coefficient, can't simplify
      if (ub_list[j].get_coef(bound.set_var(level)) != -1
          || lb_list[i].get_coef(bound.set_var(level)) != 1) {
        is_overflow_simplifiable = false;
        break;
      }
      
      for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
        switch ((*ci).var->kind()) {
        case Input_Var: {
          if ((*ci).var != bound.set_var(level))
            overflow_table[i][j][(*ci).var] += (*ci).coef;
          
          break;
        }
        case Global_Var: {
          Global_Var_ID g = (*ci).var->get_global_var();
          Variable_ID v;
          if (g->arity() == 0)
            v = bound.get_local(g);
          else
            v = bound.get_local(g, (*ci).var->function_of());
          overflow_table[i][j][(*ci).var] += (*ci).coef;
          break;
        }
        default:
          throw loop_error("failed to calculate overflow amount");
        }
      }
      overflow_table[i][j][NULL] += ub_list[j].get_const();
      
      for (Constr_Vars_Iter ci(lb_list[i]); ci; ci++) {
        switch ((*ci).var->kind()) {
        case Input_Var: {
          if ((*ci).var != bound.set_var(level)) {
            overflow_table[i][j][(*ci).var] += (*ci).coef;
            if (overflow_table[i][j][(*ci).var] == 0)
              overflow_table[i][j].erase(
                overflow_table[i][j].find((*ci).var));
          }
          break;
        }
        case Global_Var: {
          Global_Var_ID g = (*ci).var->get_global_var();
          Variable_ID v;
          if (g->arity() == 0)
            v = bound.get_local(g);
          else
            v = bound.get_local(g, (*ci).var->function_of());
          overflow_table[i][j][(*ci).var] += (*ci).coef;
          if (overflow_table[i][j][(*ci).var] == 0)
            overflow_table[i][j].erase(
              overflow_table[i][j].find((*ci).var));
          break;
        }
        default:
          throw loop_error("failed to calculate overflow amount");
        }
      }
      overflow_table[i][j][NULL] += lb_list[i].get_const();
      
      overflow_table[i][j][NULL] += stride;
      if (unroll_amount == 0
          || (overflow_table[i][j].size() == 1
              && overflow_table[i][j][NULL] / stride
              < unroll_amount))
        unroll_amount = overflow_table[i][j][NULL] / stride;
    }
  }
  
  // loop iteration count can't be determined, bail out gracefully
  if (unroll_amount == 0)
    return std::set<int>();
  
  // further simply overflow calculation using coefficients' modular
  if (is_overflow_simplifiable) {
    for (int i = 0; i < lb_list.size(); i++)
      for (int j = 0; j < ub_list.size(); j++)
        if (stride == 1) {
          for (std::map<Variable_ID, int>::iterator k =
                 overflow_table[i][j].begin();
               k != overflow_table[i][j].end();)
            if ((*k).first != NULL) {
              int t = int_mod_hat((*k).second, unroll_amount);
              if (t == 0) {
                overflow_table[i][j].erase(k++);
              } else {
                int t2 = hull.query_variable_mod((*k).first,
                                                 unroll_amount);
                if (t2 != INT_MAX) {
                  overflow_table[i][j][NULL] += t * t2;
                  overflow_table[i][j].erase(k++);
                } else {
                  (*k).second = t;
                  k++;
                }
              }
            } else
              k++;
          
          overflow_table[i][j][NULL] = int_mod_hat(
            overflow_table[i][j][NULL], unroll_amount);
          
          // Since we don't have MODULO instruction in SUIF yet (only MOD), 
          // make all coef positive in the final formula
          for (std::map<Variable_ID, int>::iterator k =
                 overflow_table[i][j].begin();
               k != overflow_table[i][j].end(); k++)
            if ((*k).second < 0)
              (*k).second += unroll_amount;
        }
  }
  
  // build overflow statement
  CG_outputBuilder *ocg = ir->builder();
  CG_outputRepr *overflow_code = NULL;
  Relation cond_upper(level), cond_lower(level);
  Relation overflow_constraint(0);
  F_And *overflow_constraint_root = overflow_constraint.add_and();
  std::vector<Free_Var_Decl *> over_var_list;
  if (is_overflow_simplifiable && lb_list.size() == 1) {
    for (int i = 0; i < ub_list.size(); i++) {
      if (overflow_table[0][i].size() == 1) {
        // upper splitting condition
        GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[i]);
        h.update_const(
          ((overflow_table[0][i][NULL] / stride) % unroll_amount)
          * -stride);
      } else {
        // upper splitting condition
        std::string over_name = overflow_var_name_prefix
          + to_string(overflow_var_name_counter++);
        Free_Var_Decl *over_free_var = new Free_Var_Decl(over_name);
        over_var_list.push_back(over_free_var);
        GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[i]);
        h.update_coef(cond_upper.get_local(over_free_var), -stride);
        
        // insert constraint 0 <= overflow < unroll_amount
        Variable_ID v = overflow_constraint.get_local(over_free_var);
        GEQ_Handle h1 = overflow_constraint_root->add_GEQ();
        h1.update_coef(v, 1);
        GEQ_Handle h2 = overflow_constraint_root->add_GEQ();
        h2.update_coef(v, -1);
        h2.update_const(unroll_amount - 1);
        
        // create overflow assignment
        bound.setup_names(); // hack to fix omega relation variable names issue
        CG_outputRepr *rhs = NULL;
        bool is_split_illegal = false;
        for (std::map<Variable_ID, int>::iterator j =
               overflow_table[0][i].begin();
             j != overflow_table[0][i].end(); j++)
          if ((*j).first != NULL) {
            if ((*j).first->kind() == Input_Var
                && (*j).first->get_position()
                >= cleanup_split_level)
              is_split_illegal = true;
            
            CG_outputRepr *t = ocg->CreateIdent((*j).first->name());
            if ((*j).second != 1)
              t = ocg->CreateTimes(ocg->CreateInt((*j).second),
                                   t);
            rhs = ocg->CreatePlus(rhs, t);
          } else if ((*j).second != 0)
            rhs = ocg->CreatePlus(rhs, ocg->CreateInt((*j).second));
        
        if (is_split_illegal) {
          rhs->clear();
          delete rhs;
          throw loop_error(
            "cannot split cleanup code at loop level "
            + to_string(cleanup_split_level)
            + " due to overflow variable data dependence");
        }
        
        if (stride != 1)
          rhs = ocg->CreateIntegerCeil(rhs, ocg->CreateInt(stride));
        rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
        
        CG_outputRepr *lhs = ocg->CreateIdent(over_name);
        init_code = ocg->StmtListAppend(init_code,
                                        ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
        lhs = ocg->CreateIdent(over_name);
        overflow_code = ocg->StmtListAppend(overflow_code,
                                            ocg->CreateAssignment(0, lhs, rhs));
      }
    }
    
    // lower splitting condition
    GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[0]);
  } else if (is_overflow_simplifiable && ub_list.size() == 1) {
    for (int i = 0; i < lb_list.size(); i++) {
      
      if (overflow_table[i][0].size() == 1) {
        // lower splitting condition
        GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[i]);
        h.update_const(overflow_table[i][0][NULL] * -stride);
      } else {
        // lower splitting condition
        std::string over_name = overflow_var_name_prefix
          + to_string(overflow_var_name_counter++);
        Free_Var_Decl *over_free_var = new Free_Var_Decl(over_name);
        over_var_list.push_back(over_free_var);
        GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[i]);
        h.update_coef(cond_lower.get_local(over_free_var), -stride);
        
        // insert constraint 0 <= overflow < unroll_amount
        Variable_ID v = overflow_constraint.get_local(over_free_var);
        GEQ_Handle h1 = overflow_constraint_root->add_GEQ();
        h1.update_coef(v, 1);
        GEQ_Handle h2 = overflow_constraint_root->add_GEQ();
        h2.update_coef(v, -1);
        h2.update_const(unroll_amount - 1);
        
        // create overflow assignment
        bound.setup_names(); // hack to fix omega relation variable names issue
        CG_outputRepr *rhs = NULL;
        for (std::map<Variable_ID, int>::iterator j =
               overflow_table[0][i].begin();
             j != overflow_table[0][i].end(); j++)
          if ((*j).first != NULL) {
            CG_outputRepr *t = ocg->CreateIdent((*j).first->name());
            if ((*j).second != 1)
              t = ocg->CreateTimes(ocg->CreateInt((*j).second),
                                   t);
            rhs = ocg->CreatePlus(rhs, t);
          } else if ((*j).second != 0)
            rhs = ocg->CreatePlus(rhs, ocg->CreateInt((*j).second));
        
        if (stride != 1)
          rhs = ocg->CreateIntegerCeil(rhs, ocg->CreateInt(stride));
        rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
        
        CG_outputRepr *lhs = ocg->CreateIdent(over_name);
        init_code = ocg->StmtListAppend(init_code,
                                        ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
        lhs = ocg->CreateIdent(over_name);
        overflow_code = ocg->StmtListAppend(overflow_code,
                                            ocg->CreateAssignment(0, lhs, rhs));
      }
    }
    
    // upper splitting condition
    GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[0]);
  } else {
    std::string over_name = overflow_var_name_prefix
      + to_string(overflow_var_name_counter++);
    Free_Var_Decl *over_free_var = new Free_Var_Decl(over_name);
    over_var_list.push_back(over_free_var);
    
    std::vector<CG_outputRepr *> lb_repr_list, ub_repr_list;
    for (int i = 0; i < lb_list.size(); i++) {
      lb_repr_list.push_back(
        output_lower_bound_repr(ocg, 
                                lb_list[i],
                                bound.set_var(dim + 1), result.first, result.second,
                                bound, Relation::True(bound.n_set()),
                                std::vector<std::pair<CG_outputRepr *, int> >(
                                  bound.n_set(),
                                  std::make_pair(
                                    static_cast<CG_outputRepr *>(NULL),
                                    0)),
                                uninterpreted_symbols[stmt_num]));
      GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[i]);
    }
    for (int i = 0; i < ub_list.size(); i++) {
      ub_repr_list.push_back(
        output_upper_bound_repr(ocg, ub_list[i],
                                bound.set_var(dim + 1), bound,
                                std::vector<std::pair<CG_outputRepr *, int> >(
                                  bound.n_set(),
                                  std::make_pair(
                                    static_cast<CG_outputRepr *>(NULL),
                                    0)),
                                uninterpreted_symbols[stmt_num]));
      GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[i]);
      h.update_coef(cond_upper.get_local(over_free_var), -stride);
    }
    
    CG_outputRepr *lbRepr, *ubRepr; 
    if (lb_repr_list.size() > 1) {
      //debug_fprintf(stderr, "loop_unroll.cc createInvoke( max )\n"); 
      lbRepr = ocg->CreateInvoke("max", lb_repr_list);
    }
    else if (lb_repr_list.size() == 1) {
      lbRepr = lb_repr_list[0];
    }
    
    if (ub_repr_list.size() > 1) {
      //debug_fprintf(stderr, "loop_unroll.cc createInvoke( min )\n"); 
      ubRepr = ocg->CreateInvoke("min", ub_repr_list);
    }
    else if (ub_repr_list.size() == 1) {
      ubRepr = ub_repr_list[0];
    }
    
    // create overflow assignment
    CG_outputRepr *rhs = ocg->CreatePlus(ocg->CreateMinus(ubRepr, lbRepr),
                                         ocg->CreateInt(1));
    if (stride != 1)
      rhs = ocg->CreateIntegerFloor(rhs, ocg->CreateInt(stride));
    rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
    CG_outputRepr *lhs = ocg->CreateIdent(over_name);
    init_code = ocg->StmtListAppend(init_code,
                                    ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
    lhs = ocg->CreateIdent(over_name);
    overflow_code = ocg->CreateAssignment(0, lhs, rhs);
    
    // insert constraint 0 <= overflow < unroll_amount
    Variable_ID v = overflow_constraint.get_local(over_free_var);
    GEQ_Handle h1 = overflow_constraint_root->add_GEQ();
    h1.update_coef(v, 1);
    GEQ_Handle h2 = overflow_constraint_root->add_GEQ();
    h2.update_coef(v, -1);
    h2.update_const(unroll_amount - 1);
  }
  
  // insert overflow statement
  int overflow_stmt_num = -1;
  if (overflow_code != NULL) {
    // build iteration space for overflow statement
    Relation mapping(level, cleanup_split_level - 1);
    F_And *f_root = mapping.add_and();
    for (int i = 1; i < cleanup_split_level; i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_coef(mapping.input_var(i), -1);
    }
    Relation overflow_IS = omega::Range(Restrict_Domain(mapping, copy(hull)));
    for (int i = 1; i < cleanup_split_level; i++)
      overflow_IS.name_set_var(i, hull.set_var(i)->name());
    overflow_IS.setup_names();
    
    // build dumb transformation relation for overflow statement
    Relation overflow_xform(cleanup_split_level - 1,
                            2 * (cleanup_split_level - 1) + 1);
    f_root = overflow_xform.add_and();
    for (int i = 1; i <= cleanup_split_level - 1; i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(overflow_xform.output_var(2 * i), 1);
      h.update_coef(overflow_xform.input_var(i), -1);
      
      h = f_root->add_EQ();
      h.update_coef(overflow_xform.output_var(2 * i - 1), 1);
      h.update_const(-lex[2 * i - 2]);
    }
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(
      overflow_xform.output_var(2 * (cleanup_split_level - 1) + 1),
      1);
    h.update_const(-lex[2 * (cleanup_split_level - 1)]);
    
    shiftLexicalOrder(lex, 2 * cleanup_split_level - 2, 1);
    Statement overflow_stmt;
    
    overflow_stmt.code = overflow_code;
    overflow_stmt.IS = overflow_IS;
    overflow_stmt.xform = overflow_xform;
    overflow_stmt.loop_level = std::vector<LoopLevel>(level - 1);
    overflow_stmt.ir_stmt_node = NULL;
    for (int i = 0; i < level - 1; i++) {
      overflow_stmt.loop_level[i].type =
        stmt[stmt_num].loop_level[i].type;
      if (stmt[stmt_num].loop_level[i].type == LoopLevelTile
          && stmt[stmt_num].loop_level[i].payload >= level)
        overflow_stmt.loop_level[i].payload = -1;
      else
        overflow_stmt.loop_level[i].payload =
          stmt[stmt_num].loop_level[i].payload;
      overflow_stmt.loop_level[i].parallel_level =
        stmt[stmt_num].loop_level[i].parallel_level;
    }
    
    debug_fprintf(stderr, "loop_unroll.cc L581 adding stmt %d\n", stmt.size()); 
    stmt.push_back(overflow_stmt);

    uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
    uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
    dep.insert();
    overflow_stmt_num = stmt.size() - 1;
    overflow[overflow_stmt_num] = over_var_list;
    
    // update the global known information on overflow variable
    this->known = Intersection(this->known,
                               Extend_Set(copy(overflow_constraint),
                                          this->known.n_set() - overflow_constraint.n_set()));
    
    // update dependence graph
    DependenceVector dv;
    dv.type = DEP_CONTROL;
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      dep.connect(overflow_stmt_num, *i, dv);
    dv.type = DEP_W2W;
    {
      IR_ScalarSymbol *overflow_sym = NULL;
      std::vector<IR_ScalarRef *> scalars = ir->FindScalarRef(overflow_code);
      for (int i = scalars.size() - 1; i >= 0; i--)
        if (scalars[i]->is_write()) {
          overflow_sym = scalars[i]->symbol();
          break;
        }
      for (int i = scalars.size() - 1; i >= 0; i--)
        delete scalars[i];
      dv.sym = overflow_sym;
    }
    dv.lbounds = std::vector<coef_t>(dep.num_dim(), 0);
    dv.ubounds = std::vector<coef_t>(dep.num_dim(), 0);
    int dep_dim = get_last_dep_dim_before(stmt_num, level);
    for (int i = dep_dim + 1; i < dep.num_dim(); i++) {
      dv.lbounds[i] = -posInfinity;
      dv.ubounds[i] = posInfinity;
    }
    for (int i = 0; i <= dep_dim; i++) {
      if (i != 0) {
        dv.lbounds[i - 1] = 0;
        dv.ubounds[i - 1] = 0;
      }
      dv.lbounds[i] = 1;
      dv.ubounds[i] = posInfinity;
      dep.connect(overflow_stmt_num, overflow_stmt_num, dv);
    }
  }
  
  // split the loop so it can be fully unrolled
  std::set<int> new_stmts = split(stmt_num, cleanup_split_level, cond_upper);
  std::set<int> new_stmts2 = split(stmt_num, cleanup_split_level, cond_lower);
  new_stmts.insert(new_stmts2.begin(), new_stmts2.end());
  
  // check if unrolled statements can be trivially lumped together as one statement
  bool can_be_lumped = true;
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      if (*i != stmt_num) {
        if (stmt[*i].loop_level.size()
            != stmt[stmt_num].loop_level.size()) {
          can_be_lumped = false;
          break;
        }
        for (int j = 0; j < stmt[stmt_num].loop_level.size(); j++)
          if (!(stmt[*i].loop_level[j].type
                == stmt[stmt_num].loop_level[j].type
                && stmt[*i].loop_level[j].payload
                == stmt[stmt_num].loop_level[j].payload)) {
            can_be_lumped = false;
            break;
          }
        if (!can_be_lumped)
          break;
        std::vector<int> lex2 = getLexicalOrder(*i);
        for (int j = 2 * level; j < lex.size() - 1; j += 2)
          if (lex[j] != lex2[j]) {
            can_be_lumped = false;
            break;
          }
        if (!can_be_lumped)
          break;
      }
  }
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      if (is_inner_loop_depend_on_level(stmt[*i].IS, level,
                                        this->known)) {
        can_be_lumped = false;
        break;
      }
  }
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      if (*i != stmt_num) {
        if (!(Must_Be_Subset(copy(stmt[*i].IS), copy(stmt[stmt_num].IS))
              && Must_Be_Subset(copy(stmt[stmt_num].IS),
                                copy(stmt[*i].IS)))) {
          can_be_lumped = false;
          break;
        }
      }
  }
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++) {
      for (DependenceGraph::EdgeList::iterator j =
             dep.vertex[*i].second.begin();
           j != dep.vertex[*i].second.end(); j++)
        if (same_loop.find(j->first) != same_loop.end()) {
          for (int k = 0; k < j->second.size(); k++)
            if (j->second[k].type == DEP_CONTROL
                || j->second[k].type == DEP_UNKNOWN) {
              can_be_lumped = false;
              break;
            }
          if (!can_be_lumped)
            break;
        }
      if (!can_be_lumped)
        break;
    }
  }
  
  // insert unrolled statements
  int old_num_stmt = stmt.size();
  if (!can_be_lumped) {
    std::map<int, std::vector<int> > what_stmt_num;
    
    for (int j = 1; j < unroll_amount; j++) {
      for (std::set<int>::iterator i = same_loop.begin();
           i != same_loop.end(); i++) {
        Statement new_stmt;
        
        std::vector<std::string> loop_vars;
        std::vector<CG_outputRepr *> subs;
        loop_vars.push_back(stmt[*i].IS.set_var(level)->name());
        subs.push_back(
          ocg->CreatePlus(
            ocg->CreateIdent(
              stmt[*i].IS.set_var(level)->name()),
            ocg->CreateInt(j * stride)));
        new_stmt.code = ocg->CreateSubstitutedStmt(0,
                                                   stmt[*i].code->clone(), loop_vars, subs);
        
        new_stmt.IS = adjust_loop_bound(stmt[*i].IS, level, j * stride);
        add_loop_stride(new_stmt.IS, bound, level - 1,
                        unroll_amount * stride);
        
        new_stmt.xform = copy(stmt[*i].xform);
        
        new_stmt.loop_level = stmt[*i].loop_level;
        new_stmt.ir_stmt_node = NULL;

        debug_fprintf(stderr, "loop_unroll.cc L740 adding stmt %d\n", stmt.size()); 
        stmt.push_back(new_stmt);

        uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
        uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
        dep.insert();
        what_stmt_num[*i].push_back(stmt.size() - 1);
      }
    }
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      add_loop_stride(stmt[*i].IS, bound, level - 1,
                      unroll_amount * stride);
    
    // update dependence graph
    if (stmt[stmt_num].loop_level[level - 1].type == LoopLevelOriginal) {
      int dep_dim = stmt[stmt_num].loop_level[level - 1].payload;
      int new_stride = unroll_amount * stride;
      for (int i = 0; i < old_num_stmt; i++) {
        std::vector<std::pair<int, DependenceVector> > D;
        
        for (DependenceGraph::EdgeList::iterator j =
               dep.vertex[i].second.begin();
             j != dep.vertex[i].second.end();) {
          if (same_loop.find(i) != same_loop.end()) {
            if (same_loop.find(j->first) != same_loop.end()) {
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.type == DEP_CONTROL
                    || dv.type == DEP_UNKNOWN) {
                  D.push_back(std::make_pair(j->first, dv));
                  for (int kk = 0; kk < unroll_amount - 1;
                       kk++)
                    if (what_stmt_num[i][kk] != -1
                        && what_stmt_num[j->first][kk]
                        != -1)
                      dep.connect(what_stmt_num[i][kk],
                                  what_stmt_num[j->first][kk],
                                  dv);
                } else {
                  coef_t lb = dv.lbounds[dep_dim];
                  coef_t ub = dv.ubounds[dep_dim];
                  if (ub == lb
                      && int_mod(lb,
                                 static_cast<coef_t>(new_stride))
                      == 0) {
                    D.push_back(
                      std::make_pair(j->first, dv));
                    for (int kk = 0; kk < unroll_amount - 1;
                         kk++)
                      if (what_stmt_num[i][kk] != -1
                          && what_stmt_num[j->first][kk]
                          != -1)
                        dep.connect(
                          what_stmt_num[i][kk],
                          what_stmt_num[j->first][kk],
                          dv);
                  } else if (lb == -posInfinity
                             && ub == posInfinity) {
                    D.push_back(
                      std::make_pair(j->first, dv));
                    for (int kk = 0; kk < unroll_amount;
                         kk++)
                      if (kk == 0)
                        D.push_back(
                          std::make_pair(j->first,
                                         dv));
                      else if (what_stmt_num[j->first][kk
                                                       - 1] != -1)
                        D.push_back(
                          std::make_pair(
                            what_stmt_num[j->first][kk
                                                    - 1],
                            dv));
                    for (int t = 0; t < unroll_amount - 1;
                         t++)
                      if (what_stmt_num[i][t] != -1)
                        for (int kk = 0;
                             kk < unroll_amount;
                             kk++)
                          if (kk == 0)
                            dep.connect(
                              what_stmt_num[i][t],
                              j->first, dv);
                          else if (what_stmt_num[j->first][kk
                                                           - 1] != -1)
                            dep.connect(
                              what_stmt_num[i][t],
                              what_stmt_num[j->first][kk
                                                      - 1],
                              dv);
                  } else {
                    for (int kk = 0; kk < unroll_amount;
                         kk++) {
                      if (lb != -posInfinity) {
                        if (kk * stride
                            < int_mod(lb,
                                      static_cast<coef_t>(new_stride)))
                          dv.lbounds[dep_dim] =
                            floor(
                              static_cast<double>(lb)
                              / new_stride)
                            * new_stride
                            + new_stride;
                        else
                          dv.lbounds[dep_dim] =
                            floor(
                              static_cast<double>(lb)
                              / new_stride)
                            * new_stride;
                      }
                      if (ub != posInfinity) {
                        if (kk * stride
                            > int_mod(ub,
                                      static_cast<coef_t>(new_stride)))
                          dv.ubounds[dep_dim] =
                            floor(
                              static_cast<double>(ub)
                              / new_stride)
                            * new_stride
                            - new_stride;
                        else
                          dv.ubounds[dep_dim] =
                            floor(
                              static_cast<double>(ub)
                              / new_stride)
                            * new_stride;
                      }
                      if (dv.ubounds[dep_dim]
                          >= dv.lbounds[dep_dim]) {
                        if (kk == 0)
                          D.push_back(
                            std::make_pair(
                              j->first,
                              dv));
                        else if (what_stmt_num[j->first][kk
                                                         - 1] != -1)
                          D.push_back(
                            std::make_pair(
                              what_stmt_num[j->first][kk
                                                      - 1],
                              dv));
                      }
                    }
                    for (int t = 0; t < unroll_amount - 1;
                         t++)
                      if (what_stmt_num[i][t] != -1)
                        for (int kk = 0;
                             kk < unroll_amount;
                             kk++) {
                          if (lb != -posInfinity) {
                            if (kk * stride
                                < int_mod(
                                  lb + t
                                  + 1,
                                  static_cast<coef_t>(new_stride)))
                              dv.lbounds[dep_dim] =
                                floor(
                                  static_cast<double>(lb
                                                      + (t
                                                         + 1)
                                                      * stride)
                                  / new_stride)
                                * new_stride
                                + new_stride;
                            else
                              dv.lbounds[dep_dim] =
                                floor(
                                  static_cast<double>(lb
                                                      + (t
                                                         + 1)
                                                      * stride)
                                  / new_stride)
                                * new_stride;
                          }
                          if (ub != posInfinity) {
                            if (kk * stride
                                > int_mod(
                                  ub + t
                                  + 1,
                                  static_cast<coef_t>(new_stride)))
                              dv.ubounds[dep_dim] =
                                floor(
                                  static_cast<double>(ub
                                                      + (t
                                                         + 1)
                                                      * stride)
                                  / new_stride)
                                * new_stride
                                - new_stride;
                            else
                              dv.ubounds[dep_dim] =
                                floor(
                                  static_cast<double>(ub
                                                      + (t
                                                         + 1)
                                                      * stride)
                                  / new_stride)
                                * new_stride;
                          }
                          if (dv.ubounds[dep_dim]
                              >= dv.lbounds[dep_dim]) {
                            if (kk == 0)
                              dep.connect(
                                what_stmt_num[i][t],
                                j->first,
                                dv);
                            else if (what_stmt_num[j->first][kk
                                                             - 1] != -1)
                              dep.connect(
                                what_stmt_num[i][t],
                                what_stmt_num[j->first][kk
                                                        - 1],
                                dv);
                          }
                        }
                  }
                }
              }
              
              dep.vertex[i].second.erase(j++);
            } else {
              for (int kk = 0; kk < unroll_amount - 1; kk++)
                if (what_stmt_num[i][kk] != -1)
                  dep.connect(what_stmt_num[i][kk], j->first,
                              j->second);
              
              j++;
            }
          } else {
            if (same_loop.find(j->first) != same_loop.end())
              for (int k = 0; k < j->second.size(); k++)
                for (int kk = 0; kk < unroll_amount - 1; kk++)
                  if (what_stmt_num[j->first][kk] != -1)
                    D.push_back(
                      std::make_pair(
                        what_stmt_num[j->first][kk],
                        j->second[k]));
            j++;
          }
        }
        
        for (int j = 0; j < D.size(); j++)
          dep.connect(i, D[j].first, D[j].second);
      }
    }
    
    // reset lexical order for the unrolled loop body
    int midx=INT16_MAX,madx=INT16_MIN;

    for (std::map<int, std::vector<int> >::iterator i =
        what_stmt_num.begin(); i != what_stmt_num.end(); i++) {
      int st = get_const(stmt[i->first].xform, dim+1, Output_Var);
      midx = min(st,midx);
      madx = max(st,madx);
    }
    midx = madx - midx + 1;
    for (std::map<int, std::vector<int> >::iterator i =
           what_stmt_num.begin(); i != what_stmt_num.end(); i++) {
      int count = 0;
      int st = get_const(stmt[i->first].xform, dim+1, Output_Var);
      count++;
      for (int j = 0; j < i->second.size(); j++) {
        assign_const(stmt[i->second[j]].xform, dim+1, st + count*midx);
        count++;
      }
    }
  } else {
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      add_loop_stride(stmt[*i].IS, bound, level - 1,
                      unroll_amount * stride);
    
    int max_level = stmt[stmt_num].loop_level.size();
    std::vector<std::pair<int, int> > stmt_order;
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++)
      stmt_order.push_back(
        std::make_pair(
          get_const(stmt[*i].xform, 2 * max_level,
                    Output_Var), *i));
    sort(stmt_order.begin(), stmt_order.end());
    
    Statement new_stmt;
    new_stmt.code = NULL;
    for (int j = 1; j < unroll_amount; j++) { 
      for (int i = 0; i < stmt_order.size(); i++) {
        std::vector<std::string> loop_vars;
        std::vector<CG_outputRepr *> subs;

        //debug_fprintf(stderr, "loop_unroll.cc, will replace '%s with '%s+%d' ??\n",
        //        stmt[stmt_order[i].second].IS.set_var(level)->name().c_str(),
        //        stmt[stmt_order[i].second].IS.set_var(level)->name().c_str(), j * stride); 
        
        loop_vars.push_back(
          stmt[stmt_order[i].second].IS.set_var(level)->name());
        subs.push_back(
          ocg->CreatePlus(ocg->CreateIdent(stmt[stmt_order[i].second].IS.set_var(level)->name()),
                          ocg->CreateInt(j * stride)));  // BUG HERE
        //debug_fprintf(stderr, "loop_unroll.cc subs  now has %d parts\n", subs.size());
        //for (int k=0; k< subs.size(); k++) //debug_fprintf(stderr, "subs[%d] = 0x%x\n", k, subs[k]); 

        //debug_fprintf(stderr, "ij %d %d  ", i, j);
        //debug_fprintf(stderr, "old src was =\n");
        //stmt[stmt_order[i].second].code->dump(); fflush(stdout); //debug_fprintf(stderr, "\n"); 



        CG_outputRepr *code = ocg->CreateSubstitutedStmt(0,
                                                         stmt[stmt_order[i].second].code->clone(), 
                                                         loop_vars,
                                                         subs);

        //debug_fprintf(stderr, "old src is =\n");
        //stmt[stmt_order[i].second].code->dump(); fflush(stdout); //debug_fprintf(stderr, "\n"); 

        //debug_fprintf(stderr, "substituted copy is =\n"); 
        //code->dump(); //debug_fprintf(stderr, "\n\n"); 


        new_stmt.code = ocg->StmtListAppend(new_stmt.code, code);
        //debug_fprintf(stderr, "appended code =\n");
        //new_stmt.code->dump();

      }
    }
    


    //debug_fprintf(stderr, "new_stmt.IS = \n"); 
    new_stmt.IS = copy(stmt[stmt_num].IS);
    new_stmt.xform = copy(stmt[stmt_num].xform);
    assign_const(new_stmt.xform, 2 * max_level,
                 stmt_order[stmt_order.size() - 1].first + 1);
    new_stmt.loop_level = stmt[stmt_num].loop_level;
    new_stmt.ir_stmt_node = NULL;

    new_stmt.has_inspector = false; //  ?? or from copied stmt?
    if (stmt[stmt_num].has_inspector) debug_fprintf(stderr, "OLD STMT HAS INSPECTOR\n");
    else debug_fprintf(stderr, "OLD STMT DOES NOT HAVE INSPECTOR\n");

    debug_fprintf(stderr, "loop_unroll.cc L1083 adding stmt %d\n", stmt.size()); 
    stmt.push_back(new_stmt);

    uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
    uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
    dep.insert();
    
    //debug_fprintf(stderr, "update dependence graph\n"); 
    // update dependence graph
    if (stmt[stmt_num].loop_level[level - 1].type == LoopLevelOriginal) {
      int dep_dim = stmt[stmt_num].loop_level[level - 1].payload;
      int new_stride = unroll_amount * stride;
      for (int i = 0; i < old_num_stmt; i++) {
        std::vector<std::pair<int, std::vector<DependenceVector> > > D;
        
        for (DependenceGraph::EdgeList::iterator j =
               dep.vertex[i].second.begin();
             j != dep.vertex[i].second.end();) {
          if (same_loop.find(i) != same_loop.end()) {
            if (same_loop.find(j->first) != same_loop.end()) {
              std::vector<DependenceVector> dvs11, dvs12, dvs22,
                dvs21;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.type == DEP_CONTROL
                    || dv.type == DEP_UNKNOWN) {
                  if (i == j->first) {
                    dvs11.push_back(dv);
                    dvs22.push_back(dv);
                  } else
                    throw loop_error(
                      "unrolled statements lumped together illegally");
                } else {
                  coef_t lb = dv.lbounds[dep_dim];
                  coef_t ub = dv.ubounds[dep_dim];
                  if (ub == lb
                      && int_mod(lb,
                                 static_cast<coef_t>(new_stride))
                      == 0) {
                    dvs11.push_back(dv);
                    dvs22.push_back(dv);
                  } else {
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = ceil(
                        static_cast<double>(lb)
                        / new_stride)
                        * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = floor(
                        static_cast<double>(ub)
                        / new_stride)
                        * new_stride;
                    if (dv.ubounds[dep_dim]
                        >= dv.lbounds[dep_dim])
                      dvs11.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = ceil(
                        static_cast<double>(lb)
                        / new_stride)
                        * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = ceil(
                        static_cast<double>(ub)
                        / new_stride)
                        * new_stride;
                    if (dv.ubounds[dep_dim]
                        >= dv.lbounds[dep_dim])
                      dvs21.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = floor(
                        static_cast<double>(lb)
                        / new_stride)
                        * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = floor(
                        static_cast<double>(ub
                                            - stride)
                        / new_stride)
                        * new_stride;
                    if (dv.ubounds[dep_dim]
                        >= dv.lbounds[dep_dim])
                      dvs12.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = floor(
                        static_cast<double>(lb)
                        / new_stride)
                        * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = ceil(
                        static_cast<double>(ub
                                            - stride)
                        / new_stride)
                        * new_stride;
                    if (dv.ubounds[dep_dim]
                        >= dv.lbounds[dep_dim])
                      dvs22.push_back(dv);
                  }
                }
              }
              if (dvs11.size() > 0)
                D.push_back(std::make_pair(i, dvs11));
              if (dvs22.size() > 0)
                dep.connect(old_num_stmt, old_num_stmt, dvs22);
              if (dvs12.size() > 0)
                D.push_back(
                  std::make_pair(old_num_stmt, dvs12));
              if (dvs21.size() > 0)
                dep.connect(old_num_stmt, i, dvs21);
              
              dep.vertex[i].second.erase(j++);
            } else {
              dep.connect(old_num_stmt, j->first, j->second);
              j++;
            }
          } else {
            if (same_loop.find(j->first) != same_loop.end())
              D.push_back(
                std::make_pair(old_num_stmt, j->second));
            j++;
          }
        }
        
        for (int j = 0; j < D.size(); j++)
          dep.connect(i, D[j].first, D[j].second);
      }
    }
  }
  
  //debug_fprintf(stderr, "                                                  loop_unroll.cc returning new_stmts\n");
  return new_stmts;
}


