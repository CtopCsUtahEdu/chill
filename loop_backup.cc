/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Core loop transformation functionality.

 Notes:
   "level" (starting from 1) means loop level and it corresponds to "dim"
 (starting from 0) in transformed iteration space [c_1,l_1,c_2,l_2,....,
 c_n,l_n,c_(n+1)], e.g., l_2 is loop level 2 in generated code, dim 3
 in transformed iteration space, and variable 4 in Omega relation.
 All c's are constant numbers only and they will not show up as actual loops.
 Formula:
    dim = 2*level - 1
    var = dim + 1

 History:
   10/2005 Created by Chun Chen.
   09/2009 Expand tile functionality, -chun
   10/2009 Initialize unfusible loop nest without bailing out, -chun
*****************************************************************************/

#include <limits.h>
#include <math.h>
#include <code_gen/code_gen.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/output_repr.h>
#include <iostream>
#include <map>
#include "loop.hh"
#include "omegatools.hh"
#include "irtools.hh"
#include "chill_error.hh"

using namespace omega;

const std::string Loop::tmp_loop_var_name_prefix = std::string("_t");
const std::string Loop::overflow_var_name_prefix = std::string("over");

//-----------------------------------------------------------------------------
// Class Loop
//-----------------------------------------------------------------------------

bool Loop::init_loop(std::vector<ir_tree_node *> &ir_tree, std::vector<ir_tree_node *> &ir_stmt) {
  ir_stmt = extract_ir_stmts(ir_tree);
  std::vector<int> stmt_nesting_level(ir_stmt.size());
  for (int i = 0; i < ir_stmt.size(); i++) {
    ir_stmt[i]->payload = i;
    int t = 0;
    ir_tree_node *itn = ir_stmt[i];
    while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP)
        t++;
    }
    stmt_nesting_level[i] = t;
  }
  
  stmt = std::vector<Statement>(ir_stmt.size());
  int n_dim = -1;
  int max_loc;
  std::vector<std::string> index;
  for (int i = 0; i < ir_stmt.size(); i++) {
    int max_nesting_level = -1;
    int loc;
    for (int j = 0; j < ir_stmt.size(); j++)
      if (stmt_nesting_level[j] > max_nesting_level) {
        max_nesting_level = stmt_nesting_level[j];
        loc = j;
      }
    
    // most deeply nested statement acting as a reference point
    if (n_dim == -1) {
      n_dim = max_nesting_level;
      max_loc = loc;
      
      index = std::vector<std::string>(n_dim);
      
      ir_tree_node *itn = ir_stmt[loc];
      int cur_dim = n_dim-1;
      while (itn->parent != NULL) {
        itn = itn->parent;
        if (itn->content->type() == IR_CONTROL_LOOP) {
          index[cur_dim] = static_cast<IR_Loop *>(itn->content)->index()->name();
          itn->payload = cur_dim--;
        }
      }
    }
    
    // align loops by names, temporary solution
    ir_tree_node *itn = ir_stmt[loc];
    while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP && itn->payload == -1) {
        std::string name = static_cast<IR_Loop *>(itn->content)->index()->name();
        for (int j = 0; j < n_dim; j++)
          if (index[j] == name) {
            itn->payload = j;
            break;
          }
        if (itn->payload == -1)
          throw loop_error("no complex alignment yet");
      }
    }
    
    // set relation variable names
    Relation r(n_dim);
    F_And *f_root = r.add_and();
    itn = ir_stmt[loc];
    while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP)
        r.name_set_var(itn->payload+1, static_cast<IR_Loop *>(itn->content)->index()->name());
    }
    
    // extract information from loop/if structures
    std::vector<bool> processed(n_dim, false);
    Tuple<std::string> vars_to_be_reversed;
    itn = ir_stmt[loc];
    while (itn->parent != NULL) {
      itn = itn->parent;
      
      switch (itn->content->type()) {
      case IR_CONTROL_LOOP: {
        IR_Loop *lp = static_cast<IR_Loop *>(itn->content);
        Variable_ID v = r.set_var(itn->payload+1);
        int c;
        
        try {
          c = lp->step_size();
          if (c > 0) {
            CG_outputRepr *lb = lp->lower_bound();
            exp2formula(ir, r, f_root, freevar, lb, v, 's', IR_COND_GE, true);
            CG_outputRepr *ub = lp->upper_bound();
            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_LT || cond == IR_COND_LE)
              exp2formula(ir, r, f_root, freevar, ub, v, 's', cond, true);
            else
              throw ir_error("loop condition not supported");
            
          }
          else if (c < 0) {
            CG_outputBuilder *ocg = ir->builder();
            CG_outputRepr *lb = lp->lower_bound();
            lb = ocg->CreateMinus(NULL, lb);
            exp2formula(ir, r, f_root, freevar, lb, v, 's', IR_COND_GE, true);
            CG_outputRepr *ub = lp->upper_bound();
            ub = ocg->CreateMinus(NULL, ub);
            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_GE)
              exp2formula(ir, r, f_root, freevar, ub, v, 's', IR_COND_LE, true);
            else if (cond == IR_COND_GT)
              exp2formula(ir, r, f_root, freevar, ub, v, 's', IR_COND_LT, true);
            else
              throw ir_error("loop condition not supported");
            
            vars_to_be_reversed.append(lp->index()->name());
          }
          else
            throw ir_error("loop step size zero");
        }
        catch (const ir_error &e) {
          for (int i = 0; i < itn->children.size(); i++)
            delete itn->children[i];
          itn->children = std::vector<ir_tree_node *>();
          itn->content = itn->content->convert();
          return false;
        }
        
        if (abs(c) != 1) {
          F_Exists *f_exists = f_root->add_exists();
          Variable_ID e = f_exists->declare();
          F_And *f_and = f_exists->add_and();
          Stride_Handle h = f_and->add_stride(abs(c));
          if (c > 0)
            h.update_coef(e, 1);
          else
            h.update_coef(e, -1);
          h.update_coef(v, -1);
          CG_outputRepr *lb = lp->lower_bound();
          exp2formula(ir, r, f_and, freevar, lb, e, 's', IR_COND_EQ, true);
        }
        
        processed[itn->payload] = true;
        break;
      }
      case IR_CONTROL_IF: {
        CG_outputRepr *cond = static_cast<IR_If *>(itn->content)->condition();
        try {
          if (itn->payload % 2 == 1)
            exp2constraint(ir, r, f_root, freevar, cond, true);
          else {
            F_Not *f_not = f_root->add_not();
            F_And *f_and = f_not->add_and();
            exp2constraint(ir, r, f_and, freevar, cond, true);
          }
        }
        catch (const ir_error &e) {
          std::vector<ir_tree_node *> *t;
          if (itn->parent == NULL)
            t = &ir_tree;
          else
            t = &(itn->parent->children);
          int id = itn->payload;
          int i = t->size() - 1;
          while (i >= 0) {
            if ((*t)[i] == itn) {
              for (int j = 0; j < itn->children.size(); j++)
                delete itn->children[j];
              itn->children = std::vector<ir_tree_node *>();
              itn->content = itn->content->convert();
            }
            else if ((*t)[i]->payload >> 1 == id >> 1) {
              delete (*t)[i];
              t->erase(t->begin()+i);
            }
            i--;
          }
          return false;
        }
        
        break;
      }
      default:
        for (int i = 0; i < itn->children.size(); i++)
          delete itn->children[i];
        itn->children = std::vector<ir_tree_node *>();
        itn->content = itn->content->convert();
        return false;
      }
    }
    
    // add information for missing loops
    for (int j = 0; j < n_dim; j++)
      if (!processed[j]) {
        ir_tree_node *itn = ir_stmt[max_loc];
        while (itn->parent != NULL) {
          itn = itn->parent;
          if (itn->content->type() == IR_CONTROL_LOOP && itn->payload == j)
            break;
        }
        
        Variable_ID v = r.set_var(j+1);
        if (loc < max_loc) {
          CG_outputRepr *lb = static_cast<IR_Loop *>(itn->content)->lower_bound();
          exp2formula(ir, r, f_root, freevar, lb, v, 's', IR_COND_EQ, true);
        }
        else { // loc > max_loc
          CG_outputRepr *ub = static_cast<IR_Loop *>(itn->content)->upper_bound();
          exp2formula(ir, r, f_root, freevar, ub, v, 's', IR_COND_EQ, true);
        }
      }
    
    r.setup_names();
    r.simplify();
    
    // insert the statement
    CG_outputBuilder *ocg = ir->builder();
    Tuple<CG_outputRepr *> reverse_expr;
    for (int j = 1; j <= vars_to_be_reversed.size(); j++) {
      CG_outputRepr *repl = ocg->CreateIdent(vars_to_be_reversed[j]);
      repl = ocg->CreateMinus(NULL, repl);
      reverse_expr.append(repl);
    }     
    CG_outputRepr *code = static_cast<IR_Block *>(ir_stmt[loc]->content)->extract();
    code = ocg->CreatePlaceHolder(0, code, reverse_expr, vars_to_be_reversed);
    stmt[loc].code = code;
    stmt[loc].IS = r;
    stmt[loc].loop_level = std::vector<LoopLevel>(n_dim);
    for (int i = 0; i < n_dim; i++) {
      stmt[loc].loop_level[i].type = LoopLevelOriginal;
      stmt[loc].loop_level[i].payload = i;
      stmt[loc].loop_level[i].parallel_level = 0;
    }
    
    stmt_nesting_level[loc] = -1;
  }
  
  return true;
}  



Loop::Loop(const IR_Control *control) {
  ir = const_cast<IR_Code *>(control->ir_);
  init_code = NULL;
  cleanup_code = NULL;
  tmp_loop_var_name_counter = 1;
  overflow_var_name_counter = 1;
  known = Relation::True(0);
  
  std::vector<ir_tree_node *> ir_tree = build_ir_tree(control->clone(), NULL);
  std::vector<ir_tree_node *> ir_stmt;
  
  while (!init_loop(ir_tree, ir_stmt)) {}
  
  // init the dependence graph
  for (int i = 0; i < stmt.size(); i++)
    dep.insert();
  
  for (int i = 0; i < stmt.size(); i++)
    for (int j = i; j < stmt.size(); j++) {
      std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > dv = test_data_dependences(ir, stmt[i].code, stmt[i].IS, stmt[j].code, stmt[j].IS, freevar);
      
      for (int k = 0; k < dv.first.size(); k++)
        if (is_dependence_valid(ir_stmt[i], ir_stmt[j], dv.first[k], true))
          dep.connect(i, j, dv.first[k]);
        else
          dep.connect(j, i, dv.first[k].reverse());
      
      for (int k = 0; k < dv.second.size(); k++)
        if (is_dependence_valid(ir_stmt[j], ir_stmt[i], dv.second[k], false))
          dep.connect(j, i, dv.second[k]);
        else
          dep.connect(i, j, dv.second[k].reverse());
    }
  
  // cleanup the IR tree
  for (int i = 0; i < ir_tree.size(); i++)
    delete ir_tree[i];
  
  // init dumb transformation relations e.g. [i, j] -> [ 0, i, 0, j, 0]
  for (int i = 0; i < stmt.size(); i++) {
    int n = stmt[i].IS.n_set();
    stmt[i].xform = Relation(n, 2*n+1);
    F_And *f_root = stmt[i].xform.add_and();
    
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(stmt[i].xform.output_var(2*j), 1);
      h.update_coef(stmt[i].xform.input_var(j), -1);
    }
    
    for (int j = 1; j <= 2*n+1; j+=2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(stmt[i].xform.output_var(j), 1);
    }
    stmt[i].xform.simplify();
  }
  
  if (stmt.size() != 0)
    num_dep_dim = stmt[0].IS.n_set();
  else
    num_dep_dim = 0;
}


Loop::~Loop() {
  for (int i = 0; i < stmt.size(); i++)
    if (stmt[i].code != NULL) {
      stmt[i].code->clear();
      delete stmt[i].code;
    }
  if (init_code != NULL) {
    init_code->clear();
    delete init_code;
  }
  if (cleanup_code != NULL) {
    cleanup_code->clear();
    delete cleanup_code;
  }
}


int Loop::get_dep_dim_of(int stmt_num, int level) const {
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invaid statement " + to_string(stmt_num));
  
  if (level < 1 || level > stmt[stmt_num].loop_level.size())
    return -1;
  
  int trip_count = 0;
  while (true) {
    switch (stmt[stmt_num].loop_level[level-1].type) {
    case LoopLevelOriginal:
      return stmt[stmt_num].loop_level[level-1].payload;
    case LoopLevelTile:
      level = stmt[stmt_num].loop_level[level-1].payload;
      if (level < 1)
        return -1;
      if (level > stmt[stmt_num].loop_level.size())
        throw loop_error("incorrect loop level information for statement " + to_string(stmt_num));
      break;
    default:
      throw loop_error("unknown loop level information for statement " + to_string(stmt_num));
    }
    trip_count++;
    if (trip_count >= stmt[stmt_num].loop_level.size())
      throw loop_error("incorrect loop level information for statement " + to_string(stmt_num));
  }
}


int Loop::get_last_dep_dim_before(int stmt_num, int level) const {
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invaid statement " + to_string(stmt_num));
  
  if (level < 1)
    return -1;
  if (level > stmt[stmt_num].loop_level.size())
    level = stmt[stmt_num].loop_level.size() + 1;
  
  for (int i = level-1; i >= 1; i--)
    if (stmt[stmt_num].loop_level[i-1].type == LoopLevelOriginal)
      return stmt[stmt_num].loop_level[i-1].payload;
  
  return -1;
}


void Loop::print_internal_loop_structure() const {
  for (int i = 0; i < stmt.size(); i++) {
    std::vector<int> lex = getLexicalOrder(i);
    std::cout << "s" << i+1 << ": ";
    for (int j = 0; j < stmt[i].loop_level.size(); j++) {
      if (2*j < lex.size())
        std::cout << lex[2*j];
      switch (stmt[i].loop_level[j].type) {
      case LoopLevelOriginal:
        std::cout << "(dim:" << stmt[i].loop_level[j].payload << ")";
        break;
      case LoopLevelTile:
        std::cout << "(tile:" << stmt[i].loop_level[j].payload << ")";
        break;
      default:
        std::cout << "(unknown)";
      }
      std::cout << ' ';
    }
    for (int j = 2*stmt[i].loop_level.size(); j < lex.size(); j+=2) {
      std::cout << lex[j];
      if (j != lex.size()-1)
        std::cout << ' ';
    }
    std::cout << std::endl;
  }
}


CG_outputRepr *Loop::getCode(int effort) const {  
  const int m = stmt.size();
  if (m == 0)
    return NULL;
  const int n = stmt[0].xform.n_out();
  
  Tuple<CG_outputRepr *> ni(m);
  Tuple<Relation> IS(m);
  Tuple<Relation> xform(m);
  for (int i = 0; i < m; i++) {
    ni[i+1] = stmt[i].code;
    IS[i+1] = stmt[i].IS;
    xform[i+1] = stmt[i].xform;
  }
  
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());  
  CG_outputBuilder *ocg = ir->builder();
  CG_outputRepr *repr = MMGenerateCode(ocg, xform, IS, ni, known, effort);
  
  if (init_code != NULL)
    repr = ocg->StmtListAppend(init_code->clone(), repr);
  if (cleanup_code != NULL)
    repr = ocg->StmtListAppend(repr, cleanup_code->clone());
  
  return repr;
}


void Loop::printCode(int effort) const {
  const int m = stmt.size();
  if (m == 0)
    return;
  const int n = stmt[0].xform.n_out();
  
  Tuple<Relation> IS(m);
  Tuple<Relation> xform(m);
  for (int i = 0; i < m; i++) {
    IS[i+1] = stmt[i].IS;
    xform[i+1] = stmt[i].xform;
  }
  
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  std::cout << MMGenerateCode(xform, IS, known, effort);
}


Relation Loop::getNewIS(int stmt_num) const {
  Relation result;
  
  if (stmt[stmt_num].xform.is_null()) {
    Relation known = Extend_Set(copy(this->known), stmt[stmt_num].IS.n_set() - this->known.n_set());
    result = Intersection(copy(stmt[stmt_num].IS), known);
  }
  else {
    Relation known = Extend_Set(copy(this->known), stmt[stmt_num].xform.n_out() - this->known.n_set()); 
    result = Intersection(Range(Restrict_Domain(copy(stmt[stmt_num].xform), copy(stmt[stmt_num].IS))), known);
  }
  
  result.simplify(2, 4);
  
  return result;
}

std::vector<Relation> Loop::getNewIS() const {
  const int m = stmt.size();
  
  std::vector<Relation> new_IS(m);
  for (int i = 0; i < m; i++)
    new_IS[i] = getNewIS(i);
  
  return new_IS;
}


void Loop::permute(const std::vector<int> &pi) {
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  
  permute(active, pi);
}


void Loop::original() {
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  setLexicalOrder(0, active);
}


void Loop::permute(const std::set<int> &active, const std::vector<int> &pi) {
  if (active.size() == 0 || pi.size() == 0)
    return;
  
  // check for sanity of parameters
  int level = pi[0];
  for (int i = 1; i < pi.size(); i++)
    if (pi[i] < level)
      level = pi[i];
  if (level < 1)
    throw std::invalid_argument("invalid permuation");
  std::vector<int> reverse_pi(pi.size(), 0);
  for (int i = 0; i < pi.size(); i++)
    if (pi[i] >= level+pi.size())
      throw std::invalid_argument("invalid permutation");
    else
      reverse_pi[pi[i]-level] = i+level;
  for (int i = 0; i < reverse_pi.size(); i++)
    if (reverse_pi[i] == 0)
      throw std::invalid_argument("invalid permuation");
  int ref_stmt_num;
  std::vector<int> lex;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument("invalid statement " + to_string(*i));
    if (i == active.begin()) {
      ref_stmt_num = *i;
      lex = getLexicalOrder(*i);
    }
    else {
      if (level+pi.size()-1 > stmt[*i].loop_level.size())
        throw std::invalid_argument("invalid permuation");
      std::vector<int> lex2 = getLexicalOrder(*i);
      for (int j = 0; j < 2*level-3; j+=2)
        if (lex[j] != lex2[j])
          throw std::invalid_argument("statements to permute must be in the same subloop");
      for (int j = 0; j < pi.size(); j++)
        if (!(stmt[*i].loop_level[level+j-1].type == stmt[ref_stmt_num].loop_level[level+j-1].type &&
              stmt[*i].loop_level[level+j-1].payload == stmt[ref_stmt_num].loop_level[level+j-1].payload))
          throw std::invalid_argument("permuted loops must have the same loop level types");
    }
  }
  
  // Update transformation relations
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].xform.n_out();
    Relation mapping(n, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j+= 2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = 0; j < pi.size(); j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2*(level+j)), 1);
      h.update_coef(mapping.input_var(2*pi[j]), -1);
    }
    for (int j = 1; j < level; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2*j), 1);
      h.update_coef(mapping.input_var(2*j), -1);
    }
    for (int j = level+pi.size(); j <= stmt[*i].loop_level.size(); j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2*j), 1);
      h.update_coef(mapping.input_var(2*j), -1);
    }
    
    stmt[*i].xform = Composition(mapping, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // get the permuation for dependence vectors
  std::vector<int> t;
  for (int i = 0; i < pi.size(); i++)
    if (stmt[ref_stmt_num].loop_level[pi[i]-1].type == LoopLevelOriginal)
      t.push_back(stmt[ref_stmt_num].loop_level[pi[i]-1].payload);
  int max_dep_dim = -1;
  int min_dep_dim = num_dep_dim;
  for (int i = 0; i < t.size(); i++) {
    if (t[i] > max_dep_dim)
      max_dep_dim = t[i];
    if (t[i] < min_dep_dim)
      min_dep_dim = t[i];
  }
  if (min_dep_dim > max_dep_dim)
    return;
  if (max_dep_dim - min_dep_dim + 1 != t.size())
    throw loop_error("cannot update the dependence graph after permuation");
  std::vector<int> dep_pi(num_dep_dim);
  for (int i = 0; i < min_dep_dim; i++)
    dep_pi[i] = i;
  for (int i = min_dep_dim; i <= max_dep_dim; i++)
    dep_pi[i] = t[i-min_dep_dim];
  for (int i = max_dep_dim+1; i < num_dep_dim; i++)
    dep_pi[i] = i;  
  
  // update the dependence graph
  DependenceGraph g;
  for (int i = 0; i < dep.vertex.size(); i++)
    g.insert();
  for (int i = 0; i < dep.vertex.size(); i++)
    for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++) {
      if ((active.find(i) != active.end() && active.find(j->first) != active.end())) {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++) {
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            std::vector<coef_t> lbounds(num_dep_dim);
            std::vector<coef_t> ubounds(num_dep_dim);
            for (int d = 0; d < num_dep_dim; d++) {
              lbounds[d] = dv[k].lbounds[dep_pi[d]];
              ubounds[d] = dv[k].ubounds[dep_pi[d]];
            }
            dv[k].lbounds = lbounds;
            dv[k].ubounds = ubounds;
            break;
          }
          case DEP_CONTROL: {
            break;
          }
          default:
            throw loop_error("unknown dependence type");
          }
        }
        g.connect(i, j->first, dv);
      }
      else if (active.find(i) == active.end() && active.find(j->first) == active.end()) {
        std::vector<DependenceVector> dv = j->second;
        g.connect(i, j->first, dv);
      }
      else {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++)
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            for (int d = 0; d < num_dep_dim; d++)
              if (dep_pi[d] != d) {
                dv[k].lbounds[d] = -posInfinity;
                dv[k].ubounds[d] = posInfinity;
              }
            break;
          }
          case DEP_CONTROL:
            break;
          default:
            throw loop_error("unknown dependence type");
          }
        g.connect(i, j->first, dv);
      }
    }
  dep = g;
  
  // update loop level information
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int cur_dep_dim = min_dep_dim;
    std::vector<LoopLevel> new_loop_level(stmt[*i].loop_level.size());
    for (int j = 1; j <= stmt[*i].loop_level.size(); j++)
      if (j >= level && j < level+pi.size()) {
        switch (stmt[*i].loop_level[reverse_pi[j-level]-1].type) {
        case LoopLevelOriginal:
          new_loop_level[j-1].type = LoopLevelOriginal;
          new_loop_level[j-1].payload = cur_dep_dim++;
          new_loop_level[j-1].parallel_level = stmt[*i].loop_level[reverse_pi[j-level]-1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j-1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[reverse_pi[j-level]-1].payload;
          if (ref_level >= level && ref_level < level+pi.size())
            new_loop_level[j-1].payload = reverse_pi[ref_level-level];
          else
            new_loop_level[j-1].payload = ref_level;
          new_loop_level[j-1].parallel_level = stmt[*i].loop_level[reverse_pi[j-level]-1].parallel_level;
          break;
        }
        default:
          throw loop_error("unknown loop level information for statement " + to_string(*i));
        }
      }
      else {
        switch (stmt[*i].loop_level[j-1].type) {
        case LoopLevelOriginal:
          new_loop_level[j-1].type = LoopLevelOriginal;
          new_loop_level[j-1].payload = stmt[*i].loop_level[j-1].payload;
          new_loop_level[j-1].parallel_level = stmt[*i].loop_level[j-1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j-1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[j-1].payload;
          if (ref_level >= level && ref_level < level+pi.size())
            new_loop_level[j-1].payload = reverse_pi[ref_level-level];
          else
            new_loop_level[j-1].payload = ref_level;
          new_loop_level[j-1].parallel_level = stmt[*i].loop_level[j-1].parallel_level;
          break;
        }
        default:
          throw loop_error("unknown loop level information for statement " + to_string(*i));          
        }
      }
    stmt[*i].loop_level = new_loop_level;
  }
  
  setLexicalOrder(2*level-2, active);
}

std::set<int> Loop::split(int stmt_num, int level, const Relation &cond) {
  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  std::set<int> result;
  int dim = 2*level-1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  Relation cond2 = copy(cond);
  cond2.simplify();
  cond2 = EQs_to_GEQs(cond2);
  Conjunct *c = cond2.single_conjunct();
  int cur_lex = lex[dim-1];
  for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
    int max_level = (*gi).max_tuple_pos();
    Relation single_cond(max_level);
    single_cond.and_with_GEQ(*gi);
    
    // TODO: should decide where to place newly created statements with
    // complementary split condition from dependence graph.
    bool place_after;
    if (max_level == 0)
      place_after = true;
    else if ((*gi).get_coef(cond2.set_var(max_level)) < 0)
      place_after = true;
    else
      place_after = false;
    
    // make adjacent lexical number available for new statements
    if (place_after) {
      lex[dim-1] = cur_lex+1;
      shiftLexicalOrder(lex, dim-1, 1);
    }
    else {
      lex[dim-1] = cur_lex-1;
      shiftLexicalOrder(lex, dim-1, -1);
    }
    
    // original statements with split condition,
    // new statements with complement of split condition
    int old_num_stmt = stmt.size();
    std::map<int, int> what_stmt_num;
    apply_xform(same_loop);
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
      int n = stmt[*i].IS.n_set();
      Relation part1, part2;
      if (max_level > n) {
        part1 = copy(stmt[*i].IS);
        part2 = Relation::False(0);
      }
      else {
        part1 = Intersection(copy(stmt[*i].IS), Extend_Set(copy(single_cond), n-max_level));
        part2 = Intersection(copy(stmt[*i].IS), Extend_Set(Complement(copy(single_cond)), n-max_level));
      }
      
      stmt[*i].IS = part1;
      
      if (Intersection(copy(part2), Extend_Set(copy(this->known), n-this->known.n_set())).is_upper_bound_satisfiable()) {
        Statement new_stmt;
        new_stmt.code = stmt[*i].code->clone();
        new_stmt.IS = part2;
        new_stmt.xform = copy(stmt[*i].xform);
        if (place_after)
          assign_const(new_stmt.xform, dim-1, cur_lex+1);
        else
          assign_const(new_stmt.xform, dim-1, cur_lex-1);
        new_stmt.loop_level = stmt[*i].loop_level;
        stmt.push_back(new_stmt);
        dep.insert();
        what_stmt_num[*i] = stmt.size() - 1;
        if (*i == stmt_num)
          result.insert(stmt.size() - 1);
      }
    }
    
    // update dependence graph
    int dep_dim = get_dep_dim_of(stmt_num, level);
    for (int i = 0; i < old_num_stmt; i++) {
      std::vector<std::pair<int, std::vector<DependenceVector> > > D;
      
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++) {
        if (same_loop.find(i) != same_loop.end()) {
          if (same_loop.find(j->first) != same_loop.end()) {
            if (what_stmt_num.find(i) != what_stmt_num.end() && what_stmt_num.find(j->first) != what_stmt_num.end())
              dep.connect(what_stmt_num[i], what_stmt_num[j->first], j->second);
            if (place_after && what_stmt_num.find(j->first) != what_stmt_num.end()) {
              std::vector<DependenceVector> dvs;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.is_data_dependence() && dep_dim != -1) {
                  dv.lbounds[dep_dim] = -posInfinity;
                  dv.ubounds[dep_dim] = posInfinity;
                }
                dvs.push_back(dv);
              }
              if (dvs.size() > 0)
                D.push_back(std::make_pair(what_stmt_num[j->first], dvs));
            }
            else if (!place_after && what_stmt_num.find(i) != what_stmt_num.end()) {
              std::vector<DependenceVector> dvs;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.is_data_dependence() && dep_dim != -1) {
                  dv.lbounds[dep_dim] = -posInfinity;
                  dv.ubounds[dep_dim] = posInfinity;
                }
                dvs.push_back(dv);
              }
              if (dvs.size() > 0)
                dep.connect(what_stmt_num[i], j->first, dvs);
              
            }
          }
          else {
            if (what_stmt_num.find(i) != what_stmt_num.end())
              dep.connect(what_stmt_num[i], j->first, j->second);
          }
        }
        else if (same_loop.find(j->first) != same_loop.end()) {
          if (what_stmt_num.find(j->first) != what_stmt_num.end())
            D.push_back(std::make_pair(what_stmt_num[j->first], j->second));
        }
      }
      
      for (int j = 0; j < D.size(); j++)
        dep.connect(i, D[j].first, D[j].second);
    }
  }
  
  return result;
}



void Loop::tile(int stmt_num, int level, int tile_size, int outer_level, TilingMethodType method, int alignment_offset, int alignment_multiple) {
  // check for sanity of parameters
  if (tile_size < 0)
    throw std::invalid_argument("invalid tile size");
  if (alignment_multiple < 1 || alignment_offset < 0)
    throw std::invalid_argument("invalid alignment for tile");
  if (stmt_num < 0  || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  if (level <= 0)
    throw std::invalid_argument("invalid loop level " + to_string(level));
  if (level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("there is no loop level " + to_string(level) + " for statement " + to_string(stmt_num));
  if (outer_level <= 0 || outer_level > level) 
    throw std::invalid_argument("invalid tile controlling loop level " + to_string(outer_level));
  
  int dim = 2*level-1;
  int outer_dim = 2*outer_level-1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_tiled_loop = getStatements(lex, dim-1);
  std::set<int> same_tile_controlling_loop = getStatements(lex, outer_dim-1);
  
  // special case for no tiling
  if (tile_size == 0) {
    for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++) {
      Relation r(stmt[*i].xform.n_out(),stmt[*i].xform.n_out()+2);
      F_And *f_root = r.add_and();
      for (int j = 1; j <= 2*outer_level-1; j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.input_var(j), 1);
        h.update_coef(r.output_var(j), -1);
      }
      EQ_Handle h1 = f_root->add_EQ();
      h1.update_coef(r.output_var(2*outer_level), 1);
      EQ_Handle h2 = f_root->add_EQ();
      h2.update_coef(r.output_var(2*outer_level+1), 1);
      for (int j = 2*outer_level; j <= stmt[*i].xform.n_out(); j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.input_var(j), 1);
        h.update_coef(r.output_var(j+2), -1);
      }
      
      stmt[*i].xform = Composition(copy(r), stmt[*i].xform);
    }
  }
  // normal tiling
  else {
    std::set<int> private_stmt;
    for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++) {
//     if (same_tiled_loop.find(*i) == same_tiled_loop.end() && !is_single_iteration(getNewIS(*i), dim))
//       same_tiled_loop.insert(*i);
      
      // should test dim's value directly but it is ok for now
//    if (same_tiled_loop.find(*i) == same_tiled_loop.end() && get_const(stmt[*i].xform, dim+1, Output_Var) == posInfinity)
      if (same_tiled_loop.find(*i) == same_tiled_loop.end() && overflow.find(*i) != overflow.end())
        private_stmt.insert(*i);
    }
    
    
    // extract the union of the iteration space to be considered
    Relation hull;
    {
      Tuple<Relation> r_list;
      Tuple<int> r_mask;
      
      for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++)
        if (private_stmt.find(*i) == private_stmt.end()) {
          Relation r = project_onto_levels(getNewIS(*i), dim+1, true);
          for (int j = outer_dim; j < dim; j++)
            r = Project(r, j+1, Set_Var);
          for (int j = 0; j < outer_dim; j += 2)
            r = Project(r, j+1, Set_Var);
          r_list.append(r);
          r_mask.append(1);
        }
      
      hull = Hull(r_list, r_mask, 1, true);
    }
    
    // extract the bound of the dimension to be tiled
    Relation bound = get_loop_bound(hull, dim);
    if (!bound.has_single_conjunct()) {
      // further simplify the bound
      hull = Approximate(hull);
      bound = get_loop_bound(hull, dim);
      
      int i = outer_dim - 2;
      while (!bound.has_single_conjunct() && i >= 0) {
        hull = Project(hull, i+1, Set_Var);
        bound = get_loop_bound(hull, dim);
        i -= 2;
      }
      
      if (!bound.has_single_conjunct())
        throw loop_error("cannot handle tile bounds");
    }
    
    // separate lower and upper bounds
    std::vector<GEQ_Handle> lb_list, ub_list;
    {
      Conjunct *c = bound.query_DNF()->single_conjunct();
      for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
        int coef = (*gi).get_coef(bound.set_var(dim+1));
        if (coef < 0)
          ub_list.push_back(*gi);
        else if (coef > 0)
          lb_list.push_back(*gi);
      }
    }
    if (lb_list.size() == 0)
      throw loop_error("unable to calculate tile controlling loop lower bound");
    if (ub_list.size() == 0)
      throw loop_error("unable to calculate tile controlling loop upper bound");
    
    // find the simplest lower bound for StridedTile or simplest iteration count for CountedTile
    int simplest_lb = 0, simplest_ub = 0;
    if (method == StridedTile) {
      int best_cost = INT_MAX;
      for (int i = 0; i < lb_list.size(); i++) {
        int cost = 0;
        for (Constr_Vars_Iter ci(lb_list[i]); ci; ci++) {
          switch ((*ci).var->kind()) {
          case Input_Var: {
            cost += 5;
            break;
          }
          case Global_Var: {
            cost += 2;
            break;
          }
          default:
            cost += 15;
            break;
          }
        }
        
        if (cost < best_cost) {
          best_cost = cost;
          simplest_lb = i;
        }
      }
    }
    else if (method == CountedTile) {
      std::map<Variable_ID, coef_t> s1, s2, s3;
      int best_cost = INT_MAX;
      for (int i = 0; i < lb_list.size(); i++) 
        for (int j = 0; j < ub_list.size(); j++) {
          int cost = 0;
          
          for (Constr_Vars_Iter ci(lb_list[i]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              s1[(*ci).var] += (*ci).coef;
              break;
            }
            case Global_Var: {
              s2[(*ci).var] += (*ci).coef;
              break;
            }
            case Exists_Var:
            case Wildcard_Var: {
              s3[(*ci).var] += (*ci).coef;
              break;
            }
            default:
              cost = INT_MAX-2;
              break;
            }
          }
          
          for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              s1[(*ci).var] += (*ci).coef;
              break;
            }
            case Global_Var: {
              s2[(*ci).var] += (*ci).coef;
              break;
            }
            case Exists_Var:
            case Wildcard_Var: {
              s3[(*ci).var] += (*ci).coef;
              break;
            }
            default:
              if (cost == INT_MAX-2)
                cost = INT_MAX-1;
              else
                cost = INT_MAX-3;
              break;
            }
          }
          
          if (cost == 0) {
            for (std::map<Variable_ID, coef_t>::iterator k = s1.begin(); k != s1.end(); k++)
              if ((*k).second != 0)
                cost += 5;
            for (std::map<Variable_ID, coef_t>::iterator k = s2.begin(); k != s2.end(); k++)
              if ((*k).second != 0)
                cost += 2;
            for (std::map<Variable_ID, coef_t>::iterator k = s3.begin(); k != s3.end(); k++)
              if ((*k).second != 0)
                cost += 15;
          }
          
          if (cost < best_cost) {
            best_cost = cost;
            simplest_lb = i;
            simplest_ub = j;
          }
        }
    }
    
    // prepare the new transformation relations
    for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++) {
      Relation r(stmt[*i].xform.n_out(), stmt[*i].xform.n_out()+2);
      F_And *f_root = r.add_and();
      for (int j = 0; j < outer_dim-1; j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.output_var(j+1), 1);
        h.update_coef(r.input_var(j+1), -1);
      }
      
      for (int j = outer_dim-1; j < stmt[*i].xform.n_out(); j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.output_var(j+3), 1);
        h.update_coef(r.input_var(j+1), -1);
      }
      
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(r.output_var(outer_dim), 1);
      h.update_const(-lex[outer_dim-1]);
      
      stmt[*i].xform = Composition(r, stmt[*i].xform);
    }
    
    // add tiling constraints.
    for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++) {    
      F_And *f_super_root = stmt[*i].xform.and_with_and();
      F_Exists *f_exists = f_super_root->add_exists();
      F_And *f_root = f_exists->add_and();
      
      // create a lower bound variable for easy formula creation later
      Variable_ID aligned_lb;
      {
        Variable_ID lb = f_exists->declare();
        coef_t coef = lb_list[simplest_lb].get_coef(bound.set_var(dim+1));
        if (coef == 1) { // e.g. if i >= m+5, then LB = m+5
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(lb, 1);
          for (Constr_Vars_Iter ci(lb_list[simplest_lb]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              if (pos != dim + 1)
                h.update_coef(stmt[*i].xform.output_var(pos), (*ci).coef);
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = stmt[*i].xform.get_local(g);
              else
                v = stmt[*i].xform.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot handle tile bounds");
            }
          }
          h.update_const(lb_list[simplest_lb].get_const());
        }
        else { // e.g. if 2i >= m+5, then m+5 <= 2*LB < m+5+2
          GEQ_Handle h1 = f_root->add_GEQ();
          GEQ_Handle h2 = f_root->add_GEQ();
          for (Constr_Vars_Iter ci(lb_list[simplest_lb]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              if (pos == dim + 1) {
                h1.update_coef(lb, (*ci).coef);
                h2.update_coef(lb, -(*ci).coef);
              }
              else {
                h1.update_coef(stmt[*i].xform.output_var(pos), (*ci).coef);
                h2.update_coef(stmt[*i].xform.output_var(pos), -(*ci).coef);
              }
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = stmt[*i].xform.get_local(g);
              else
                v = stmt[*i].xform.get_local(g, (*ci).var->function_of());
              h1.update_coef(v, (*ci).coef);
              h2.update_coef(v, -(*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot handle tile bounds");
            }
          }
          h1.update_const(lb_list[simplest_lb].get_const());
          h2.update_const(-lb_list[simplest_lb].get_const());
          h2.update_const(coef-1);
        }
        
        Variable_ID offset_lb;
        if (alignment_offset == 0)
          offset_lb = lb;
        else {
          EQ_Handle h = f_root->add_EQ();
          offset_lb = f_exists->declare();
          h.update_coef(offset_lb, 1);
          h.update_coef(lb, -1);
          h.update_const(alignment_offset);
        }
        
        if (alignment_multiple == 1) { // trivial
          aligned_lb = offset_lb;
        }
        else { // e.g. to align at 4, aligned_lb = 4*alpha && LB-4 < 4*alpha <= LB
          aligned_lb = f_exists->declare();
          Variable_ID e = f_exists->declare();
          
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(aligned_lb, 1);
          h.update_coef(e, -alignment_multiple);
          
          GEQ_Handle h1 = f_root->add_GEQ();
          GEQ_Handle h2 = f_root->add_GEQ();
          h1.update_coef(e, alignment_multiple);
          h2.update_coef(e, -alignment_multiple);
          h1.update_coef(offset_lb, -1);
          h2.update_coef(offset_lb, 1);
          h1.update_const(alignment_multiple-1);
        }
      }
      
      // create an upper bound variable for easy formula creation later
      Variable_ID ub = f_exists->declare();
      {
        coef_t coef = -ub_list[simplest_ub].get_coef(bound.set_var(dim+1));
        if (coef == 1) { // e.g. if i <= m+5, then UB = m+5
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(ub, -1);
          for (Constr_Vars_Iter ci(ub_list[simplest_ub]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              if (pos != dim + 1)
                h.update_coef(stmt[*i].xform.output_var(pos), (*ci).coef);
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = stmt[*i].xform.get_local(g);
              else
                v = stmt[*i].xform.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot handle tile bounds");
            }
          }
          h.update_const(ub_list[simplest_ub].get_const());
        }
        else { // e.g. if 2i <= m+5, then m+5-2 < 2*UB <= m+5
          GEQ_Handle h1 = f_root->add_GEQ();
          GEQ_Handle h2 = f_root->add_GEQ();
          for (Constr_Vars_Iter ci(ub_list[simplest_ub]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              if (pos == dim + 1) {
                h1.update_coef(ub, -(*ci).coef);
                h2.update_coef(ub, (*ci).coef);
              }
              else {
                h1.update_coef(stmt[*i].xform.output_var(pos), -(*ci).coef);
                h2.update_coef(stmt[*i].xform.output_var(pos), (*ci).coef);
              }
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = stmt[*i].xform.get_local(g);
              else
                v = stmt[*i].xform.get_local(g, (*ci).var->function_of());
              h1.update_coef(v, -(*ci).coef);
              h2.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot handle tile bounds");
            }
          }
          h1.update_const(-ub_list[simplest_ub].get_const());
          h2.update_const(ub_list[simplest_ub].get_const());
          h1.update_const(coef-1);
        }
      }
      
      // insert tile controlling loop constraints
      if (method == StridedTile) { // e.g. ii = LB + 32 * alpha && alpha >= 0
        Variable_ID e = f_exists->declare();
        GEQ_Handle h1 = f_root->add_GEQ();
        h1.update_coef(e, 1);
        
        EQ_Handle h2 = f_root->add_EQ();
        h2.update_coef(stmt[*i].xform.output_var(outer_dim+1), 1);
        h2.update_coef(e, -tile_size);
        h2.update_coef(aligned_lb, -1);
      }        
      else if (method == CountedTile) { // e.g. 0 <= ii < ceiling((UB-LB+1)/32)
        GEQ_Handle h1 = f_root->add_GEQ();
        h1.update_coef(stmt[*i].xform.output_var(outer_dim+1), 1);
        
        GEQ_Handle h2 = f_root->add_GEQ();
        h2.update_coef(stmt[*i].xform.output_var(outer_dim+1), -tile_size);
        h2.update_coef(aligned_lb, -1);
        h2.update_coef(ub, 1);
      }
      
      // special care for private statements like overflow assignment
      if (private_stmt.find(*i) != private_stmt.end()) { // e.g. ii <= UB
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(stmt[*i].xform.output_var(outer_dim+1), -1); 
        h.update_coef(ub, 1);
      }       
      // if (private_stmt.find(*i) != private_stmt.end()) {
      //   if (stmt[*i].xform.n_out() > dim+3) { // e.g. ii <= UB && i = ii
      //     GEQ_Handle h = f_root->add_GEQ();
      //     h.update_coef(stmt[*i].xform.output_var(outer_dim+1), -1); 
      //     h.update_coef(ub, 1);
      
      //     stmt[*i].xform = Project(stmt[*i].xform, dim+3, Output_Var);
      //     f_root = stmt[*i].xform.and_with_and();
      //     EQ_Handle h1 = f_root->add_EQ();
      //     h1.update_coef(stmt[*i].xform.output_var(dim+3), 1);
      //     h1.update_coef(stmt[*i].xform.output_var(outer_dim+1), -1);
      //   }
      //   else if (method == StridedTile) { // e.g. ii <= UB since i does not exist
      //     GEQ_Handle h = f_root->add_GEQ();
      //     h.update_coef(stmt[*i].xform.output_var(outer_dim+1), -1); 
      //     h.update_coef(ub, 1);
      //   } 
      // }
      
      // restrict original loop index inside the tile
      else {
        if (method == StridedTile) { // e.g. ii <= i < ii + tile_size
          GEQ_Handle h1 = f_root->add_GEQ();
          h1.update_coef(stmt[*i].xform.output_var(dim+3), 1);
          h1.update_coef(stmt[*i].xform.output_var(outer_dim+1), -1);
          
          GEQ_Handle h2 = f_root->add_GEQ();
          h2.update_coef(stmt[*i].xform.output_var(dim+3), -1);
          h2.update_coef(stmt[*i].xform.output_var(outer_dim+1), 1);
          h2.update_const(tile_size-1);
        }
        else if (method == CountedTile) { // e.g. LB+32*ii <= i < LB+32*ii+tile_size
          GEQ_Handle h1 = f_root->add_GEQ();
          h1.update_coef(stmt[*i].xform.output_var(outer_dim+1), -tile_size);
          h1.update_coef(stmt[*i].xform.output_var(dim+3), 1);
          h1.update_coef(aligned_lb, -1);
          
          GEQ_Handle h2 = f_root->add_GEQ();
          h2.update_coef(stmt[*i].xform.output_var(outer_dim+1), tile_size);
          h2.update_coef(stmt[*i].xform.output_var(dim+3), -1);
          h2.update_const(tile_size-1);
          h2.update_coef(aligned_lb, 1);          
        }
      }
    }
  }
  
  // update loop level information
  for (std::set<int>::iterator i = same_tile_controlling_loop.begin(); i != same_tile_controlling_loop.end(); i++) {
    for (int j = 1; j <= stmt[*i].loop_level.size(); j++)
      switch (stmt[*i].loop_level[j-1].type) {
      case LoopLevelOriginal:
        break;
      case LoopLevelTile:
        if (stmt[*i].loop_level[j-1].payload >= outer_level)
          stmt[*i].loop_level[j-1].payload++;
        break;
      default:
        throw loop_error("unknown loop level type for statement " + to_string(*i));
      }
    
    LoopLevel ll;
    ll.type = LoopLevelTile;
    ll.payload = level+1;
    ll.parallel_level = 0;
    stmt[*i].loop_level.insert(stmt[*i].loop_level.begin()+(outer_level-1), ll);
  }
}



std::set<int> Loop::unroll(int stmt_num, int level, int unroll_amount) {
  // check for sanity of parameters
  if (unroll_amount < 0)
    throw std::invalid_argument("invalid unroll amount " + to_string(unroll_amount));
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  int dim = 2*level - 1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  // nothing to do
  if (unroll_amount == 1)
    return std::set<int>();
  
  // extract the intersection of the iteration space to be considered
  Relation hull = Relation::True(level);
  apply_xform(same_loop);
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
    if (stmt[*i].IS.is_upper_bound_satisfiable()) {
      Relation mapping(stmt[*i].IS.n_set(), level);
      F_And *f_root = mapping.add_and();
      for (int j = 1; j <= level; j++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(j), 1);
        h.update_coef(mapping.output_var(j), -1);
      }
      hull = Intersection(hull, Range(Restrict_Domain(mapping, copy(stmt[*i].IS))));
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
  if (!bound.has_single_conjunct() || !bound.is_satisfiable() || bound.is_tautology())
    throw loop_error("unable to extract loop bound for unrolling");
  
  // extract the loop stride
  EQ_Handle stride_eq;
  int stride = 1;
  {
    bool simple_stride = true;
    int strides = countStrides(bound.query_DNF()->single_conjunct(), bound.set_var(level), stride_eq, simple_stride);
    if (strides > 1)
      throw loop_error("too many strides");
    else if (strides == 1) {
      int sign = stride_eq.get_coef(bound.set_var(level));
      Constr_Vars_Iter it(stride_eq, true);
      stride = abs((*it).coef/sign);
    }
  }
  
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
  std::vector<std::vector<std::map<Variable_ID, int> > > overflow_table(lb_list.size(), std::vector<std::map<Variable_ID, int> >(ub_list.size(), std::map<Variable_ID, int>()));
  bool is_overflow_simplifiable = true;
  for (int i = 0; i < lb_list.size(); i++) {
    if (!is_overflow_simplifiable)
      break;
    
    for (int j = 0; j < ub_list.size(); j++) {
      // lower bound or upper bound has non-unit coefficient, can't simplify
      if (ub_list[j].get_coef(bound.set_var(level)) != -1 || lb_list[i].get_coef(bound.set_var(level)) != 1) {
        is_overflow_simplifiable = false;
        break;
      }
      
      for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
        switch((*ci).var->kind()) {
        case Input_Var:
        {
          if ((*ci).var != bound.set_var(level))
            overflow_table[i][j][(*ci).var] += (*ci).coef;
          
          break;
        }
        case Global_Var:
        {
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
        switch((*ci).var->kind()) {
        case Input_Var:
        {
          if ((*ci).var != bound.set_var(level)) {
            overflow_table[i][j][(*ci).var] += (*ci).coef;
            if (overflow_table[i][j][(*ci).var] == 0)
              overflow_table[i][j].erase(overflow_table[i][j].find((*ci).var));
          }
          break;
        }
        case Global_Var:
        {
          Global_Var_ID g = (*ci).var->get_global_var();
          Variable_ID v;
          if (g->arity() == 0)
            v = bound.get_local(g);
          else
            v = bound.get_local(g, (*ci).var->function_of());
          overflow_table[i][j][(*ci).var] += (*ci).coef;
          if (overflow_table[i][j][(*ci).var] == 0)
            overflow_table[i][j].erase(overflow_table[i][j].find((*ci).var));
          break;
        }
        default:
          throw loop_error("failed to calculate overflow amount");
        }
      }
      overflow_table[i][j][NULL] += lb_list[i].get_const();
      
      overflow_table[i][j][NULL] += stride;
      if (unroll_amount == 0 || (overflow_table[i][j].size() == 1 && overflow_table[i][j][NULL]/stride < unroll_amount))
        unroll_amount = overflow_table[i][j][NULL]/stride;
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
          for (std::map<Variable_ID, int>::iterator k = overflow_table[i][j].begin(); k != overflow_table[i][j].end(); )
            if ((*k).first != NULL) {
              int t = int_mod_hat((*k).second, unroll_amount);
              if (t == 0) {
                overflow_table[i][j].erase(k++);
              }
              else {
                int t2 = hull.query_variable_mod((*k).first, unroll_amount);
                if (t2 != INT_MAX) {
                  overflow_table[i][j][NULL] += t * t2;
                  overflow_table[i][j].erase(k++);
                }
                else {
                  (*k).second = t;
                  k++;
                }
              }
            }
            else
              k++;
          
          overflow_table[i][j][NULL] = int_mod_hat(overflow_table[i][j][NULL], unroll_amount);
          
          // Since we don't have MODULO instruction in SUIF yet (only MOD), make all coef positive in the final formula
          for (std::map<Variable_ID, int>::iterator k = overflow_table[i][j].begin(); k != overflow_table[i][j].end(); k++)
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
        h.update_const(((overflow_table[0][i][NULL]/stride)%unroll_amount) * -stride);
      }
      else {
        // upper splitting condition
        std::string over_name = overflow_var_name_prefix + to_string(overflow_var_name_counter++);
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
        h2.update_const(unroll_amount-1);
        
        // create overflow assignment
        bound.setup_names();
        CG_outputRepr *rhs = NULL;
        for (std::map<Variable_ID, int>::iterator j = overflow_table[0][i].begin(); j != overflow_table[0][i].end(); j++)
          if ((*j).first != NULL) {
            CG_outputRepr *t = ocg->CreateIdent((*j).first->name());
            if ((*j).second != 1)
              t = ocg->CreateTimes(ocg->CreateInt((*j).second), t);
            rhs = ocg->CreatePlus(rhs, t);
          }
          else
            if ((*j).second != 0)
              rhs = ocg->CreatePlus(rhs, ocg->CreateInt((*j).second));
        
        if (stride != 1)
          rhs = ocg->CreateIntegerCeil(rhs, ocg->CreateInt(stride));
        rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
        
        CG_outputRepr *lhs = ocg->CreateIdent(over_name);
        init_code = ocg->StmtListAppend(init_code, ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
        lhs = ocg->CreateIdent(over_name);
        overflow_code = ocg->StmtListAppend(overflow_code, ocg->CreateAssignment(0, lhs, rhs));
      }
    }
    
    // lower splitting condition
    GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[0]);
  }
  else if (is_overflow_simplifiable && ub_list.size() == 1) {
    for (int i = 0; i < lb_list.size(); i++) {
      
      if (overflow_table[i][0].size() == 1) {
        // lower splitting condition
        GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[i]);
        h.update_const(overflow_table[i][0][NULL] * -stride);
      }
      else {
        // lower splitting condition
        std::string over_name = overflow_var_name_prefix + to_string(overflow_var_name_counter++);
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
        h2.update_const(unroll_amount-1);
        
        // create overflow assignment
        bound.setup_names();
        CG_outputRepr *rhs = NULL;
        for (std::map<Variable_ID, int>::iterator j = overflow_table[0][i].begin(); j != overflow_table[0][i].end(); j++)
          if ((*j).first != NULL) {
            CG_outputRepr *t = ocg->CreateIdent((*j).first->name());
            if ((*j).second != 1)
              t = ocg->CreateTimes(ocg->CreateInt((*j).second), t);
            rhs = ocg->CreatePlus(rhs, t);
          }
          else
            if ((*j).second != 0)
              rhs = ocg->CreatePlus(rhs, ocg->CreateInt((*j).second));
        
        if (stride != 1)
          rhs = ocg->CreateIntegerCeil(rhs, ocg->CreateInt(stride));
        rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
        
        CG_outputRepr *lhs = ocg->CreateIdent(over_name);
        init_code = ocg->StmtListAppend(init_code, ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
        lhs = ocg->CreateIdent(over_name);
        overflow_code = ocg->StmtListAppend(overflow_code, ocg->CreateAssignment(0, lhs, rhs));
      }
    }
    
    // upper splitting condition
    GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[0]);
  }
  else {
    std::string over_name = overflow_var_name_prefix + to_string(overflow_var_name_counter++);
    Free_Var_Decl *over_free_var = new Free_Var_Decl(over_name);
    over_var_list.push_back(over_free_var);
    
    Tuple<CG_outputRepr *> lb_repr_list, ub_repr_list;
    for (int i = 0; i < lb_list.size(); i++) {
      //lb_repr_list.append(outputLBasRepr(ocg, lb_list[i], bound, bound.set_var(dim+1), stride, stride_eq, Relation::True(bound.n_set()), std::vector<CG_outputRepr *>(bound.n_set(), NULL)));
      lb_repr_list.append(outputLBasRepr(ocg, lb_list[i], bound, bound.set_var(dim+1), stride, stride_eq, Relation::True(bound.n_set()), std::vector<CG_outputRepr *>(bound.n_set())));
      GEQ_Handle h = cond_lower.and_with_GEQ(lb_list[i]);
    }
    for (int i = 0; i < ub_list.size(); i++) {
      //ub_repr_list.append(outputUBasRepr(ocg, ub_list[i], bound, bound.set_var(dim+1), stride, stride_eq, std::vector<CG_outputRepr *>(bound.n_set(), NULL)));
      ub_repr_list.append(outputUBasRepr(ocg, ub_list[i], bound, bound.set_var(dim+1), stride, stride_eq, std::vector<CG_outputRepr *>(bound.n_set())));
      GEQ_Handle h = cond_upper.and_with_GEQ(ub_list[i]);
      h.update_coef(cond_upper.get_local(over_free_var), -stride);
    }
    
    CG_outputRepr *lbRepr, *ubRepr;
    if (lb_repr_list.size() > 1)
      lbRepr = ocg->CreateInvoke("max", lb_repr_list);
    else if (lb_repr_list.size() == 1)
      lbRepr = lb_repr_list[1];
    
    if (ub_repr_list.size() > 1)
      ubRepr = ocg->CreateInvoke("min", ub_repr_list);
    else if (ub_repr_list.size() == 1)
      ubRepr = ub_repr_list[1];
    
    // create overflow assignment
    bound.setup_names();
    CG_outputRepr *rhs = ocg->CreatePlus(ocg->CreateMinus(ubRepr, lbRepr), ocg->CreateInt(1));
    if (stride != 1)
      rhs = ocg->CreateIntegerDivide(rhs, ocg->CreateInt(stride));
    rhs = ocg->CreateIntegerMod(rhs, ocg->CreateInt(unroll_amount));
    CG_outputRepr *lhs = ocg->CreateIdent(over_name);
    init_code = ocg->StmtListAppend(init_code, ocg->CreateAssignment(0, lhs, ocg->CreateInt(0)));
    lhs = ocg->CreateIdent(over_name);
    overflow_code = ocg->CreateAssignment(0, lhs, rhs);
    
    // insert constraint 0 <= overflow < unroll_amount
    Variable_ID v = overflow_constraint.get_local(over_free_var);
    GEQ_Handle h1 = overflow_constraint_root->add_GEQ();
    h1.update_coef(v, 1);
    GEQ_Handle h2 = overflow_constraint_root->add_GEQ();
    h2.update_coef(v, -1);
    h2.update_const(unroll_amount-1);
  }
  
  // insert overflow statement
  int overflow_stmt_num = -1;
  if (overflow_code != NULL) {
    // build iteration space for overflow statement
    Relation mapping(level, level-1);
    F_And *f_root = mapping.add_and();
    for (int i = 1; i < level; i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_coef(mapping.input_var(i), -1);
    }
    Relation overflow_IS = Range(Restrict_Domain(mapping, copy(hull)));
    for (int i = 1; i < level; i++)
      overflow_IS.name_set_var(i, hull.set_var(i)->name());
    overflow_IS.setup_names();  
    
    // build dumb transformation relation for overflow statement
    Relation overflow_xform(level-1, 2*(level-1)+1);
    f_root = overflow_xform.add_and();
    for (int i = 1; i <= level-1; i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(overflow_xform.output_var(2*i), 1);
      h.update_coef(overflow_xform.input_var(i), -1);
      
      h = f_root->add_EQ();
      h.update_coef(overflow_xform.output_var(2*i-1), 1);
      h.update_const(-lex[2*i-2]);
    }
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(overflow_xform.output_var(2*(level-1)+1), 1);
    h.update_const(-lex[2*(level-1)]);
    
    shiftLexicalOrder(lex, dim-1, 1);
    Statement overflow_stmt;
    overflow_stmt.code = overflow_code;
    overflow_stmt.IS = overflow_IS;
    overflow_stmt.xform = overflow_xform;
    overflow_stmt.loop_level = std::vector<LoopLevel>(level-1);
    for (int i = 0; i < level-1; i++) {
      overflow_stmt.loop_level[i].type = stmt[stmt_num].loop_level[i].type;
      if (stmt[stmt_num].loop_level[i].type == LoopLevelTile &&
          stmt[stmt_num].loop_level[i].payload >= level)
        overflow_stmt.loop_level[i].payload = -1;
      else
        overflow_stmt.loop_level[i].payload = stmt[stmt_num].loop_level[i].payload;
      overflow_stmt.loop_level[i].parallel_level = stmt[stmt_num].loop_level[i].parallel_level;
    }
    stmt.push_back(overflow_stmt);
    dep.insert();
    overflow_stmt_num = stmt.size() - 1;
    overflow[overflow_stmt_num] = over_var_list;
    
    // update the global known information on overflow variable
    this->known = Intersection(this->known, Extend_Set(copy(overflow_constraint), this->known.n_set()-overflow_constraint.n_set()));
    
    // update dependence graph
    DependenceVector dv;
    dv.type = DEP_CONTROL;
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      dep.connect(overflow_stmt_num, *i, dv);
    dv.type = DEP_W2W;
    {
      IR_ScalarSymbol *overflow_sym = NULL;
      std::vector<IR_ScalarRef *> scalars = ir->FindScalarRef(overflow_code);
      for (int i = scalars.size()-1; i >=0; i--)
        if (scalars[i]->is_write()) {
          overflow_sym = scalars[i]->symbol();
          break;
        }
      for (int i = scalars.size()-1; i >=0; i--)
        delete scalars[i];
      dv.sym = overflow_sym;
    }
    dv.lbounds = std::vector<coef_t>(num_dep_dim, 0);
    dv.ubounds = std::vector<coef_t>(num_dep_dim, 0);
    int dep_dim = get_last_dep_dim_before(stmt_num, level);
    for (int i = dep_dim + 1; i < num_dep_dim; i++) {
      dv.lbounds[i] = -posInfinity;
      dv.ubounds[i] = posInfinity;
    }
    for (int i = 0; i <= dep_dim; i++) {
      if (i != 0) {
        dv.lbounds[i-1] = 0;
        dv.ubounds[i-1] = 0;
      }
      dv.lbounds[i] = 1;
      dv.ubounds[i] = posInfinity;
      dep.connect(overflow_stmt_num, overflow_stmt_num, dv);
    }
  }
  
  // split the loop so it can be fully unrolled
  std::set<int> result = split(stmt_num, level, cond_upper);
  std::set<int> result2 = split(stmt_num, level, cond_lower);
  for (std::set<int>::iterator i = result2.begin(); i != result2.end(); i++)
    result.insert(*i);
  
  // check if unrolled statements can be trivially lumped together as one statement
  bool can_be_lumped = true;
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      if (*i != stmt_num) {
        if (stmt[*i].loop_level.size() != stmt[stmt_num].loop_level.size()) {
          can_be_lumped = false;
          break;
        }
        for (int j = 0; j < stmt[stmt_num].loop_level.size(); j++)
          if (!(stmt[*i].loop_level[j].type == stmt[stmt_num].loop_level[j].type &&
                stmt[*i].loop_level[j].payload == stmt[stmt_num].loop_level[j].payload)) {
            can_be_lumped = false;
            break;
          }
        if (!can_be_lumped)
          break;
        std::vector<int> lex2 = getLexicalOrder(*i);
        for (int j = 2*level; j < lex.size()-1; j+=2)
          if (lex[j] != lex2[j]) {
            can_be_lumped = false;
            break;
          }
        if (!can_be_lumped)
          break;
      }
  }
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      if (is_inner_loop_depend_on_level(stmt[*i].IS, level, known)) {
        can_be_lumped = false;
        break;
      }
  }
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      if (*i != stmt_num) {
        if (!(Must_Be_Subset(copy(stmt[*i].IS), copy(stmt[stmt_num].IS)) && Must_Be_Subset(copy(stmt[stmt_num].IS), copy(stmt[*i].IS)))) {
          can_be_lumped = false;
          break;
        }
      }
  }    
  if (can_be_lumped) {
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[*i].second.begin(); j != dep.vertex[*i].second.end(); j++)
        if (same_loop.find(j->first) != same_loop.end()) {        
          for (int k = 0; k < j->second.size(); k++)
            if (j->second[k].type == DEP_CONTROL || j->second[k].type == DEP_UNKNOWN) {
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
  
  
  // add strides to original statements
  // for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
  //   add_loop_stride(stmt[*i].IS, bound, level-1, unroll_amount * stride);
  
  
  // std::vector<Free_Var_Decl *> depending_overflow_var;
  // for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
  //   add_loop_stride(stmt[*i].IS, bound, level-1, unroll_amount * stride);
  //   if (overflow.find(*i) != overflow.end()) {
  //     // TO DO: It should check whether overflow vaiable depends on
  //     // this loop index and by how much.  This step is important if
  //     // you want to unroll loops in arbitrary order.
  //     depending_overflow_var.insert(depending_overflow_var.end(), overflow[*i].begin(), overflow[*i].end());
  
  //     continue;
  //   }
  // }
  
  
  
//   std::map<int, std::vector<Statement> > pending;
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
//     add_loop_stride(stmt[*i].IS, bound, level-1, unroll_amount * stride);
  
//     if (overflow.find(*i) != overflow.end()) {
//       // TO DO: It should check whether overflow vaiable depends on
//       // this loop index and by how much.  This step is important if
//       // you want to unroll loops in arbitrary order.
//       depending_overflow_var.insert(depending_overflow_var.end(), overflow[*i].begin(), overflow[*i].end());
  
//       continue;
//     }
  
//     // create copy for each unroll amount
//     for (int j = 1; j < unroll_amount; j++) {
//       Tuple<CG_outputRepr *> funcList;
//       Tuple<std::string> loop_vars;
//       loop_vars.append(stmt[*i].IS.set_var((dim+1)/2)->name());
//       funcList.append(ocg->CreatePlus(ocg->CreateIdent(stmt[*i].IS.set_var(level)->name()), ocg->CreateInt(j*stride)));
//       CG_outputRepr *code = ocg->CreatePlaceHolder(0, stmt[*i].code->clone(), funcList, loop_vars);
  
//       // prepare the new statment to insert
//       Statement unrolled_stmt;
//       unrolled_stmt.IS = copy(stmt[*i].IS);
// //      adjust_loop_bound(unrolled_stmt.IS, (dim-1)/2, j);
//       unrolled_stmt.xform = copy(stmt[*i].xform);
//       unrolled_stmt.code = code;
//       unrolled_stmt.loop_level = stmt[*i].loop_level;
//       pending[*i].push_back(unrolled_stmt);
//     }
//   }
  
//   // adjust iteration space due to loop bounds depending on this loop
//   // index and affected overflow variables
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
//     for (int j = 0; j < pending[*i].size(); j++) {
//       adjust_loop_bound(pending[*i][j].IS, (dim-1)/2, j+1, depending_overflow_var);
//       //pending[*i][j].IS = Intersection(pending[*i][j].IS, Extend_Set(copy(this->known), pending[*i][j].IS.n_set() - this->known.n_set()));
//     }
//   }
  
  // insert unrolled statements
  int old_num_stmt = stmt.size();
  if (!can_be_lumped) {
    std::map<int, std::vector<int> > what_stmt_num;
    
    for (int j = 1; j < unroll_amount; j++) {
      for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
        Statement new_stmt;
        
        Tuple<CG_outputRepr *> funcList;
        Tuple<std::string> loop_vars;
        loop_vars.append(stmt[*i].IS.set_var(level)->name());
        funcList.append(ocg->CreatePlus(ocg->CreateIdent(stmt[*i].IS.set_var(level)->name()), ocg->CreateInt(j*stride)));
        new_stmt.code = ocg->CreatePlaceHolder(0, stmt[*i].code->clone(), funcList, loop_vars);
        
        new_stmt.IS = adjust_loop_bound(stmt[*i].IS, level, j * stride);
        add_loop_stride(new_stmt.IS, bound, level-1, unroll_amount * stride);
        
        new_stmt.xform = copy(stmt[*i].xform);
        new_stmt.loop_level = stmt[*i].loop_level;
        stmt.push_back(new_stmt);
        dep.insert();
        what_stmt_num[*i].push_back(stmt.size() - 1);
      }
    }
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      add_loop_stride(stmt[*i].IS, bound, level-1, unroll_amount * stride);      
    
    
    // update dependence graph
    if (stmt[stmt_num].loop_level[level-1].type == LoopLevelOriginal) {
      int dep_dim = stmt[stmt_num].loop_level[level-1].payload;
      int new_stride = unroll_amount * stride;
      for (int i = 0; i < old_num_stmt; i++) {
        std::vector<std::pair<int, DependenceVector> > D;
        
        for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); ) {
          if (same_loop.find(i) != same_loop.end()) {
            if (same_loop.find(j->first) != same_loop.end()) {
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.type == DEP_CONTROL || dv.type == DEP_UNKNOWN) {
                  D.push_back(std::make_pair(j->first, dv));
                  for (int kk = 0; kk < unroll_amount - 1; kk++)
                    if (what_stmt_num[i][kk] != -1 && what_stmt_num[j->first][kk] != -1)
                      dep.connect(what_stmt_num[i][kk], what_stmt_num[j->first][kk], dv);
                }
                else {
                  coef_t lb = dv.lbounds[dep_dim];
                  coef_t ub = dv.ubounds[dep_dim];
                  if (ub == lb && int_mod(lb, static_cast<coef_t>(new_stride)) == 0) {
                    D.push_back(std::make_pair(j->first, dv));
                    for (int kk = 0; kk < unroll_amount - 1; kk++)
                      if (what_stmt_num[i][kk] != -1 && what_stmt_num[j->first][kk] != -1)
                        dep.connect(what_stmt_num[i][kk], what_stmt_num[j->first][kk], dv);
                  }
                  else if (lb == -posInfinity && ub == posInfinity) {
                    D.push_back(std::make_pair(j->first, dv));
                    for (int kk = 0; kk < unroll_amount; kk++)
                      if (kk == 0)
                        D.push_back(std::make_pair(j->first, dv));
                      else if (what_stmt_num[j->first][kk-1] != -1)
                        D.push_back(std::make_pair(what_stmt_num[j->first][kk-1], dv));
                    for (int t = 0; t < unroll_amount - 1; t++)
                      if (what_stmt_num[i][t] != -1)
                        for (int kk = 0; kk < unroll_amount; kk++)
                          if (kk == 0)
                            dep.connect(what_stmt_num[i][t], j->first, dv);
                          else if (what_stmt_num[j->first][kk-1] != -1)
                            dep.connect(what_stmt_num[i][t], what_stmt_num[j->first][kk-1], dv);
                  }
                  else {
                    for (int kk = 0; kk < unroll_amount; kk++) {
                      if (lb != -posInfinity) {
                        if (kk * stride < int_mod(lb, static_cast<coef_t>(new_stride)))
                          dv.lbounds[dep_dim] = floor(static_cast<double>(lb)/new_stride) * new_stride + new_stride;
                        else
                          dv.lbounds[dep_dim] = floor(static_cast<double>(lb)/new_stride) * new_stride;
                      }
                      if (ub != posInfinity) {
                        if (kk * stride > int_mod(ub, static_cast<coef_t>(new_stride)))
                          dv.ubounds[dep_dim] = floor(static_cast<double>(ub)/new_stride) * new_stride - new_stride;
                        else
                          dv.ubounds[dep_dim] = floor(static_cast<double>(ub)/new_stride) * new_stride;
                      }
                      if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim]) {
                        if (kk == 0)
                          D.push_back(std::make_pair(j->first, dv));
                        else if (what_stmt_num[j->first][kk-1] != -1)
                          D.push_back(std::make_pair(what_stmt_num[j->first][kk-1], dv));
                      }
                    }
                    for (int t = 0; t < unroll_amount-1; t++)
                      if (what_stmt_num[i][t] != -1)
                        for (int kk = 0; kk < unroll_amount; kk++) {
                          if (lb != -posInfinity) {
                            if (kk * stride < int_mod(lb+t+1, static_cast<coef_t>(new_stride)))
                              dv.lbounds[dep_dim] = floor(static_cast<double>(lb+(t+1)*stride)/new_stride) * new_stride + new_stride;
                            else
                              dv.lbounds[dep_dim] = floor(static_cast<double>(lb+(t+1)*stride)/new_stride) * new_stride;
                          }
                          if (ub != posInfinity) {
                            if (kk * stride > int_mod(ub+t+1, static_cast<coef_t>(new_stride)))
                              dv.ubounds[dep_dim] = floor(static_cast<double>(ub+(t+1)*stride)/new_stride) * new_stride - new_stride;
                            else
                              dv.ubounds[dep_dim] = floor(static_cast<double>(ub+(t+1)*stride)/new_stride) * new_stride;
                          }
                          if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim]) {
                            if (kk == 0)
                              dep.connect(what_stmt_num[i][t], j->first, dv);
                            else if (what_stmt_num[j->first][kk-1] != -1)
                              dep.connect(what_stmt_num[i][t], what_stmt_num[j->first][kk-1], dv);
                          }
                        }
                  }
                }
              }
              
              dep.vertex[i].second.erase(j++);
            }
            else {
              for (int kk = 0; kk < unroll_amount - 1; kk++)
                if (what_stmt_num[i][kk] != -1)
                  dep.connect(what_stmt_num[i][kk], j->first, j->second);
              
              j++;
            }
          }
          else {
            if (same_loop.find(j->first) != same_loop.end())
              for (int k = 0; k < j->second.size(); k++)
                for (int kk = 0; kk < unroll_amount - 1; kk++)
                  if (what_stmt_num[j->first][kk] != -1)
                    D.push_back(std::make_pair(what_stmt_num[j->first][kk], j->second[k]));
            j++;
          }
        }
        
        for (int j = 0; j < D.size(); j++)
          dep.connect(i, D[j].first, D[j].second);        
      }
    }
    
    // reset lexical order for the unrolled loop body
    std::set<int> new_same_loop;
    for (std::map<int, std::vector<int> >::iterator i = what_stmt_num.begin(); i != what_stmt_num.end(); i++) {
      new_same_loop.insert(i->first);
      for (int j = 0; j < i->second.size(); j++)
        new_same_loop.insert(i->second[j]);
    }
    setLexicalOrder(dim+1, new_same_loop);
  }
  else {
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      add_loop_stride(stmt[*i].IS, bound, level-1, unroll_amount * stride);
    
    int max_level = stmt[stmt_num].loop_level.size();
    std::vector<std::pair<int, int> > stmt_order;
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
      stmt_order.push_back(std::make_pair(get_const(stmt[*i].xform, 2*max_level, Output_Var), *i));
    sort(stmt_order.begin(), stmt_order.end());
    
    Statement new_stmt;
    new_stmt.code = NULL;
    for (int j = 1; j < unroll_amount; j++)
      for (int i = 0; i < stmt_order.size(); i++) {
        Tuple<CG_outputRepr *> funcList;
        Tuple<std::string> loop_vars;
        loop_vars.append(stmt[stmt_order[i].second].IS.set_var(level)->name());
        funcList.append(ocg->CreatePlus(ocg->CreateIdent(stmt[stmt_order[i].second].IS.set_var(level)->name()), ocg->CreateInt(j*stride)));
        CG_outputRepr *code = ocg->CreatePlaceHolder(0, stmt[stmt_order[i].second].code->clone(), funcList, loop_vars);
        new_stmt.code = ocg->StmtListAppend(new_stmt.code, code);
      }
    
    new_stmt.IS = copy(stmt[stmt_num].IS);
    new_stmt.xform = copy(stmt[stmt_num].xform);
    assign_const(new_stmt.xform, 2*max_level, stmt_order[stmt_order.size()-1].first+1);
    new_stmt.loop_level = stmt[stmt_num].loop_level;
    stmt.push_back(new_stmt);
    dep.insert();
    
    // update dependence graph
    if (stmt[stmt_num].loop_level[level-1].type == LoopLevelOriginal) {
      int dep_dim = stmt[stmt_num].loop_level[level-1].payload;
      int new_stride = unroll_amount * stride;
      for (int i = 0; i < old_num_stmt; i++) {
        std::vector<std::pair<int, std::vector<DependenceVector> > > D;
        
        for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); ) {
          if (same_loop.find(i) != same_loop.end()) {
            if (same_loop.find(j->first) != same_loop.end()) {
              std::vector<DependenceVector> dvs11, dvs12, dvs22, dvs21;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.type == DEP_CONTROL || dv.type == DEP_UNKNOWN) {
                  if (i == j->first) {
                    dvs11.push_back(dv);
                    dvs22.push_back(dv);
                  }
                  else
                    throw loop_error("unrolled statements lumped together illegally");
                }
                else {
                  coef_t lb = dv.lbounds[dep_dim];
                  coef_t ub = dv.ubounds[dep_dim];
                  if (ub == lb && int_mod(lb, static_cast<coef_t>(new_stride)) == 0) {
                    dvs11.push_back(dv);
                    dvs22.push_back(dv);
                  }
                  else {
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = ceil(static_cast<double>(lb)/new_stride) * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = floor(static_cast<double>(ub)/new_stride) * new_stride;
                    if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim])
                      dvs11.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = ceil(static_cast<double>(lb)/new_stride) * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = ceil(static_cast<double>(ub)/new_stride) * new_stride;
                    if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim])
                      dvs21.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = floor(static_cast<double>(lb)/new_stride) * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = floor(static_cast<double>(ub-stride)/new_stride) * new_stride;
                    if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim])
                      dvs12.push_back(dv);
                    
                    if (lb != -posInfinity)
                      dv.lbounds[dep_dim] = floor(static_cast<double>(lb)/new_stride) * new_stride;
                    if (ub != posInfinity)
                      dv.ubounds[dep_dim] = ceil(static_cast<double>(ub-stride)/new_stride) * new_stride;
                    if (dv.ubounds[dep_dim] >= dv.lbounds[dep_dim])
                      dvs22.push_back(dv);
                  }
                }
              }
              if (dvs11.size() > 0)
                D.push_back(std::make_pair(i, dvs11));
              if (dvs22.size() > 0)
                dep.connect(old_num_stmt, old_num_stmt, dvs22);
              if (dvs12.size() > 0)
                D.push_back(std::make_pair(old_num_stmt, dvs12));
              if (dvs21.size() > 0)
                dep.connect(old_num_stmt, i, dvs21);
              
              dep.vertex[i].second.erase(j++);
            }
            else {
              dep.connect(old_num_stmt, j->first, j->second);
              j++;
            }
          }
          else {
            if (same_loop.find(j->first) != same_loop.end()) 
              D.push_back(std::make_pair(old_num_stmt, j->second));
            j++;
          }
        }
        
        for (int j = 0; j < D.size(); j++)
          dep.connect(i, D[j].first, D[j].second);
      }
    }
  }
  
  return result;
}


std::vector<int> Loop::getLexicalOrder(int stmt_num) const {
  assert(stmt_num < stmt.size());
  
  const int n = stmt[stmt_num].xform.n_out();
  std::vector<int> lex(n,0);
  
  for (int i = 0; i < n; i += 2)
    lex[i] = get_const(stmt[stmt_num].xform, i, Output_Var);
  
  return lex;
}

std::set<int> Loop::getStatements(const std::vector<int> &lex, int dim) const {
  const int m = stmt.size();
  
  std::set<int> same_loops;
  for (int i = 0; i < m; i++) {
    if (dim < 0)
      same_loops.insert(i);
    else {
      std::vector<int> a_lex = getLexicalOrder(i);
      int j;
      for (j = 0; j <= dim; j+=2)
        if (lex[j] != a_lex[j])
          break;
      if (j > dim)
        same_loops.insert(i);
    }
  }
  
  return same_loops;
}


void Loop::shiftLexicalOrder(const std::vector<int> &lex, int dim, int amount) {
  const int m = stmt.size();
  
  if (amount == 0)
    return;
  
  for (int i = 0; i < m; i++) {
    std::vector<int> lex2 = getLexicalOrder(i);
    
    bool need_shift = true;
    
    for (int j = 0; j < dim; j++)
      if (lex2[j] != lex[j]) {
        need_shift = false;
        break;
      }
    
    if (!need_shift)
      continue;
    
    if (amount > 0) {
      if (lex2[dim] < lex[dim])
        continue;
    }
    else if (amount < 0) {
      if (lex2[dim] > lex[dim])
        continue;
    }
    
    assign_const(stmt[i].xform, dim, lex2[dim] + amount);
  }
}


void Loop::setLexicalOrder(int dim, const std::set<int> &active, int starting_order) {
  if (active.size() == 0)
    return;
  
  // check for sanity of parameters
  if (dim < 0 || dim % 2 != 0)
    throw std::invalid_argument("invalid constant loop level to set lexicographical order");
  std::vector<int> lex;
  int ref_stmt_num;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if ((*i) < 0 || (*i) >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(*i));
    if (dim >= stmt[*i].xform.n_out())
      throw std::invalid_argument("invalid constant loop level to set lexicographical order");
    if (i == active.begin()) {
      lex = getLexicalOrder(*i);
      ref_stmt_num = *i;
    }
    else {
      std::vector<int> lex2 = getLexicalOrder(*i);
      for (int j = 0; j < dim; j+=2)
        if (lex[j] != lex2[j])
          throw std::invalid_argument("statements are not in the same sub loop nest");
    }
  }
  
  // sepearate statements by current loop level types
  int level = (dim+2)/2;
  std::map<std::pair<LoopLevelType, int>, std::set<int> > active_by_level_type;
  std::set<int> active_by_no_level;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if (level > stmt[*i].loop_level.size())
      active_by_no_level.insert(*i);
    else
      active_by_level_type[std::make_pair(stmt[*i].loop_level[level-1].type, stmt[*i].loop_level[level-1].payload)].insert(*i);
  }
  
  // further separate statements due to control dependences
  std::vector<std::set<int> > active_by_level_type_splitted;
  for (std::map<std::pair<LoopLevelType, int>, std::set<int> >::iterator i = active_by_level_type.begin(); i != active_by_level_type.end(); i++)
    active_by_level_type_splitted.push_back(i->second);
  for (std::set<int>::iterator i = active_by_no_level.begin(); i != active_by_no_level.end(); i++)
    for (int j = active_by_level_type_splitted.size() - 1; j >= 0; j--) {
      std::set<int> controlled, not_controlled;
      for (std::set<int>::iterator k = active_by_level_type_splitted[j].begin(); k != active_by_level_type_splitted[j].end(); k++) {
        std::vector<DependenceVector> dvs = dep.getEdge(*i, *k);
        bool is_controlled = false;
        for (int kk = 0; kk < dvs.size(); kk++)
          if (dvs[kk].type = DEP_CONTROL) {
            is_controlled = true;
            break;
          }
        if (is_controlled)
          controlled.insert(*k);
        else
          not_controlled.insert(*k);
      }
      if (controlled.size() != 0 && not_controlled.size() != 0) {
        active_by_level_type_splitted.erase(active_by_level_type_splitted.begin() + j);
        active_by_level_type_splitted.push_back(controlled);
        active_by_level_type_splitted.push_back(not_controlled);
      }
    }
  
  // set lexical order separating loops with different loop types first
  if (active_by_level_type_splitted.size() + active_by_no_level.size() > 1) {
    int dep_dim = get_last_dep_dim_before(ref_stmt_num, level) + 1;
    
    Graph<std::set<int>, Empty> g;
    for (std::vector<std::set<int> >::iterator i = active_by_level_type_splitted.begin(); i != active_by_level_type_splitted.end(); i++)
      g.insert(*i);
    for (std::set<int>::iterator i = active_by_no_level.begin(); i != active_by_no_level.end(); i++) {
      std::set<int> t;
      t.insert(*i);
      g.insert(t);
    }
    for (int i = 0; i < g.vertex.size(); i++)
      for (int j = i+1; j < g.vertex.size(); j++) {
        bool connected = false;
        for (std::set<int>::iterator ii = g.vertex[i].first.begin(); ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin(); jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*ii, *jj);
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence() ||
                  (dvs[k].is_data_dependence() && !dvs[k].has_been_carried_before(dep_dim))) {
                g.connect(i, j);
                connected = true;
                break;
              }
            if (connected)
              break;
          }
          if (connected)
            break;
        }
        connected = false;
        for (std::set<int>::iterator ii = g.vertex[i].first.begin(); ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin(); jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*jj, *ii);
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence() ||
                  (dvs[k].is_data_dependence() && !dvs[k].has_been_carried_before(dep_dim))) {
                g.connect(j, i);
                connected = true;
                break;
              }
            if (connected)
              break;
          }
          if (connected)
            break;
        }
      }
    
    std::vector<std::set<int> > s = g.topoSort();
    if (s.size() != g.vertex.size())
      throw loop_error("cannot separate statements with different loop types at loop level " + to_string(level));
    
    // assign lexical order
    int order = starting_order;
    for (int i = 0; i < s.size(); i++) {
      std::set<int> &cur_scc = g.vertex[*(s[i].begin())].first;
      int sz = cur_scc.size();
      if (sz == 1) {
        int cur_stmt = *(cur_scc.begin());
        assign_const(stmt[cur_stmt].xform, dim, order);
        for (int j = dim+2; j < stmt[cur_stmt].xform.n_out(); j+=2)
          assign_const(stmt[cur_stmt].xform, j, 0);
        order++;
      }
      else {
        setLexicalOrder(dim, cur_scc, order);
        order += sz;
      }
    }
  }
  // set lexical order seperating single iteration statements and loops
  else {
    std::set<int> true_singles;
    std::set<int> nonsingles;
    std::map<coef_t, std::set<int> > fake_singles;
    
    // sort out statements that do not require loops
    for(std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
      Relation cur_IS = getNewIS(*i);
      if (is_single_iteration(cur_IS, dim+1)) {
        bool is_all_single = true;
        for (int j = dim+3; j < stmt[*i].xform.n_out(); j+=2)
          if (!is_single_iteration(cur_IS, j)) {
            is_all_single = false;
            break;
          }
        if (is_all_single) 
          true_singles.insert(*i);
        else {
          try {
            fake_singles[get_const(cur_IS, dim+1, Set_Var)].insert(*i);
          }
          catch (const std::exception &e) {
            fake_singles[posInfinity].insert(*i);
          }
        }
      }
      else
        nonsingles.insert(*i);
    }
    
    // split nonsingles forcibly according to negative dependences present (loop unfusible)
    int dep_dim = get_dep_dim_of(ref_stmt_num, level);
    Graph<int, Empty> g2;
    for (std::set<int>::iterator i = nonsingles.begin(); i != nonsingles.end(); i++)
      g2.insert(*i);
    for (int i = 0; i < g2.vertex.size(); i++)
      for (int j = i+1; j < g2.vertex.size(); j++) {
        std::vector<DependenceVector> dvs = dep.getEdge(g2.vertex[i].first, g2.vertex[j].first);
        for (int k = 0; k < dvs.size(); k++)
          if (dvs[k].is_control_dependence() ||
              (dvs[k].is_data_dependence() && dvs[k].has_negative_been_carried_at(dep_dim))) {
            g2.connect(i, j);
            break;
          }
        dvs = dep.getEdge(g2.vertex[j].first, g2.vertex[i].first);
        for (int k = 0; k < dvs.size(); k++)
          if (dvs[k].is_control_dependence() ||
              (dvs[k].is_data_dependence() && dvs[k].has_negative_been_carried_at(dep_dim))) {
            g2.connect(j, i);
            break;
          }
      }
    
    std::vector<std::set<int> > s2 = g2.packed_topoSort();
    
    std::vector<std::set<int> > splitted_nonsingles;
    for (int i = 0; i < s2.size(); i++) {
      std::set<int> cur_scc;
      for (std::set<int>::iterator j = s2[i].begin(); j != s2[i].end(); j++)
        cur_scc.insert(g2.vertex[*j].first);
      splitted_nonsingles.push_back(cur_scc);
    }
    
    // convert to dependence graph for grouped statements
    dep_dim = get_last_dep_dim_before(ref_stmt_num, level) + 1;
    Graph<std::set<int>, Empty> g;
    for (std::set<int>::iterator i = true_singles.begin(); i != true_singles.end(); i++) {
      std::set<int> t;
      t.insert(*i);
      g.insert(t);
    }
    for (int i = 0; i < splitted_nonsingles.size(); i++) {
      g.insert(splitted_nonsingles[i]);
    }   
    for (std::map<coef_t, std::set<int> >::iterator i = fake_singles.begin(); i != fake_singles.end(); i++)
      g.insert((*i).second);
    
    for (int i = 0; i < g.vertex.size(); i++)
      for (int j = i + 1; j < g.vertex.size(); j++) {
        bool connected = false;
        for (std::set<int>::iterator ii = g.vertex[i].first.begin(); ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin(); jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*ii, *jj);
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence() ||
                  (dvs[k].is_data_dependence() && !dvs[k].has_been_carried_before(dep_dim))) {
                g.connect(i, j);
                connected = true;
                break;
              }
            if (connected)
              break;
          }
          if (connected)
            break;
        }
        connected = false;
        for (std::set<int>::iterator ii = g.vertex[i].first.begin(); ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin(); jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*jj, *ii);
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence() ||
                  (dvs[k].is_data_dependence() && !dvs[k].has_been_carried_before(dep_dim))) {
                g.connect(j, i);
                connected = true;
                break;
              }
            if (connected)
              break;
          }
          if (connected)
            break;  
        }
      }
    
    // topological sort according to chun's permute algorithm
    std::vector<std::set<int> > s = g.topoSort();
    
    // assign lexical order
    int order = starting_order;
    for (int i = 0; i < s.size(); i++) {
      // translate each SCC into original statements
      std::set<int> cur_scc;
      for (std::set<int>::iterator j = s[i].begin(); j != s[i].end(); j++)
        copy(g.vertex[*j].first.begin(), g.vertex[*j].first.end(), inserter(cur_scc, cur_scc.begin()));
      
      // now assign the constant
      for(std::set<int>::iterator j = cur_scc.begin(); j != cur_scc.end(); j++)
        assign_const(stmt[*j].xform, dim, order);
      
      if (cur_scc.size() > 1)
        setLexicalOrder(dim+2, cur_scc);
      else if (cur_scc.size() == 1) {
        int cur_stmt =*(cur_scc.begin());
        for (int j = dim+2; j < stmt[cur_stmt].xform.n_out(); j+=2)
          assign_const(stmt[cur_stmt].xform, j, 0);
      }
      
      if (cur_scc.size() > 0)
        order++;
    }
  }
}


void Loop::apply_xform() {
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  apply_xform(active);
}


void Loop::apply_xform(int stmt_num) {
  std::set<int> active;
  active.insert(stmt_num);
  apply_xform(active);
}


void Loop::apply_xform(std::set<int> &active) {
  int max_n = 0;
  
  CG_outputBuilder *ocg = ir->builder();
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].loop_level.size();
    if (n > max_n)
      max_n = n;
    
    std::vector<int> lex = getLexicalOrder(*i);
    
    Relation mapping(2*n+1, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(2*j), -1);
    }
    mapping = Composition(mapping, stmt[*i].xform);
    mapping.simplify();
    
    // match omega input/output variables to variable names in the code
    for (int j = 1; j <= stmt[*i].IS.n_set(); j++)
      mapping.name_input_var(j, stmt[*i].IS.set_var(j)->name());
    for (int j = 1; j <= n; j++)
      mapping.name_output_var(j, tmp_loop_var_name_prefix + to_string(tmp_loop_var_name_counter+j-1));
    mapping.setup_names();
    
    Relation known = Extend_Set(copy(this->known), mapping.n_out() - this->known.n_set());
    //stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out(), NULL));
    stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out()));
    stmt[*i].IS = Range(Restrict_Domain(mapping, stmt[*i].IS));
    stmt[*i].IS.simplify();
    
    // replace original transformation relation with straight 1-1 mapping
    mapping = Relation(n, 2*n+1);
    f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2*j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = 1; j <= 2*n+1; j+=2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_const(-lex[j-1]);
    }  
    stmt[*i].xform = mapping;
  }
  
  tmp_loop_var_name_counter += max_n;
}


void Loop::addKnown(const Relation &cond) {
  int n1 = this->known.n_set();
  
  Relation r = copy(cond);
  int n2 = r.n_set();
  
  if (n1 < n2)
    this->known = Extend_Set(this->known, n2-n1);
  else if (n1 > n2)
    r = Extend_Set(r, n1-n2);
  
  this->known = Intersection(this->known, r);
}


bool Loop::nonsingular(const std::vector<std::vector<int> > &T) {
  if (stmt.size() == 0)
    return true;
  
  // check for sanity of parameters
  for (int i = 0; i < stmt.size(); i++) {
    if (stmt[i].loop_level.size() != num_dep_dim)
      throw std::invalid_argument("nonsingular loop transformations must be applied to original perfect loop nest");
    for (int j = 0; j < stmt[i].loop_level.size(); j++)
      if (stmt[i].loop_level[j].type != LoopLevelOriginal)
        throw std::invalid_argument("nonsingular loop transformations must be applied to original perfect loop nest");
  }
  if (T.size() != num_dep_dim)
    throw std::invalid_argument("invalid transformation matrix");
  for (int i = 0; i < stmt.size(); i++)
    if (T[i].size() != num_dep_dim + 1 && T[i].size() != num_dep_dim)
      throw std::invalid_argument("invalid transformation matrix");
  
  // build relation from matrix
  Relation mapping(2*num_dep_dim+1, 2*num_dep_dim+1);
  F_And *f_root = mapping.add_and();
  for (int i = 0; i < num_dep_dim; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(2*(i+1)), -1);
    for (int j = 0; j < num_dep_dim; j++)
      if (T[i][j] != 0) 
        h.update_coef(mapping.input_var(2*(j+1)), T[i][j]);
    if (T[i].size() == num_dep_dim+1)
      h.update_const(T[i][num_dep_dim]);
  }
  for (int i = 1; i <= 2*num_dep_dim+1; i+=2) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(i), -1);
    h.update_coef(mapping.input_var(i), 1);
  }
  
  // update transformation relations
  for (int i = 0; i < stmt.size(); i++)
    stmt[i].xform = Composition(copy(mapping), stmt[i].xform);
  
  // update dependence graph
  for (int i = 0; i < dep.vertex.size(); i++)
    for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++) {
      std::vector<DependenceVector> dvs = j->second;
      for (int k = 0; k < dvs.size(); k++) {
        DependenceVector &dv = dvs[k];
        switch (dv.type) {
        case DEP_W2R:
        case DEP_R2W:
        case DEP_W2W:
        case DEP_R2R: {
          std::vector<coef_t> lbounds(num_dep_dim), ubounds(num_dep_dim);
          for (int p = 0; p < num_dep_dim; p++) {
            coef_t lb = 0;
            coef_t ub = 0;
            for (int q = 0; q < num_dep_dim; q++) {
              if (T[p][q] > 0) {
                if (lb == -posInfinity || dv.lbounds[q] == -posInfinity)
                  lb = -posInfinity;
                else
                  lb += T[p][q] * dv.lbounds[q];
                if (ub == posInfinity || dv.ubounds[q] == posInfinity)
                  ub = posInfinity;
                else
                  ub += T[p][q] * dv.ubounds[q];
              }
              else if (T[p][q] < 0) {
                if (lb == -posInfinity || dv.ubounds[q] == posInfinity)
                  lb = -posInfinity;
                else
                  lb += T[p][q] * dv.ubounds[q];
                if (ub == posInfinity || dv.lbounds[q] == -posInfinity)
                  ub = posInfinity;
                else
                  ub += T[p][q] * dv.lbounds[q];
              }
            }
            if (T[p].size() == num_dep_dim+1) {
              if (lb != -posInfinity)
                lb += T[p][num_dep_dim];
              if (ub != posInfinity)
                ub += T[p][num_dep_dim];
            }
            lbounds[p] = lb;
            ubounds[p] = ub;
          }
          dv.lbounds = lbounds;
          dv.ubounds = ubounds;
          
          break;
        }
        default:
          ;
        }
      }
      j->second = dvs;
    }
  
  // set constant loop values
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  setLexicalOrder(0, active);
  
  return true;
}


void Loop::skew(const std::set<int> &stmt_nums, int level, const std::vector<int> &skew_amount) {
  if (stmt_nums.size() == 0)
    return;
  
  // check for sanity of parameters
  int ref_stmt_num = *(stmt_nums.begin());
  for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(*i));
    if (level < 1 || level > stmt[*i].loop_level.size())
      throw std::invalid_argument("invalid loop level " + to_string(level));
    for (int j = stmt[*i].loop_level.size(); j < skew_amount.size(); j++)
      if (skew_amount[j] != 0)
        throw std::invalid_argument("invalid skewing formula");
  }
  
  // set trasformation relations
  for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
    int n = stmt[*i].xform.n_out();
    Relation r(n,n);
    F_And *f_root = r.add_and();
    for (int j = 1; j <= n; j++)
      if (j != 2*level) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.input_var(j), 1);
        h.update_coef(r.output_var(j), -1);
      }
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(r.output_var(2*level), -1);
    for (int j = 0; j < skew_amount.size(); j++)
      if (skew_amount[j] != 0)
        h.update_coef(r.input_var(2*(j+1)), skew_amount[j]);
    
    stmt[*i].xform = Composition(r, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // update dependence graph
  if (stmt[ref_stmt_num].loop_level[level-1].type == LoopLevelOriginal) {
    int dep_dim = stmt[ref_stmt_num].loop_level[level-1].payload;
    for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++)
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[*i].second.begin(); j != dep.vertex[*i].second.end(); j++)
        if (stmt_nums.find(j->first) != stmt_nums.end()) {
          // dependence between skewed statements
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              coef_t lb = 0;
              coef_t ub = 0;
              for (int kk = 0; kk < skew_amount.size(); kk++) {
                int cur_dep_dim = get_dep_dim_of(*i, kk+1);
                if (skew_amount[kk] > 0) {
                  if (lb != -posInfinity &&
                      stmt[*i].loop_level[kk].type == LoopLevelOriginal &&
                      dv.lbounds[cur_dep_dim] != -posInfinity)
                    lb += skew_amount[kk] * dv.lbounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1 && !(dv.lbounds[cur_dep_dim] == 0 && dv.ubounds[cur_dep_dim] == 0))
                      lb = -posInfinity;
                  }
                  if (ub != posInfinity &&
                      stmt[*i].loop_level[kk].type == LoopLevelOriginal &&
                      dv.ubounds[cur_dep_dim] != posInfinity)
                    ub += skew_amount[kk] * dv.ubounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1 && !(dv.lbounds[cur_dep_dim] == 0 && dv.ubounds[cur_dep_dim] == 0))
                      ub = posInfinity;
                  }
                }
                else if (skew_amount[kk] < 0) {
                  if (lb != -posInfinity &&
                      stmt[*i].loop_level[kk].type == LoopLevelOriginal &&
                      dv.ubounds[cur_dep_dim] != posInfinity)
                    lb += skew_amount[kk] * dv.ubounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1 && !(dv.lbounds[cur_dep_dim] == 0 && dv.ubounds[cur_dep_dim] == 0))
                      lb = -posInfinity;
                  }
                  if (ub != posInfinity &&
                      stmt[*i].loop_level[kk].type == LoopLevelOriginal &&
                      dv.lbounds[cur_dep_dim] != -posInfinity)
                    ub += skew_amount[kk] * dv.lbounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1 && !(dv.lbounds[cur_dep_dim] == 0 && dv.ubounds[cur_dep_dim] == 0))
                      ub = posInfinity;
                  }
                }
              }
              dv.lbounds[dep_dim] = lb;
              dv.ubounds[dep_dim] = ub;
            }
          }
          j->second = dvs;
        }
        else {
          // dependence from skewed statement to unskewed statement becomes jumbled,
          // put distance value at skewed dimension to unknown
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              dv.lbounds[dep_dim] = -posInfinity;
              dv.ubounds[dep_dim] = posInfinity;
            }
          }
          j->second = dvs;
        }
    for (int i = 0; i < dep.vertex.size(); i++)
      if (stmt_nums.find(i) == stmt_nums.end())
        for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++)
          if (stmt_nums.find(j->first) != stmt_nums.end()) {
            // dependence from unskewed statement to skewed statement becomes jumbled,
            // put distance value at skewed dimension to unknown
            std::vector<DependenceVector> dvs = j->second;
            for (int k = 0; k < dvs.size(); k++) {
              DependenceVector &dv = dvs[k];
              if (dv.is_data_dependence()) {
                dv.lbounds[dep_dim] = -posInfinity;
                dv.ubounds[dep_dim] = posInfinity;
              }
            }
            j->second = dvs;
          }
  }
}


void Loop::shift(const std::set<int> &stmt_nums, int level, int shift_amount) {
  if (stmt_nums.size() == 0)
    return;
  
  // check for sanity of parameters
  int ref_stmt_num = *(stmt_nums.begin());
  for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(*i));
    if (level < 1 || level > stmt[*i].loop_level.size())
      throw std::invalid_argument("invalid loop level " + to_string(level));
  }
  
  // do nothing
  if (shift_amount == 0)
    return;
  
  // set trasformation relations
  for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
    int n = stmt[*i].xform.n_out();
    
    Relation r(n, n);
    F_And *f_root = r.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(r.input_var(j), 1);
      h.update_coef(r.output_var(j), -1);
      if (j == 2*level)
        h.update_const(shift_amount);
    }
    
    stmt[*i].xform = Composition(r, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // update dependence graph
  if (stmt[ref_stmt_num].loop_level[level-1].type == LoopLevelOriginal) {
    int dep_dim = stmt[ref_stmt_num].loop_level[level-1].payload;
    for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++)
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[*i].second.begin(); j != dep.vertex[*i].second.end(); j++)
        if (stmt_nums.find(j->first) == stmt_nums.end()) {
          // dependence from shifted statement to unshifted statement
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              if (dv.lbounds[dep_dim] != -posInfinity)
                dv.lbounds[dep_dim] -= shift_amount;
              if (dv.ubounds[dep_dim] != posInfinity)
                dv.ubounds[dep_dim] -= shift_amount;
            }
          }
          j->second = dvs;
        }
    for (int i = 0; i < dep.vertex.size(); i++)
      if (stmt_nums.find(i) == stmt_nums.end())
        for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++)
          if (stmt_nums.find(j->first) != stmt_nums.end()) {
            // dependence from unshifted statement to shifted statement
            std::vector<DependenceVector> dvs = j->second;
            for (int k = 0; k < dvs.size(); k++) {
              DependenceVector &dv = dvs[k];
              if (dv.is_data_dependence()) {
                if (dv.lbounds[dep_dim] != -posInfinity)
                  dv.lbounds[dep_dim] += shift_amount;
                if (dv.ubounds[dep_dim] != posInfinity)
                  dv.ubounds[dep_dim] += shift_amount;
              }
            }
            j->second = dvs;
          }
  }
}



// bool Loop::fuse(const std::set<int> &stmt_nums, int level) {
//   if (stmt_nums.size() == 0 || stmt_nums.size() == 1)
//     return true;
//   int dim = 2*level-1;

//   // check for sanity of parameters
//   std::vector<int> ref_lex;
//   for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
//     if (*i < 0 || *i >= stmt.size())
//       throw std::invalid_argument("invalid statement number " + to_string(*i));
//     if (level < 1 || level > (stmt[*i].xform.n_out()-1)/2)
//       throw std::invalid_argument("invalid loop level " + to_string(level));
//     if (ref_lex.size() == 0)
//       ref_lex = getLexicalOrder(*i);
//     else {
//       std::vector<int> lex = getLexicalOrder(*i);
//       for (int j = 0; j < dim-1; j+=2)
//         if (lex[j] != ref_lex[j])
//           throw std::invalid_argument("statements for fusion must be in the same level-" + to_string(level-1) + " subloop");
//     }
//   }

//   // collect lexicographical order values from to-be-fused statements
//   std::set<int> lex_values;
//   for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
//     std::vector<int> lex = getLexicalOrder(*i);
//     lex_values.insert(lex[dim-1]);
//   }
//   if (lex_values.size() == 1)
//     return true;

//   // negative dependence would prevent fusion
//   int dep_dim = xform_index[dim].first;
//   for (std::set<int>::iterator i = lex_values.begin(); i != lex_values.end(); i++) {
//     ref_lex[dim-1] = *i;
//     std::set<int> a = getStatements(ref_lex, dim-1);
//     std::set<int>::iterator j = i;
//     j++;
//     for (; j != lex_values.end(); j++) {
//       ref_lex[dim-1] = *j;
//       std::set<int> b = getStatements(ref_lex, dim-1);
//       for (std::set<int>::iterator ii = a.begin(); ii != a.end(); ii++)
//         for (std::set<int>::iterator jj = b.begin(); jj != b.end(); jj++) {
//           std::vector<DependenceVector> dvs;
//           dvs = dep.getEdge(*ii, *jj);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim) && dvs[k].hasNegative(dep_dim))
//               throw loop_error("loop error: statements " + to_string(*ii) + " and " + to_string(*jj) + " cannot be fused together due to negative dependence");
//           dvs = dep.getEdge(*jj, *ii);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim) && dvs[k].hasNegative(dep_dim))
//               throw loop_error("loop error: statements " + to_string(*jj) + " and " + to_string(*ii) + " cannot be fused together due to negative dependence");
//         }
//     }
//   }

//   // collect all other lexicographical order values from the subloop
//   // enclosing these to-be-fused loops
//   std::set<int> same_loop = getStatements(ref_lex, dim-3);
//   std::set<int> other_lex_values;
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
//     std::vector<int> lex = getLexicalOrder(*i);
//     if (lex_values.find(lex[dim-1]) == lex_values.end())
//       other_lex_values.insert(lex[dim-1]);
//   }

//   // update to-be-fused loops due to dependence cycle
//   Graph<std::set<int>, Empty> g;
//   {
//     std::set<int> t;
//     for (std::set<int>::iterator i = lex_values.begin(); i != lex_values.end(); i++) {
//       ref_lex[dim-1] = *i;
//       std::set<int> t2 = getStatements(ref_lex, dim-1);
//       std::set_union(t.begin(), t.end(), t2.begin(), t2.end(), inserter(t, t.begin()));
//     }
//     g.insert(t);
//   }
//   for (std::set<int>::iterator i = other_lex_values.begin(); i != other_lex_values.end(); i++) {
//     ref_lex[dim-1] = *i;
//     std::set<int> t = getStatements(ref_lex, dim-1);
//     g.insert(t);
//   }
//   for (int i = 0; i < g.vertex.size(); i++)
//     for (int j = i+1; j < g.vertex.size(); j++)
//       for (std::set<int>::iterator ii = g.vertex[i].first.begin(); ii != g.vertex[i].first.end(); ii++)
//         for (std::set<int>::iterator jj = g.vertex[j].first.begin(); jj != g.vertex[j].first.end(); jj++) {
//           std::vector<DependenceVector> dvs;
//           dvs = dep.getEdge(*ii, *jj);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim)) {
//               g.connect(i, j);
//               break;
//             }
//           dvs = dep.getEdge(*jj, *ii);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim)) {
//               g.connect(j, i);
//               break;
//             }
//         }
//   std::vector<std::set<int> > s = g.topoSort();
//   int fused_lex_value = 0;
//   for (int i = 0; i < s.size(); i++)
//     if (s[i].find(0) != s[i].end()) {
//       // now add additional lexicographical order values
//       for (std::set<int>::iterator j = s[i].begin(); j != s[i].end(); j++)
//         if (*j != 0) {
//           int stmt = *(g.vertex[*j].first.begin());
//           std::vector<int> lex = getLexicalOrder(stmt);
//           lex_values.insert(lex[dim-1]);
//         }

//       if (s.size() > 1) {
//         if (i == 0) {
//           int min_lex_value;
//           for (std::set<int>::iterator j = s[i+1].begin(); j != s[i+1].end(); j++) {
//             int stmt = *(g.vertex[*j].first.begin());
//             std::vector<int> lex = getLexicalOrder(stmt);
//             if (j == s[i+1].begin())
//               min_lex_value = lex[dim-1];
//             else if (lex[dim-1] < min_lex_value)
//               min_lex_value = lex[dim-1];
//           }
//           fused_lex_value = min_lex_value - 1;
//         }
//         else {
//           int max_lex_value;
//           for (std::set<int>::iterator j = s[i-1].begin(); j != s[i-1].end(); j++) {
//             int stmt = *(g.vertex[*j].first.begin());
//             std::vector<int> lex = getLexicalOrder(stmt);
//             if (j == s[i-1].begin())
//               max_lex_value = lex[dim-1];
//             else if (lex[dim-1] > max_lex_value)
//               max_lex_value = lex[dim-1];
//           }
//           fused_lex_value = max_lex_value + 1;
//         }
//       }

//       break;
//     }

//   // sort the newly updated to-be-fused lexicographical order values
//   std::vector<int> ordered_lex_values;
//   for (std::set<int>::iterator i = lex_values.begin(); i != lex_values.end(); i++)
//     ordered_lex_values.push_back(*i);
//   std::sort(ordered_lex_values.begin(), ordered_lex_values.end());

//   // make sure internal loops inside to-be-fused loops have the same
//   // lexicographical order before and after fusion
//   std::vector<std::pair<int, int> > inside_lex_range(ordered_lex_values.size());
//   for (int i = 0; i < ordered_lex_values.size(); i++) {
//     ref_lex[dim-1] = ordered_lex_values[i];
//     std::set<int> the_stmts = getStatements(ref_lex, dim-1);
//     std::set<int>::iterator j = the_stmts.begin();
//     std::vector<int> lex = getLexicalOrder(*j);
//     int min_inside_lex_value = lex[dim+1];
//     int max_inside_lex_value = lex[dim+1];
//     j++;
//     for (; j != the_stmts.end(); j++) {
//       std::vector<int> lex = getLexicalOrder(*j);
//       if (lex[dim+1] < min_inside_lex_value)
//         min_inside_lex_value = lex[dim+1];
//       if (lex[dim+1] > max_inside_lex_value)
//         max_inside_lex_value = lex[dim+1];
//     }
//     inside_lex_range[i].first = min_inside_lex_value;
//     inside_lex_range[i].second = max_inside_lex_value;
//   }
//   for (int i = 1; i < ordered_lex_values.size(); i++)
//     if (inside_lex_range[i].first <= inside_lex_range[i-1].second) {
//       int shift_lex_value = inside_lex_range[i-1].second - inside_lex_range[i].first + 1;
//       ref_lex[dim-1] = ordered_lex_values[i];
//       ref_lex[dim+1] = inside_lex_range[i].first;
//       shiftLexicalOrder(ref_lex, dim+1, shift_lex_value);
//       inside_lex_range[i].first += shift_lex_value;
//       inside_lex_range[i].second += shift_lex_value;
//     }

//   // set lexicographical order for fused loops
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
//     std::vector<int> lex = getLexicalOrder(*i);
//     if (lex_values.find(lex[dim-1]) != lex_values.end())
//       assign_const(stmt[*i].xform, dim-1, fused_lex_value);      
//   }

//   // no need to update dependence graph
//   ;

//   return true;
// }


// bool Loop::distribute(const std::set<int> &stmt_nums, int level) {
//   if (stmt_nums.size() == 0 || stmt_nums.size() == 1)
//     return true;
//   int dim = 2*level-1;

//   // check for sanity of parameters
//   std::vector<int> ref_lex;
//   for (std::set<int>::const_iterator i = stmt_nums.begin(); i != stmt_nums.end(); i++) {
//     if (*i < 0 || *i >= stmt.size())
//       throw std::invalid_argument("invalid statement number " + to_string(*i));
//     if (level < 1 || level > (stmt[*i].xform.n_out()-1)/2)
//       throw std::invalid_argument("invalid loop level " + to_string(level));
//     if (ref_lex.size() == 0)
//       ref_lex = getLexicalOrder(*i);
//     else {
//       std::vector<int> lex = getLexicalOrder(*i);
//       for (int j = 0; j <= dim-1; j+=2)
//         if (lex[j] != ref_lex[j])
//           throw std::invalid_argument("statements for distribution must be in the same level-" + to_string(level) + " subloop");
//     }
//   }

//   // find SCC in the to-be-distributed loop
//   int dep_dim = xform_index[dim].first;
//   std::set<int> same_loop = getStatements(ref_lex, dim-1);
//   Graph<int, Empty> g;
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
//     g.insert(*i);
//   for (int i = 0; i < g.vertex.size(); i++)
//     for (int j = i+1; j < g.vertex.size(); j++) {
//       std::vector<DependenceVector> dvs;
//       dvs = dep.getEdge(g.vertex[i].first, g.vertex[j].first);
//       for (int k = 0; k < dvs.size(); k++)
//         if (dvs[k].isCarried(dep_dim)) {
//           g.connect(i, j);
//           break;
//         }
//       dvs = dep.getEdge(g.vertex[j].first, g.vertex[i].first);
//       for (int k = 0; k < dvs.size(); k++)
//         if (dvs[k].isCarried(dep_dim)) {
//           g.connect(j, i);
//           break;
//         }
//     }
//   std::vector<std::set<int> > s = g.topoSort();

//   // find statements that cannot be distributed due to dependence cycle
//   Graph<std::set<int>, Empty> g2;
//   for (int i = 0; i < s.size(); i++) {
//     std::set<int> t;
//     for (std::set<int>::iterator j = s[i].begin(); j != s[i].end(); j++)
//       if (stmt_nums.find(g.vertex[*j].first) != stmt_nums.end())
//         t.insert(g.vertex[*j].first);
//     if (!t.empty())
//       g2.insert(t);
//   }
//   for (int i = 0; i < g2.vertex.size(); i++)
//     for (int j = i+1; j < g2.vertex.size(); j++)
//       for (std::set<int>::iterator ii = g2.vertex[i].first.begin(); ii != g2.vertex[i].first.end(); ii++)
//         for (std::set<int>::iterator jj = g2.vertex[j].first.begin(); jj != g2.vertex[j].first.end(); jj++) {
//           std::vector<DependenceVector> dvs;
//           dvs = dep.getEdge(*ii, *jj);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim)) {
//               g2.connect(i, j);
//               break;
//             }
//           dvs = dep.getEdge(*jj, *ii);
//           for (int k = 0; k < dvs.size(); k++)
//             if (dvs[k].isCarried(dep_dim)) {
//               g2.connect(j, i);
//               break;
//             }
//         }
//   std::vector<std::set<int> > s2 = g2.topoSort();

//   // nothing to distribute
//   if (s2.size() == 1)
//     throw loop_error("loop error: no statement can be distributed due to dependence cycle");

//   std::vector<std::set<int> > s3;
//   for (int i = 0; i < s2.size(); i++) {
//     std::set<int> t;
//     for (std::set<int>::iterator j = s2[i].begin(); j != s2[i].end(); j++)
//       std::set_union(t.begin(), t.end(), g2.vertex[*j].first.begin(), g2.vertex[*j].first.end(), inserter(t, t.begin()));
//     s3.push_back(t);
//   }

//   // associate other affected statements with the right distributed statements
//   for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++)
//     if (stmt_nums.find(*i) == stmt_nums.end()) {
//       bool is_inserted = false;
//       int potential_insertion_point = 0;
//       for (int j = 0; j < s3.size(); j++) {
//         for (std::set<int>::iterator k = s3[j].begin(); k != s3[j].end(); k++) {
//           std::vector<DependenceVector> dvs;
//           dvs = dep.getEdge(*i, *k);
//           for (int kk = 0; kk < dvs.size(); kk++)
//             if (dvs[kk].isCarried(dep_dim)) {
//               s3[j].insert(*i);
//               is_inserted = true;
//               break;
//             }
//           dvs = dep.getEdge(*k, *i);
//           for (int kk = 0; kk < dvs.size(); kk++)
//             if (dvs[kk].isCarried(dep_dim))
//               potential_insertion_point = j;
//         }
//         if (is_inserted)
//           break;
//       }

//       if (!is_inserted)
//         s3[potential_insertion_point].insert(*i);
//     }

//   // set lexicographical order after distribution
//   int order = ref_lex[dim-1];
//   shiftLexicalOrder(ref_lex, dim-1, s3.size()-1);
//   for (std::vector<std::set<int> >::iterator i = s3.begin(); i != s3.end(); i++) {
//     for (std::set<int>::iterator j = (*i).begin(); j != (*i).end(); j++)
//       assign_const(stmt[*j].xform, dim-1, order);
//     order++;
//   }

//   // no need to update dependence graph
//   ;

//   return true;
// }








