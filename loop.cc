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
#include <codegen.h>
#include <code_gen/CG_utils.h>
#include <iostream>
#include <algorithm>
#include <map>
#include "loop.hh"
#include "omegatools.hh"
#include "irtools.hh"
#include "chill_error.hh"
#include <string.h>
#include <list>
using namespace omega;

const std::string Loop::tmp_loop_var_name_prefix = std::string("chill_t"); // Manu:: In fortran, first character of a variable name must be a letter, so this change
const std::string Loop::overflow_var_name_prefix = std::string("over");

//-----------------------------------------------------------------------------
// Class Loop
//-----------------------------------------------------------------------------
// --begin Anand: Added from CHiLL 0.2

bool Loop::isInitialized() const {
  return stmt.size() != 0 && !stmt[0].xform.is_null();
}

//--end Anand: added from CHiLL 0.2

bool Loop::init_loop(std::vector<ir_tree_node *> &ir_tree,
                     std::vector<ir_tree_node *> &ir_stmt) {

  ir_stmt = extract_ir_stmts(ir_tree);
  stmt_nesting_level_.resize(ir_stmt.size());
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
    stmt_nesting_level_[i] = t;
    stmt_nesting_level[i] = t;
  }
  
  stmt = std::vector<Statement>(ir_stmt.size());
  int n_dim = -1;
  int max_loc;
  //std::vector<std::string> index;
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
      int cur_dim = n_dim - 1;
      while (itn->parent != NULL) {
        itn = itn->parent;
        if (itn->content->type() == IR_CONTROL_LOOP) {
          index[cur_dim] =
            static_cast<IR_Loop *>(itn->content)->index()->name();
          itn->payload = cur_dim--;
        }
      }
    }
    
    // align loops by names, temporary solution
    ir_tree_node *itn = ir_stmt[loc];
    int depth = stmt_nesting_level_[loc] - 1;
    /*   while (itn->parent != NULL) {
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
    */
    for (int t = depth; t >= 0; t--) {
      int y = t;
      ir_tree_node *itn = ir_stmt[loc];
      
      while ((itn->parent != NULL) && (y >= 0)) {
        itn = itn->parent;
        if (itn->content->type() == IR_CONTROL_LOOP)
          y--;
      }
      
      if (itn->content->type() == IR_CONTROL_LOOP && itn->payload == -1) {
        CG_outputBuilder *ocg = ir->builder();
        
        itn->payload = depth - t;
        
        CG_outputRepr *code =
          static_cast<IR_Block *>(ir_stmt[loc]->content)->extract();
        
        std::vector<CG_outputRepr *> index_expr;
        std::vector<std::string> old_index;
        CG_outputRepr *repl = ocg->CreateIdent(index[itn->payload]);
        index_expr.push_back(repl);
        old_index.push_back(
          static_cast<IR_Loop *>(itn->content)->index()->name());
        code = ocg->CreateSubstitutedStmt(0, code, old_index,
                                          index_expr);
        
        replace.insert(std::pair<int, CG_outputRepr*>(loc, code));
        //stmt[loc].code = code;
        
      }
    }
    
    // set relation variable names
    Relation r(n_dim);
    F_And *f_root = r.add_and();
    itn = ir_stmt[loc];
    int temp_depth = depth;
    while (itn->parent != NULL) {
      
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP) {
        r.name_set_var(itn->payload + 1, index[temp_depth]);
        
        temp_depth--;
      }
      //static_cast<IR_Loop *>(itn->content)->index()->name());
    }
    
    /*while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP)
      r.name_set_var(itn->payload+1, static_cast<IR_Loop *>(itn->content)->index()->name());
      }*/
    
    // extract information from loop/if structures
    std::vector<bool> processed(n_dim, false);
    std::vector<std::string> vars_to_be_reversed;
    itn = ir_stmt[loc];
    while (itn->parent != NULL) {
      itn = itn->parent;
      
      switch (itn->content->type()) {
      case IR_CONTROL_LOOP: {
        IR_Loop *lp = static_cast<IR_Loop *>(itn->content);
        Variable_ID v = r.set_var(itn->payload + 1);
        int c;
        
        try {
          c = lp->step_size();
          if (c > 0) {
            CG_outputRepr *lb = lp->lower_bound();
            exp2formula(ir, r, f_root, freevar, lb, v, 's',
                        IR_COND_GE, true);
            CG_outputRepr *ub = lp->upper_bound();
            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_LT || cond == IR_COND_LE)
              exp2formula(ir, r, f_root, freevar, ub, v, 's',
                          cond, true);
            else
              throw ir_error("loop condition not supported");
            
          } else if (c < 0) {
            CG_outputBuilder *ocg = ir->builder();
            CG_outputRepr *lb = lp->lower_bound();
            lb = ocg->CreateMinus(NULL, lb);
            exp2formula(ir, r, f_root, freevar, lb, v, 's',
                        IR_COND_GE, true);
            CG_outputRepr *ub = lp->upper_bound();
            ub = ocg->CreateMinus(NULL, ub);
            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_GE)
              exp2formula(ir, r, f_root, freevar, ub, v, 's',
                          IR_COND_LE, true);
            else if (cond == IR_COND_GT)
              exp2formula(ir, r, f_root, freevar, ub, v, 's',
                          IR_COND_LT, true);
            else
              throw ir_error("loop condition not supported");
            
            vars_to_be_reversed.push_back(lp->index()->name());
          } else
            throw ir_error("loop step size zero");
        } catch (const ir_error &e) {
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
          exp2formula(ir, r, f_and, freevar, lb, e, 's', IR_COND_EQ,
                      true);
        }
        
        processed[itn->payload] = true;
        break;
      }
      case IR_CONTROL_IF: {
        CG_outputRepr *cond =
          static_cast<IR_If *>(itn->content)->condition();
        try {
          if (itn->payload % 2 == 1)
            exp2constraint(ir, r, f_root, freevar, cond, true);
          else {
            F_Not *f_not = f_root->add_not();
            F_And *f_and = f_not->add_and();
            exp2constraint(ir, r, f_and, freevar, cond, true);
          }
        } catch (const ir_error &e) {
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
            } else if ((*t)[i]->payload >> 1 == id >> 1) {
              delete (*t)[i];
              t->erase(t->begin() + i);
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
          if (itn->content->type() == IR_CONTROL_LOOP
              && itn->payload == j)
            break;
        }
        
        Variable_ID v = r.set_var(j + 1);
        if (loc < max_loc) {
          
          CG_outputBuilder *ocg = ir->builder();
          
          CG_outputRepr *lb =
            static_cast<IR_Loop *>(itn->content)->lower_bound();
          
          exp2formula(ir, r, f_root, freevar, lb, v, 's', IR_COND_EQ,
                      false);
          
          /*    if (ir->QueryExpOperation(
                static_cast<IR_Loop *>(itn->content)->lower_bound())
                == IR_OP_VARIABLE) {
                IR_ScalarRef *ref =
                static_cast<IR_ScalarRef *>(ir->Repr2Ref(
                static_cast<IR_Loop *>(itn->content)->lower_bound()));
                std::string name_ = ref->name();
                
                for (int i = 0; i < index.size(); i++)
                if (index[i] == name_) {
                exp2formula(ir, r, f_root, freevar, lb, v, 's',
                IR_COND_GE, false);
                
                CG_outputRepr *ub =
                static_cast<IR_Loop *>(itn->content)->upper_bound();
                IR_CONDITION_TYPE cond =
                static_cast<IR_Loop *>(itn->content)->stop_cond();
                if (cond == IR_COND_LT || cond == IR_COND_LE)
                exp2formula(ir, r, f_root, freevar, ub, v,
                's', cond, false);
                
                
                
                }
                
                }
          */
          
        } else { // loc > max_loc
          
          CG_outputBuilder *ocg = ir->builder();
          CG_outputRepr *ub =
            static_cast<IR_Loop *>(itn->content)->upper_bound();
          
          exp2formula(ir, r, f_root, freevar, ub, v, 's', IR_COND_EQ,
                      false);
          /*if (ir->QueryExpOperation(
            static_cast<IR_Loop *>(itn->content)->upper_bound())
            == IR_OP_VARIABLE) {
            IR_ScalarRef *ref =
            static_cast<IR_ScalarRef *>(ir->Repr2Ref(
            static_cast<IR_Loop *>(itn->content)->upper_bound()));
            std::string name_ = ref->name();
            
            for (int i = 0; i < index.size(); i++)
            if (index[i] == name_) {
            
            CG_outputRepr *lb =
            static_cast<IR_Loop *>(itn->content)->lower_bound();
            
            exp2formula(ir, r, f_root, freevar, lb, v, 's',
            IR_COND_GE, false);
            
            CG_outputRepr *ub =
            static_cast<IR_Loop *>(itn->content)->upper_bound();
            IR_CONDITION_TYPE cond =
            static_cast<IR_Loop *>(itn->content)->stop_cond();
            if (cond == IR_COND_LT || cond == IR_COND_LE)
            exp2formula(ir, r, f_root, freevar, ub, v,
            's', cond, false);
            
            
            }
            }
          */
        }
      }
    
    r.setup_names();
    r.simplify();
    
    // insert the statement
    CG_outputBuilder *ocg = ir->builder();
    std::vector<CG_outputRepr *> reverse_expr;
    for (int j = 1; j <= vars_to_be_reversed.size(); j++) {
      CG_outputRepr *repl = ocg->CreateIdent(vars_to_be_reversed[j]);
      repl = ocg->CreateMinus(NULL, repl);
      reverse_expr.push_back(repl);
    }
    CG_outputRepr *code =
      static_cast<IR_Block *>(ir_stmt[loc]->content)->extract();
    code = ocg->CreateSubstitutedStmt(0, code, vars_to_be_reversed,
                                      reverse_expr);
    stmt[loc].code = code;
    stmt[loc].IS = r;
    stmt[loc].loop_level = std::vector<LoopLevel>(n_dim);
    stmt[loc].ir_stmt_node = ir_stmt[loc];
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

  last_compute_cgr_ = NULL;
  last_compute_cg_ = NULL;
  
  ir = const_cast<IR_Code *>(control->ir_);
  init_code = NULL;
  cleanup_code = NULL;
  tmp_loop_var_name_counter = 1;
  overflow_var_name_counter = 1;
  known = Relation::True(0);
  
  ir_tree = build_ir_tree(control->clone(), NULL);
  //    std::vector<ir_tree_node *> ir_stmt;
  
  while (!init_loop(ir_tree, ir_stmt)) {
  }

  
  
  for (int i = 0; i < stmt.size(); i++) {
    std::map<int, CG_outputRepr*>::iterator it = replace.find(i);
    
    if (it != replace.end())
      stmt[i].code = it->second;
    else
      stmt[i].code = stmt[i].code;
  }
  
  if (stmt.size() != 0)
    dep = DependenceGraph(stmt[0].IS.n_set());
  else
    dep = DependenceGraph(0);
  // init the dependence graph
  for (int i = 0; i < stmt.size(); i++)
    dep.insert();
  
  for (int i = 0; i < stmt.size(); i++)
    for (int j = i; j < stmt.size(); j++) {
      std::pair<std::vector<DependenceVector>,
        std::vector<DependenceVector> > dv = test_data_dependences(
          ir, stmt[i].code, stmt[i].IS, stmt[j].code, stmt[j].IS,
          freevar, index, stmt_nesting_level_[i],
          stmt_nesting_level_[j]);
      
      for (int k = 0; k < dv.first.size(); k++) {
        if (is_dependence_valid(ir_stmt[i], ir_stmt[j], dv.first[k],
                                true))
          dep.connect(i, j, dv.first[k]);
        else {
          dep.connect(j, i, dv.first[k].reverse());
        }
        
      }
      for (int k = 0; k < dv.second.size(); k++)
        if (is_dependence_valid(ir_stmt[j], ir_stmt[i], dv.second[k],
                                false))
          dep.connect(j, i, dv.second[k]);
        else {
          dep.connect(i, j, dv.second[k].reverse());
        }
      // std::pair<std::vector<DependenceVector>,
      //                std::vector<DependenceVector> > dv_ = test_data_dependences(
      
    }
  


  // init dumb transformation relations e.g. [i, j] -> [ 0, i, 0, j, 0]
  for (int i = 0; i < stmt.size(); i++) {
    int n = stmt[i].IS.n_set();
    stmt[i].xform = Relation(n, 2 * n + 1);
    F_And *f_root = stmt[i].xform.add_and();
    
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(stmt[i].xform.output_var(2 * j), 1);
      h.update_coef(stmt[i].xform.input_var(j), -1);
    }
    
    for (int j = 1; j <= 2 * n + 1; j += 2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(stmt[i].xform.output_var(j), 1);
    }
    stmt[i].xform.simplify();
  }
  
  if (stmt.size() != 0)
    num_dep_dim = stmt[0].IS.n_set();
  else
    num_dep_dim = 0;
  // debug
  /*for (int i = 0; i < stmt.size(); i++) {
    std::cout << i << ": ";
    //stmt[i].xform.print();
    stmt[i].IS.print();
    std::cout << std::endl;
    
    }*/
  //end debug
}

Loop::~Loop() {
  
  delete last_compute_cgr_;
  delete last_compute_cg_;
  
  for (int i = 0; i < stmt.size(); i++)
    if (stmt[i].code != NULL) {
      stmt[i].code->clear();
      delete stmt[i].code;
    }
  
  for (int i = 0; i < ir_tree.size(); i++)
    delete ir_tree[i];
  
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
    switch (stmt[stmt_num].loop_level[level - 1].type) {
    case LoopLevelOriginal:
      return stmt[stmt_num].loop_level[level - 1].payload;
    case LoopLevelTile:
      level = stmt[stmt_num].loop_level[level - 1].payload;
      if (level < 1)
        return -1;
      if (level > stmt[stmt_num].loop_level.size())
        throw loop_error(
          "incorrect loop level information for statement "
          + to_string(stmt_num));
      break;
    default:
      throw loop_error(
        "unknown loop level information for statement "
        + to_string(stmt_num));
    }
    trip_count++;
    if (trip_count >= stmt[stmt_num].loop_level.size())
      throw loop_error(
        "incorrect loop level information for statement "
        + to_string(stmt_num));
  }
}

int Loop::get_last_dep_dim_before(int stmt_num, int level) const {
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invaid statement " + to_string(stmt_num));
  
  if (level < 1)
    return -1;
  if (level > stmt[stmt_num].loop_level.size())
    level = stmt[stmt_num].loop_level.size() + 1;
  
  for (int i = level - 1; i >= 1; i--)
    if (stmt[stmt_num].loop_level[i - 1].type == LoopLevelOriginal)
      return stmt[stmt_num].loop_level[i - 1].payload;
  
  return -1;
}

void Loop::print_internal_loop_structure() const {
  for (int i = 0; i < stmt.size(); i++) {
    std::vector<int> lex = getLexicalOrder(i);
    std::cout << "s" << i + 1 << ": ";
    for (int j = 0; j < stmt[i].loop_level.size(); j++) {
      if (2 * j < lex.size())
        std::cout << lex[2 * j];
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
    for (int j = 2 * stmt[i].loop_level.size(); j < lex.size(); j += 2) {
      std::cout << lex[j];
      if (j != lex.size() - 1)
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
  
  if (last_compute_cg_ == NULL) {
    std::vector<Relation> IS(m);
    std::vector<Relation> xforms(m);
    for (int i = 0; i < m; i++) {
      IS[i] = stmt[i].IS;
      xforms[i] = stmt[i].xform;
    }
    Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
    
    last_compute_cg_ = new CodeGen(xforms, IS, known);
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  
  std::vector<CG_outputRepr *> stmts(m);
  for (int i = 0; i < m; i++)
    stmts[i] = stmt[i].code;
  CG_outputBuilder *ocg = ir->builder();
  CG_outputRepr *repr = last_compute_cgr_->printRepr(ocg, stmts);
  
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
  
  if (last_compute_cg_ == NULL) {
    std::vector<Relation> IS(m);
    std::vector<Relation> xforms(m);
    for (int i = 0; i < m; i++) {
      IS[i] = stmt[i].IS;
      xforms[i] = stmt[i].xform;
    }
    Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
    
    last_compute_cg_ = new CodeGen(xforms, IS, known);
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  
  std::string repr = last_compute_cgr_->printString();
  std::cout << repr << std::endl;
}

void Loop::printIterationSpace() const {
  for (int i = 0; i < stmt.size(); i++) {
    std::cout << "s" << i << ": ";
    Relation r = getNewIS(i);
    for (int j = 1; j <= r.n_inp(); j++)
      r.name_input_var(j, CodeGen::loop_var_name_prefix + to_string(j));
    r.setup_names();
    r.print();
  }
}

void Loop::printDependenceGraph() const {
  if (dep.edgeCount() == 0)
    std::cout << "no dependence exists" << std::endl;
  else {
    std::cout << "dependence graph:" << std::endl;
    std::cout << dep;
  }
}

Relation Loop::getNewIS(int stmt_num) const {
  Relation result;
  
  if (stmt[stmt_num].xform.is_null()) {
    Relation known = Extend_Set(copy(this->known),
                                stmt[stmt_num].IS.n_set() - this->known.n_set());
    result = Intersection(copy(stmt[stmt_num].IS), known);
  } else {
    Relation known = Extend_Set(copy(this->known),
                                stmt[stmt_num].xform.n_out() - this->known.n_set());
    result = Intersection(
      Range(
        Restrict_Domain(copy(stmt[stmt_num].xform),
                        copy(stmt[stmt_num].IS))), known);
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

void Loop::pragma(int stmt_num, int level, const std::string &pragmaText) {
	// check sanity of parameters
	if(stmt_num < 0)
		throw std::invalid_argument("invalid statement " + to_string(stmt_num));
	
	CG_outputBuilder *ocg = ir->builder();
	CG_outputRepr *code = stmt[stmt_num].code;
	ocg->CreatePragmaAttribute(code, level, pragmaText);
}
/*
void Loop::prefetch(int stmt_num, int level, const std::string &arrName, const std::string &indexName, int offset, int hint) {
	// check sanity of parameters
	if(stmt_num < 0)
		throw std::invalid_argument("invalid statement " + to_string(stmt_num));

	CG_outputBuilder *ocg = ir->builder();
	CG_outputRepr *code = stmt[stmt_num].code;
	ocg->CreatePrefetchAttribute(code, level, arrName, indexName, int offset, hint);
}
*/

void Loop::prefetch(int stmt_num, int level, const std::string &arrName, int hint) {
	// check sanity of parameters
	if(stmt_num < 0)
		throw std::invalid_argument("invalid statement " + to_string(stmt_num));

	CG_outputBuilder *ocg = ir->builder();
	CG_outputRepr *code = stmt[stmt_num].code;
	ocg->CreatePrefetchAttribute(code, level, arrName, hint);
}

std::vector<int> Loop::getLexicalOrder(int stmt_num) const {
  assert(stmt_num < stmt.size());
  
  const int n = stmt[stmt_num].xform.n_out();
  std::vector<int> lex(n, 0);
  
  for (int i = 0; i < n; i += 2)
    lex[i] = get_const(stmt[stmt_num].xform, i, Output_Var);
  
  return lex;
}

// find the sub loop nest specified by stmt_num and level,
// only iteration space satisfiable statements returned.
std::set<int> Loop::getSubLoopNest(int stmt_num, int level) const {
  assert(stmt_num >= 0 && stmt_num < stmt.size());
  assert(level > 0 && level <= stmt[stmt_num].loop_level.size());
  
  std::set<int> working;
  for (int i = 0; i < stmt.size(); i++)
    if (const_cast<Loop *>(this)->stmt[i].IS.is_upper_bound_satisfiable()
        && stmt[i].loop_level.size() >= level)
      working.insert(i);
  
  for (int i = 1; i <= level; i++) {
    int a = getLexicalOrder(stmt_num, i);
    for (std::set<int>::iterator j = working.begin(); j != working.end();) {
      int b = getLexicalOrder(*j, i);
      if (b != a)
        working.erase(j++);
      else
        ++j;
    }
  }
  
  return working;
}

int Loop::getLexicalOrder(int stmt_num, int level) const {
  assert(stmt_num >= 0 && stmt_num < stmt.size());
  assert(level > 0 && level <= stmt[stmt_num].loop_level.size()+1);
  
  Relation &r = const_cast<Loop *>(this)->stmt[stmt_num].xform;
  for (EQ_Iterator e(r.single_conjunct()->EQs()); e; e++)
    if (abs((*e).get_coef(r.output_var(2 * level - 1))) == 1) {
      bool is_const = true;
      for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
        if (cvi.curr_var() != r.output_var(2 * level - 1)) {
          is_const = false;
          break;
        }
      if (is_const) {
        int t = static_cast<int>((*e).get_const());
        return (*e).get_coef(r.output_var(2 * level - 1)) > 0 ? -t : t;
      }
    }
  
  throw loop_error(
    "can't find lexical order for statement " + to_string(stmt_num)
    + "'s loop level " + to_string(level));
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
      for (j = 0; j <= dim; j += 2)
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
    } else if (amount < 0) {
      if (lex2[dim] > lex[dim])
        continue;
    }
    
    assign_const(stmt[i].xform, dim, lex2[dim] + amount);
  }
}

std::vector<std::set<int> > Loop::sort_by_same_loops(std::set<int> active,
                                                     int level) {
  
  std::set<int> not_nested_at_this_level;
  std::map<ir_tree_node*, std::set<int> > sorted_by_loop;
  std::map<int, std::set<int> > sorted_by_lex_order;
  std::vector<std::set<int> > to_return;
  bool lex_order_already_set = false;
  for (std::set<int>::iterator it = active.begin(); it != active.end();
       it++) {
    
    if (stmt[*it].ir_stmt_node == NULL)
      lex_order_already_set = true;
  }
  
  if (lex_order_already_set) {
    
    for (std::set<int>::iterator it = active.begin(); it != active.end();
         it++) {
      std::map<int, std::set<int> >::iterator it2 =
        sorted_by_lex_order.find(
          get_const(stmt[*it].xform, 2 * (level - 1),
                    Output_Var));
      
      if (it2 != sorted_by_lex_order.end())
        it2->second.insert(*it);
      else {
        
        std::set<int> to_insert;
        
        to_insert.insert(*it);
        
        sorted_by_lex_order.insert(
          std::pair<int, std::set<int> >(
            get_const(stmt[*it].xform, 2 * (level - 1),
                      Output_Var), to_insert));
        
      }
      
    }
    
    for (std::map<int, std::set<int> >::iterator it2 =
           sorted_by_lex_order.begin(); it2 != sorted_by_lex_order.end();
         it2++)
      to_return.push_back(it2->second);
    
  } else {
    
    for (std::set<int>::iterator it = active.begin(); it != active.end();
         it++) {
      
      ir_tree_node* itn = stmt[*it].ir_stmt_node;
      itn = itn->parent;
      while ((itn != NULL) && (itn->payload != level - 1))
        itn = itn->parent;
      
      if (itn == NULL)
        not_nested_at_this_level.insert(*it);
      else {
        std::map<ir_tree_node*, std::set<int> >::iterator it2 =
          sorted_by_loop.find(itn);
        
        if (it2 != sorted_by_loop.end())
          it2->second.insert(*it);
        else {
          std::set<int> to_insert;
          
          to_insert.insert(*it);
          
          sorted_by_loop.insert(
            std::pair<ir_tree_node*, std::set<int> >(itn,
                                                     to_insert));
          
        }
        
      }
      
    }
    if (not_nested_at_this_level.size() > 0) {
      for (std::set<int>::iterator it = not_nested_at_this_level.begin();
           it != not_nested_at_this_level.end(); it++) {
        std::set<int> temp;
        temp.insert(*it);
        to_return.push_back(temp);
        
      }
    }
    for (std::map<ir_tree_node*, std::set<int> >::iterator it2 =
           sorted_by_loop.begin(); it2 != sorted_by_loop.end(); it2++)
      to_return.push_back(it2->second);
  }
  return to_return;
}

void update_successors(int n, int node_num[], int cant_fuse_with[],
                       Graph<std::set<int>, bool> &g, std::list<int> &work_list) {
  
  std::set<int> disconnect;
  for (Graph<std::set<int>, bool>::EdgeList::iterator i =
         g.vertex[n].second.begin(); i != g.vertex[n].second.end(); i++) {
    int m = i->first;
    
    if (node_num[m] != -1)
      throw loop_error("Graph input for fusion has cycles not a DAG!!");
    
    std::vector<bool> check_ = g.getEdge(n, m);
    
    bool has_bad_edge_path = false;
    for (int i = 0; i < check_.size(); i++)
      if (!check_[i]) {
        has_bad_edge_path = true;
        break;
      }
    if (has_bad_edge_path)
      cant_fuse_with[m] = std::max(cant_fuse_with[m], node_num[n]);
    else
      cant_fuse_with[m] = std::max(cant_fuse_with[m], cant_fuse_with[n]);
    disconnect.insert(m);
  }
  
  
  for (std::set<int>::iterator i = disconnect.begin(); i != disconnect.end();
       i++) {
    g.disconnect(n, *i);
    
    bool no_incoming_edges = true;
    for (int j = 0; j < g.vertex.size(); j++)
      if (j != *i)
        if (g.hasEdge(j, *i)) {
          no_incoming_edges = false;
          break;
        }
    
    
    if (no_incoming_edges)
      work_list.push_back(*i);
  }
  
}

Graph<std::set<int>, bool> Loop::construct_induced_graph_at_level(
  std::vector<std::set<int> > s, DependenceGraph dep, int dep_dim) {
  Graph<std::set<int>, bool> g;
  
  for (int i = 0; i < s.size(); i++)
    g.insert(s[i]);
  
  for (int i = 0; i < s.size(); i++) {
    
    for (int j = i + 1; j < s.size(); j++) {
      bool has_true_edge_i_to_j = false;
      bool has_true_edge_j_to_i = false;
      bool is_connected_i_to_j = false;
      bool is_connected_j_to_i = false;
      for (std::set<int>::iterator ii = s[i].begin(); ii != s[i].end();
           ii++) {
        
        for (std::set<int>::iterator jj = s[j].begin();
             jj != s[j].end(); jj++) {
          
          std::vector<DependenceVector> dvs = dep.getEdge(*ii, *jj);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].is_control_dependence()
                || (dvs[k].is_data_dependence()
                    && dvs[k].has_been_carried_at(dep_dim))) {
              
              if (dvs[k].is_data_dependence()
                  && dvs[k].has_negative_been_carried_at(
                    dep_dim)) {
                //g.connect(i, j, false);
                is_connected_i_to_j = true;
                break;
              } else {
                //g.connect(i, j, true);
                
                has_true_edge_i_to_j = true;
                //break
              }
            }
          
          //if (is_connected)
          
          //    break;
          //        if (has_true_edge_i_to_j && !is_connected_i_to_j)
          //                g.connect(i, j, true);
          dvs = dep.getEdge(*jj, *ii);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].is_control_dependence()
                || (dvs[k].is_data_dependence()
                    && dvs[k].has_been_carried_at(dep_dim))) {
              
              if (is_connected_i_to_j || has_true_edge_i_to_j)
                throw loop_error(
                  "Graph input for fusion has cycles not a DAG!!");
              
              if (dvs[k].is_data_dependence()
                  && dvs[k].has_negative_been_carried_at(
                    dep_dim)) {
                //g.connect(i, j, false);
                is_connected_j_to_i = true;
                break;
              } else {
                //g.connect(i, j, true);
                
                has_true_edge_j_to_i = true;
                //break;
              }
            }
          
          //    if (is_connected)
          //break;
          //    if (is_connected)
          //break;
        }
        
        
        //if (is_connected)
        //  break;
      }
      
      
      if (is_connected_i_to_j)
        g.connect(i, j, false);
      else if (has_true_edge_i_to_j)
        g.connect(i, j, true);
      
      if (is_connected_j_to_i)
        g.connect(j, i, false);
      else if (has_true_edge_j_to_i)
        g.connect(j, i, true);
      
      
    }
  }
  return g;
}

std::vector<std::set<int> > Loop::typed_fusion(Graph<std::set<int>, bool> g) {
  
  bool roots[g.vertex.size()];
  
  for (int i = 0; i < g.vertex.size(); i++)
    roots[i] = true;
  
  for (int i = 0; i < g.vertex.size(); i++)
    for (int j = i + 1; j < g.vertex.size(); j++) {
      
      if (g.hasEdge(i, j))
        roots[j] = false;
      
      if (g.hasEdge(j, i))
        roots[i] = false;
      
    }
  
  std::list<int> work_list;
  int cant_fuse_with[g.vertex.size()];
  std::vector<std::set<int> > s;
  //Each Fused set's representative node
  
  int node_to_fused_nodes[g.vertex.size()];
  int node_num[g.vertex.size()];
  for (int i = 0; i < g.vertex.size(); i++) {
    if (roots[i] == true)
      work_list.push_back(i);
    cant_fuse_with[i] = 0;
    node_to_fused_nodes[i] = 0;
    node_num[i] = -1;
  }
  // topological sort according to chun's permute algorithm
  //   std::vector<std::set<int> > s = g.topoSort();
  std::vector<std::set<int> > s2 = g.topoSort();
  if (work_list.empty() || (s2.size() != g.vertex.size())) {
    
    std::cout << s2.size() << "\t" << g.vertex.size() << std::endl;
    throw loop_error("Input for fusion not a DAG!!");
    
    
  }
  int fused_nodes_counter = 0;
  while (!work_list.empty()) {
    int n = work_list.front();
    //int n_ = g.vertex[n].first;
    work_list.pop_front();
    int node;
    if (cant_fuse_with[n] == 0)
      node = 0;
    else
      node = cant_fuse_with[n];
    
    if ((fused_nodes_counter != 0) && (node != fused_nodes_counter)) {
      int rep_node = node_to_fused_nodes[node];
      node_num[n] = node_num[rep_node];
      
      try {
        update_successors(n, node_num, cant_fuse_with, g, work_list);
      } catch (const loop_error &e) {
        
        throw loop_error(
          "statements cannot be fused together due to negative dependence");
        
        
      }
      for (std::set<int>::iterator it = g.vertex[n].first.begin();
           it != g.vertex[n].first.end(); it++)
        s[node].insert(*it);
    } else {
      //std::set<int> new_node;
      //new_node.insert(n_);
      s.push_back(g.vertex[n].first);
      node_to_fused_nodes[node] = n;
      node_num[n] = ++node;
      try {
        update_successors(n, node_num, cant_fuse_with, g, work_list);
      } catch (const loop_error &e) {
        
        throw loop_error(
          "statements cannot be fused together due to negative dependence");
        
        
      }
      fused_nodes_counter++;
    }
  }
  
  return s;
}

void Loop::setLexicalOrder(int dim, const std::set<int> &active,
                           int starting_order, std::vector<std::vector<std::string> > idxNames) {
  if (active.size() == 0)
    return;
  
  // check for sanity of parameters
  if (dim < 0 || dim % 2 != 0)
    throw std::invalid_argument(
      "invalid constant loop level to set lexicographical order");
  std::vector<int> lex;
  int ref_stmt_num;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if ((*i) < 0 || (*i) >= stmt.size())
      throw std::invalid_argument(
        "invalid statement number " + to_string(*i));
    if (dim >= stmt[*i].xform.n_out())
      throw std::invalid_argument(
        "invalid constant loop level to set lexicographical order");
    if (i == active.begin()) {
      lex = getLexicalOrder(*i);
      ref_stmt_num = *i;
    } else {
      std::vector<int> lex2 = getLexicalOrder(*i);
      for (int j = 0; j < dim; j += 2)
        if (lex[j] != lex2[j])
          throw std::invalid_argument(
            "statements are not in the same sub loop nest");
    }
  }
  
  // sepearate statements by current loop level types
  int level = (dim + 2) / 2;
  std::map<std::pair<LoopLevelType, int>, std::set<int> > active_by_level_type;
  std::set<int> active_by_no_level;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if (level > stmt[*i].loop_level.size())
      active_by_no_level.insert(*i);
    else
      active_by_level_type[std::make_pair(
          stmt[*i].loop_level[level - 1].type,
          stmt[*i].loop_level[level - 1].payload)].insert(*i);
  }
  
  // further separate statements due to control dependences
  std::vector<std::set<int> > active_by_level_type_splitted;
  for (std::map<std::pair<LoopLevelType, int>, std::set<int> >::iterator i =
         active_by_level_type.begin(); i != active_by_level_type.end(); i++)
    active_by_level_type_splitted.push_back(i->second);
  for (std::set<int>::iterator i = active_by_no_level.begin();
       i != active_by_no_level.end(); i++)
    for (int j = active_by_level_type_splitted.size() - 1; j >= 0; j--) {
      std::set<int> controlled, not_controlled;
      for (std::set<int>::iterator k =
             active_by_level_type_splitted[j].begin();
           k != active_by_level_type_splitted[j].end(); k++) {
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
        active_by_level_type_splitted.erase(
          active_by_level_type_splitted.begin() + j);
        active_by_level_type_splitted.push_back(controlled);
        active_by_level_type_splitted.push_back(not_controlled);
      }
    }
  
  // set lexical order separating loops with different loop types first
  if (active_by_level_type_splitted.size() + active_by_no_level.size() > 1) {
    int dep_dim = get_last_dep_dim_before(ref_stmt_num, level) + 1;
    
    Graph<std::set<int>, Empty> g;
    for (std::vector<std::set<int> >::iterator i =
           active_by_level_type_splitted.begin();
         i != active_by_level_type_splitted.end(); i++)
      g.insert(*i);
    for (std::set<int>::iterator i = active_by_no_level.begin();
         i != active_by_no_level.end(); i++) {
      std::set<int> t;
      t.insert(*i);
      g.insert(t);
    }
    for (int i = 0; i < g.vertex.size(); i++)
      for (int j = i + 1; j < g.vertex.size(); j++) {
        bool connected = false;
        for (std::set<int>::iterator ii = g.vertex[i].first.begin();
             ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin();
               jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*ii,
                                                            *jj);
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence()
                  || (dvs[k].is_data_dependence()
                      && !dvs[k].has_been_carried_before(
                        dep_dim))) {
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
        for (std::set<int>::iterator ii = g.vertex[i].first.begin();
             ii != g.vertex[i].first.end(); ii++) {
          for (std::set<int>::iterator jj = g.vertex[j].first.begin();
               jj != g.vertex[j].first.end(); jj++) {
            std::vector<DependenceVector> dvs = dep.getEdge(*jj,
                                                            *ii);
            // find the sub loop nest specified by stmt_num and level,
            // only iteration space satisfiable statements returned.
            for (int k = 0; k < dvs.size(); k++)
              if (dvs[k].is_control_dependence()
                  || (dvs[k].is_data_dependence()
                      && !dvs[k].has_been_carried_before(
                        dep_dim))) {
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
      throw loop_error(
        "cannot separate statements with different loop types at loop level "
        + to_string(level));
    
    // assign lexical order
    int order = starting_order;
    for (int i = 0; i < s.size(); i++) {
      std::set<int> &cur_scc = g.vertex[*(s[i].begin())].first;
      int sz = cur_scc.size();
      if (sz == 1) {
        int cur_stmt = *(cur_scc.begin());
        assign_const(stmt[cur_stmt].xform, dim, order);
        for (int j = dim + 2; j < stmt[cur_stmt].xform.n_out(); j += 2)
          assign_const(stmt[cur_stmt].xform, j, 0);
        order++;
      } else {
        setLexicalOrder(dim, cur_scc, order, idxNames);
        order += sz;
      }
    }
  }
  // set lexical order seperating single iteration statements and loops
  else {
    std::set<int> true_singles;
    std::set<int> nonsingles;
    std::map<coef_t, std::set<int> > fake_singles;
    std::set<int> fake_singles_;
    
    // sort out statements that do not require loops
    for (std::set<int>::iterator i = active.begin(); i != active.end();
         i++) {
      Relation cur_IS = getNewIS(*i);
      if (is_single_iteration(cur_IS, dim + 1)) {
        bool is_all_single = true;
        for (int j = dim + 3; j < stmt[*i].xform.n_out(); j += 2)
          if (!is_single_iteration(cur_IS, j)) {
            is_all_single = false;
            break;
          }
        if (is_all_single)
          true_singles.insert(*i);
        else {
          fake_singles_.insert(*i);
          try {
            fake_singles[get_const(cur_IS, dim + 1, Set_Var)].insert(
              *i);
          } catch (const std::exception &e) {
            fake_singles[posInfinity].insert(*i);
          }
        }
      } else
        nonsingles.insert(*i);
    }
    
    
    // split nonsingles forcibly according to negative dependences present (loop unfusible)
    int dep_dim = get_dep_dim_of(ref_stmt_num, level);
    
    if (dim < stmt[ref_stmt_num].xform.n_out() - 1) {
      
      bool dummy_level_found = false;
      
      std::vector<std::set<int> > s;
      
      s = sort_by_same_loops(active, level);
      bool further_levels_exist = false;
      
      if (!idxNames.empty())
        if (level <= idxNames[ref_stmt_num].size())
          if (idxNames[ref_stmt_num][level - 1].length() == 0) {
            //  && s.size() == 1) {
            int order1 = 0;
            dummy_level_found = true;
            
            for (int i = level; i < idxNames[ref_stmt_num].size();
                 i++)
              if (idxNames[ref_stmt_num][i].length() > 0)
                further_levels_exist = true;
            
          }
      
      //if (!dummy_level_found) {
      
      if (s.size() > 1) {
        
        Graph<std::set<int>, bool> g = construct_induced_graph_at_level(
          s, dep, dep_dim);
        s = typed_fusion(g);
      }
      int order = 0;
      for (int i = 0; i < s.size(); i++) {
        
        for (std::set<int>::iterator it = s[i].begin();
             it != s[i].end(); it++)
          assign_const(stmt[*it].xform, dim, order);
        
        if ((dim + 2) <= (stmt[ref_stmt_num].xform.n_out() - 1))
          setLexicalOrder(dim + 2, s[i], order, idxNames);
        
        order++;
      }
      //}
      /*    else {
            
            int order1 = 0;
            int order = 0;
            for (std::set<int>::iterator i = active.begin();
            i != active.end(); i++) {
            if (!further_levels_exist)
            assign_const(stmt[*i].xform, dim, order1++);
            else
            assign_const(stmt[*i].xform, dim, order1);
            
            }
            
            if ((dim + 2) <= (stmt[ref_stmt_num].xform.n_out() - 1) && further_levels_exist)
            setLexicalOrder(dim + 2, active, order, idxNames);
            }
      */
    } else {
      int dummy_order = 0;
      for (std::set<int>::iterator i = active.begin(); i != active.end();
           i++)
        assign_const(stmt[*i].xform, dim, dummy_order++);
    }
    /*for (int i = 0; i < g2.vertex.size(); i++)
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
    */
    //convert to dependence graph for grouped statements
    //dep_dim = get_last_dep_dim_before(ref_stmt_num, level) + 1;
    /*int order = 0;
      for (std::set<int>::iterator j = active.begin(); j != active.end();
      j++) {
      std::set<int> continuous;
      std::cout<< active.size()<<std::endl;
      while (nonsingles.find(*j) != nonsingles.end() && j != active.end()) {
      continuous.insert(*j);
      j++;
      }
      
      printf("continuous size is %d\n", continuous.size());
      
      
      
      if (continuous.size() > 0) {
      std::vector<std::set<int> > s = typed_fusion(continuous, dep,
      dep_dim);
      
      for (int i = 0; i < s.size(); i++) {
      for (std::set<int>::iterator l = s[i].begin();
      l != s[i].end(); l++) {
      assign_const(stmt[*l].xform, dim + 2, order);
      setLexicalOrder(dim + 2, s[i]);
      }
      order++;
      }
      }
      
      if (j != active.end()) {
      assign_const(stmt[*j].xform, dim + 2, order);
      
      for (int k = dim + 4; k < stmt[*j].xform.n_out(); k += 2)
      assign_const(stmt[*j].xform, k, 0);
      order++;
      }
      
      if( j == active.end())
      break;
      }
    */
    
    
    // assign lexical order
    /*int order = starting_order;
      for (int i = 0; i < s.size(); i++) {
      // translate each SCC into original statements
      std::set<int> cur_scc;
      for (std::set<int>::iterator j = s[i].begin(); j != s[i].end(); j++)
      copy(s[i].begin(), s[i].end(),
      inserter(cur_scc, cur_scc.begin()));
      
      // now assign the constant
      for (std::set<int>::iterator j = cur_scc.begin();
      j != cur_scc.end(); j++)
      assign_const(stmt[*j].xform, dim, order);
      
      if (cur_scc.size() > 1)
      setLexicalOrder(dim + 2, cur_scc);
      else if (cur_scc.size() == 1) {
      int cur_stmt = *(cur_scc.begin());
      for (int j = dim + 2; j < stmt[cur_stmt].xform.n_out(); j += 2)
      assign_const(stmt[cur_stmt].xform, j, 0);
      }
      
      if (cur_scc.size() > 0)
      order++;
      }
    */
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
    
    Relation mapping(2 * n + 1, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(2 * j), -1);
    }
    mapping = Composition(mapping, stmt[*i].xform);
    mapping.simplify();
    
    // match omega input/output variables to variable names in the code
    for (int j = 1; j <= stmt[*i].IS.n_set(); j++)
      mapping.name_input_var(j, stmt[*i].IS.set_var(j)->name());
    for (int j = 1; j <= n; j++)
      mapping.name_output_var(j,
                              tmp_loop_var_name_prefix
                              + to_string(tmp_loop_var_name_counter + j - 1));
    mapping.setup_names();
    
    Relation known = Extend_Set(copy(this->known),
                                mapping.n_out() - this->known.n_set());
    //stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out(), NULL));
    std::vector<std::string> loop_vars;
    for (int j = 1; j <= stmt[*i].IS.n_set(); j++)
      loop_vars.push_back(stmt[*i].IS.set_var(j)->name());
    std::vector<CG_outputRepr *> subs = output_substitutions(ocg,
                                                             Inverse(copy(mapping)),
                                                             std::vector<std::pair<CG_outputRepr *, int> >(mapping.n_out(),
                                                                                                           std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)));
    stmt[*i].code = ocg->CreateSubstitutedStmt(0, stmt[*i].code, loop_vars,
                                               subs);
    stmt[*i].IS = Range(Restrict_Domain(mapping, stmt[*i].IS));
    stmt[*i].IS.simplify();
    
    // replace original transformation relation with straight 1-1 mapping
    mapping = Relation(n, 2 * n + 1);
    f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = 1; j <= 2 * n + 1; j += 2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_const(-lex[j - 1]);
    }
    stmt[*i].xform = mapping;
  }
  
  tmp_loop_var_name_counter += max_n;
}

void Loop::addKnown(const Relation &cond) {
  
  // invalidate saved codegen computation
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  
  int n1 = this->known.n_set();
  
  Relation r = copy(cond);
  int n2 = r.n_set();
  
  if (n1 < n2)
    this->known = Extend_Set(this->known, n2 - n1);
  else if (n1 > n2)
    r = Extend_Set(r, n1 - n2);
  
  this->known = Intersection(this->known, r);
}

void Loop::removeDependence(int stmt_num_from, int stmt_num_to) {
  // check for sanity of parameters
  if (stmt_num_from >= stmt.size())
    throw std::invalid_argument(
      "invalid statement number " + to_string(stmt_num_from));
  if (stmt_num_to >= stmt.size())
    throw std::invalid_argument(
      "invalid statement number " + to_string(stmt_num_to));
  
  dep.disconnect(stmt_num_from, stmt_num_to);
}

void Loop::dump() const {
  for (int i = 0; i < stmt.size(); i++) {
    std::vector<int> lex = getLexicalOrder(i);
    std::cout << "s" << i + 1 << ": ";
    for (int j = 0; j < stmt[i].loop_level.size(); j++) {
      if (2 * j < lex.size())
        std::cout << lex[2 * j];
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
    for (int j = 2 * stmt[i].loop_level.size(); j < lex.size(); j += 2) {
      std::cout << lex[j];
      if (j != lex.size() - 1)
        std::cout << ' ';
    }
    std::cout << std::endl;
  }
}

bool Loop::nonsingular(const std::vector<std::vector<int> > &T) {
  if (stmt.size() == 0)
    return true;
  
  // check for sanity of parameters
  for (int i = 0; i < stmt.size(); i++) {
    if (stmt[i].loop_level.size() != num_dep_dim)
      throw std::invalid_argument(
        "nonsingular loop transformations must be applied to original perfect loop nest");
    for (int j = 0; j < stmt[i].loop_level.size(); j++)
      if (stmt[i].loop_level[j].type != LoopLevelOriginal)
        throw std::invalid_argument(
          "nonsingular loop transformations must be applied to original perfect loop nest");
  }
  if (T.size() != num_dep_dim)
    throw std::invalid_argument("invalid transformation matrix");
  for (int i = 0; i < stmt.size(); i++)
    if (T[i].size() != num_dep_dim + 1 && T[i].size() != num_dep_dim)
      throw std::invalid_argument("invalid transformation matrix");
  // invalidate saved codegen computation
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  // build relation from matrix
  Relation mapping(2 * num_dep_dim + 1, 2 * num_dep_dim + 1);
  F_And *f_root = mapping.add_and();
  for (int i = 0; i < num_dep_dim; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(2 * (i + 1)), -1);
    for (int j = 0; j < num_dep_dim; j++)
      if (T[i][j] != 0)
        h.update_coef(mapping.input_var(2 * (j + 1)), T[i][j]);
    if (T[i].size() == num_dep_dim + 1)
      h.update_const(T[i][num_dep_dim]);
  }
  for (int i = 1; i <= 2 * num_dep_dim + 1; i += 2) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(i), -1);
    h.update_coef(mapping.input_var(i), 1);
  }
  
  // update transformation relations
  for (int i = 0; i < stmt.size(); i++)
    stmt[i].xform = Composition(copy(mapping), stmt[i].xform);
  
  // update dependence graph
  for (int i = 0; i < dep.vertex.size(); i++)
    for (DependenceGraph::EdgeList::iterator j =
           dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();
         j++) {
      std::vector<DependenceVector> dvs = j->second;
      for (int k = 0; k < dvs.size(); k++) {
        DependenceVector &dv = dvs[k];
        switch (dv.type) {
        case DEP_W2R:
        case DEP_R2W:
        case DEP_W2W:
        case DEP_R2R: {
          std::vector<coef_t> lbounds(num_dep_dim), ubounds(
            num_dep_dim);
          for (int p = 0; p < num_dep_dim; p++) {
            coef_t lb = 0;
            coef_t ub = 0;
            for (int q = 0; q < num_dep_dim; q++) {
              if (T[p][q] > 0) {
                if (lb == -posInfinity
                    || dv.lbounds[q] == -posInfinity)
                  lb = -posInfinity;
                else
                  lb += T[p][q] * dv.lbounds[q];
                if (ub == posInfinity
                    || dv.ubounds[q] == posInfinity)
                  ub = posInfinity;
                else
                  ub += T[p][q] * dv.ubounds[q];
              } else if (T[p][q] < 0) {
                if (lb == -posInfinity
                    || dv.ubounds[q] == posInfinity)
                  lb = -posInfinity;
                else
                  lb += T[p][q] * dv.ubounds[q];
                if (ub == posInfinity
                    || dv.lbounds[q] == -posInfinity)
                  ub = posInfinity;
                else
                  ub += T[p][q] * dv.lbounds[q];
              }
            }
            if (T[p].size() == num_dep_dim + 1) {
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


bool Loop::is_dependence_valid_based_on_lex_order(int i, int j,
                                                  const DependenceVector &dv, bool before) {
  std::vector<int> lex_i = getLexicalOrder(i);
  std::vector<int> lex_j = getLexicalOrder(j);
  int last_dim;
  if (!dv.is_scalar_dependence) {
    for (last_dim = 0;
         last_dim < lex_i.size() && (lex_i[last_dim] == lex_j[last_dim]);
         last_dim++)
      ;
    last_dim = last_dim / 2;
    if (last_dim == 0)
      return true;
    
    for (int i = 0; i < last_dim; i++) {
      if (dv.lbounds[i] > 0)
        return true;
      else if (dv.lbounds[i] < 0)
        return false;
    }
  }
  if (before)
    return true;
  
  return false;
  
}

