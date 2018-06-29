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
#include <omega/code_gen/include/codegen.h>
#include <code_gen/CG_utils.h>
#include <code_gen/CG_chillBuilder.h> // Manu   bad idea.  TODO
#include <code_gen/CG_stringRepr.h>
#include <code_gen/CG_chillRepr.h>   // Mark.  Bad idea.  TODO 
#include <iostream>
#include <algorithm>
#include <map>
#include "loop.hh"
#include "omegatools.hh"
#include "irtools.hh"
#include "chill_error.hh"
#include <string.h>
#include <list>

// TODO 
#define _DEBUG_ true



using namespace omega;

const std::string Loop::tmp_loop_var_name_prefix = std::string("chill_t"); // Manu:: In fortran, first character of a variable name must be a letter, so this change
const std::string Loop::overflow_var_name_prefix = std::string("over");

void echocontroltype( const IR_Control *control ) { 
  switch(control->type()) { 
  case IR_CONTROL_BLOCK: {
    debug_fprintf(stderr, "IR_CONTROL_BLOCK\n"); 
    break;
  }
  case IR_CONTROL_LOOP: {
    debug_fprintf(stderr, "IR_CONTROL_LOOP\n"); 
    break;
  }
  case IR_CONTROL_IF: {
    debug_fprintf(stderr, "IR_CONTROL_IF\n"); 
    break;
  }
  default:
    debug_fprintf(stderr, "just a bunch of statements?\n"); 
    
  } // switch
}

omega::Relation Loop::getNewIS(int stmt_num) const {
  
  omega::Relation result;
  
  if (stmt[stmt_num].xform.is_null()) {
    omega::Relation known = omega::Extend_Set(omega::copy(this->known),
                                              stmt[stmt_num].IS.n_set() - this->known.n_set());
    result = omega::Intersection(omega::copy(stmt[stmt_num].IS), known);
  } else {
    omega::Relation known = omega::Extend_Set(omega::copy(this->known),
                                              stmt[stmt_num].xform.n_out() - this->known.n_set());
    result = omega::Intersection(
                                 omega::Range(
                                              omega::Restrict_Domain(
                                                                     omega::copy(stmt[stmt_num].xform),
                                                                     omega::copy(stmt[stmt_num].IS))), known);
  }
  result.simplify(2, 4);
  return result;
}



void Loop::reduce(int stmt_num, 
                  std::vector<int> &level, 
                  int param,
                  std::string func_name, 
                  std::vector<int> &seq_levels,
                  std::vector<int> cudaized_levels, 
                  int bound_level) {

  // illegal instruction?? debug_fprintf(stderr, " Loop::reduce( stmt %d, param %d, func_name (encrypted)...)\n", stmt, param); // , func_name.c_str()); 
  
  //std::cout << "Reducing stmt# " << stmt_num << " at level " << level << "\n";
  //ir->printStmt(stmt[stmt_num].code);
  
  if (stmt[stmt_num].reduction != 1) {
    std::cout << "loop.cc Cannot reduce this statement\n";
    return;
  }
  debug_fprintf(stderr, "loop.cc CAN reduce this statment?\n"); 

  /*for (int i = 0; i < level.size(); i++)
    if (stmt[stmt_num].loop_level[level[i] - 1].segreducible != true) {
    std::cout << "Cannot reduce this statement\n";
    return;
    }
    for (int i = 0; i < seq_levels.size(); i++)
    if (stmt[stmt_num].loop_level[seq_levels[i] - 1].segreducible != true) {
    std::cout << "Cannot reduce this statement\n";
    return;
    }
  */
  //  std::pair<int, std::string> to_insert(level, func_name);
  //  reduced_statements.insert(std::pair<int, std::pair<int, std::string> >(stmt_num, to_insert ));
  // invalidate saved codegen computation
  this->invalidateCodeGen();
  debug_fprintf(stderr, "set last_compute_cg_ = NULL;\n");
  
  omega::CG_outputBuilder *ocg = ir->builder();
  
  omega::CG_outputRepr *funCallRepr;
  std::vector<omega::CG_outputRepr *> arg_repr_list;
  apply_xform(stmt_num);
  std::vector<IR_ArrayRef *> access = ir->FindArrayRef(stmt[stmt_num].code);
  std::set<std::string> names;
  for (int i = 0; i < access.size(); i++) {
    std::vector<IR_ArrayRef *> access2;
    for (int j = 0; j < access[i]->n_dim(); j++) {
      std::vector<IR_ArrayRef *> access3 = ir->FindArrayRef(
                                                            access[i]->index(j));
      access2.insert(access2.end(), access3.begin(), access3.end());
    }
    if (access2.size() == 0) {
      if (names.find(access[i]->name()) == names.end()) {
        arg_repr_list.push_back(
                                ocg->CreateAddressOf(access[i]->convert()));
        names.insert(access[i]->name());
        if (access[i]->is_write())
          reduced_write_refs.insert(access[i]->name());
      }
    } else {
      if (names.find(access[i]->name()) == names.end()) {
        arg_repr_list.push_back(ocg->CreateAddressOf(ocg->CreateArrayRefExpression(ocg->CreateIdent(access[i]->name()),
                                                                                   ocg->CreateInt(0))));
        names.insert(access[i]->name());
        if (access[i]->is_write())
          reduced_write_refs.insert(access[i]->name());
      }
    }
  }
  
  for (int i = 0; i < seq_levels.size(); i++)
    arg_repr_list.push_back(
                            ocg->CreateIdent(
                                             stmt[stmt_num].IS.set_var(seq_levels[i])->name()));
  
  if (bound_level != -1) {
    
    omega::Relation new_IS = copy(stmt[stmt_num].IS);
    new_IS.copy_names(stmt[stmt_num].IS);
    new_IS.setup_names();
    new_IS.simplify();
    int dim = bound_level;
    //omega::Relation r = getNewIS(stmt_num);
    for (int j = dim + 1; j <= new_IS.n_set(); j++)
      new_IS = omega::Project(new_IS, new_IS.set_var(j));
    
    new_IS.simplify(2, 4);
    
    omega::Relation bound_ = get_loop_bound(copy(new_IS), dim - 1);
    omega::Variable_ID v = bound_.set_var(dim);
    std::vector<omega::CG_outputRepr *> ubList;
    for (omega::GEQ_Iterator e(
                               const_cast<omega::Relation &>(bound_).single_conjunct()->GEQs());
         e; e++) {
      if ((*e).get_coef(v) < 0) {
        //  && (*e).is_const_except_for_global(v))
        omega::CG_outputRepr *UPPERBOUND =
          omega::output_upper_bound_repr(ir->builder(), *e, v,
                                         bound_,
                                         std::vector<
                                           std::pair<omega::CG_outputRepr *, int> >(
                                                                                    bound_.n_set(),
                                                                                    std::make_pair(
                                                                                                   static_cast<omega::CG_outputRepr *>(NULL),
                                                                                                   0)), uninterpreted_symbols[stmt_num]);
        if (UPPERBOUND != NULL)
          ubList.push_back(UPPERBOUND);
        
      }
      
    }
    
    omega::CG_outputRepr * ubRepr;
    if (ubList.size() > 1) {
      
      ubRepr = ir->builder()->CreateInvoke("min", ubList);
      arg_repr_list.push_back(ubRepr);
    } else if (ubList.size() == 1)
      arg_repr_list.push_back(ubList[0]);
  }
  
  funCallRepr = ocg->CreateInvoke(func_name, arg_repr_list);
  stmt[stmt_num].code = funCallRepr;
  for (int i = 0; i < level.size(); i++) {
    //stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out(), NULL));
    std::vector<std::string> loop_vars;
    loop_vars.push_back(stmt[stmt_num].IS.set_var(level[i])->name());
    
    std::vector<omega::CG_outputRepr *> subs;
    subs.push_back(ocg->CreateInt(0));
    
    stmt[stmt_num].code = ocg->CreateSubstitutedStmt(0, stmt[stmt_num].code,
                                                     loop_vars, subs);
    
  }
  
  omega::Relation new_IS = copy(stmt[stmt_num].IS);
  new_IS.copy_names(stmt[stmt_num].IS);
  new_IS.setup_names();
  new_IS.simplify();
  int old_size = new_IS.n_set();
  
  omega::Relation R = omega::copy(stmt[stmt_num].IS);
  R.copy_names(stmt[stmt_num].IS);
  R.setup_names();
  
  for (int i = level.size() - 1; i >= 0; i--) {
    int j;
    
    for (j = 0; j < cudaized_levels.size(); j++) {
      if (cudaized_levels[j] == level[i])
        break;
      
    }
    
    if (j == cudaized_levels.size()) {
      R = omega::Project(R, level[i], omega::Input_Var);
      R.simplify();
      
    }
    //
    
  }
  
  omega::F_And *f_Root = R.and_with_and();
  for (int i = level.size() - 1; i >= 0; i--) {
    int j;
    
    for (j = 0; j < cudaized_levels.size(); j++) {
      if (cudaized_levels[j] == level[i])
        break;
      
    }
    
    if (j == cudaized_levels.size()) {
      
      omega::EQ_Handle h = f_Root->add_EQ();
      
      h.update_coef(R.set_var(level[i]), 1);
      h.update_const(-1);
    }
    //
    
  }
  
  R.simplify();
  stmt[stmt_num].IS = R;
}






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
  
  debug_fprintf(stderr, "\n                                                  Loop::init_loop()\n");

  debug_fprintf(stderr, "extract_ir_stmts()\n");
  debug_fprintf(stderr, "ir_tree has %d statements\n", ir_tree.size());

  // Mahdi: a temporary hack for getting dependence extraction changes integrated
  replaceCode_ind = 1;

  ir_stmt = extract_ir_stmts(ir_tree);

  debug_fprintf(stderr,"nesting level stmt size = %d\n", (int)ir_stmt.size());
  stmt_nesting_level_.resize(ir_stmt.size());

  std::vector<int> stmt_nesting_level(ir_stmt.size());
  
  debug_fprintf(stderr, "%d statements?\n", (int)ir_stmt.size());

  // find out how deeply nested each statement is.  (how can these be different?)
  for (int i = 0; i < ir_stmt.size(); i++) {
    debug_fprintf(stderr, "i %d\n", i); 
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
    debug_fprintf(stderr, "stmt_nesting_level[%d] = %d\n", i, t); 
  }
  
  if (actual_code.size() == 0)
    actual_code = std::vector<CG_outputRepr*>(ir_stmt.size());
  
  stmt = std::vector<Statement>(ir_stmt.size());
  debug_fprintf(stderr, "in init_loop, made %d stmts\n", (int)ir_stmt.size());
  
  uninterpreted_symbols =             std::vector<std::map<std::string, std::vector<omega::CG_outputRepr * > > >(ir_stmt.size());
  uninterpreted_symbols_stringrepr =  std::vector<std::map<std::string, std::vector<omega::CG_outputRepr * > > >(ir_stmt.size());
  unin_rel = std::vector<std::map<std::string, std::vector<omega::Relation> > >(ir_stmt.size());
  
  int n_dim = -1;
  int max_loc;
  //std::vector<std::string> index;
  for (int i = 0; i < ir_stmt.size(); i++) {
    int max_nesting_level = -1;
    int loc = -1;
    
    // find the max nesting level and remember the statement that was at that level
    for (int j = 0; j < ir_stmt.size(); j++) {
      if (stmt_nesting_level[j] > max_nesting_level) {
        max_nesting_level = stmt_nesting_level[j];
        loc = j;
      }
    }
    
    debug_fprintf(stderr, "max nesting level %d at location %d\n", max_nesting_level, loc); 
    
    // most deeply nested statement acting as a reference point
    if (n_dim == -1) {
      debug_fprintf(stderr, "loop.cc L356  n_dim now max_nesting_level %d\n", max_nesting_level); 
      n_dim = max_nesting_level;
      max_loc = loc;
      
      index = std::vector<std::string>(n_dim);
      
      ir_tree_node *itn = ir_stmt[loc];
      debug_fprintf(stderr, "itn = stmt[%d]\n", loc); 
      int cur_dim = n_dim - 1;
      while (itn->parent != NULL) {
        debug_fprintf(stderr, "parent\n"); 
        
        itn = itn->parent;
        if (itn->content->type() == IR_CONTROL_LOOP) {
          debug_fprintf(stderr, "IR_CONTROL_LOOP  cur_dim %d\n", cur_dim); 
          IR_Loop *IRL = static_cast<IR_Loop *>(itn->content);
          index[cur_dim] = IRL->index()->name();
          debug_fprintf(stderr, "index[%d] = '%s'\n", cur_dim, index[cur_dim].c_str());
          itn->payload = cur_dim--;
        }
      }
    }
    
    debug_fprintf(stderr, "align loops by names,\n"); 
    // align loops by names, temporary solution
    ir_tree_node *itn = ir_stmt[loc];  // defined outside loops?? 
    int depth = stmt_nesting_level_[loc] - 1;
    
    for (int t = depth; t >= 0; t--) {
      int y = t;
      itn = ir_stmt[loc];
      
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
    
    debug_fprintf(stderr, "\nset relation variable names                      ****\n");
    // set relation variable names
    
    // this finds the loop variables for loops enclosing this statement and puts
    // them in an Omega Relation (just their names, which could fail) 
    
    debug_fprintf(stderr, "Relation r(%d)\n", n_dim); 
    Relation r(n_dim);
    std::vector<string> insp_lb;
    std::vector<string> insp_ub;
    F_And *f_root = r.add_and();
    itn = ir_stmt[loc];
    int temp_depth = depth;

    for (int i =1; i<= n_dim; ++i)
      r.name_set_var(i, index[i - 1]);

    debug_begin
      printf("Relation r   ");
      r.print();
    debug_end

    debug_fprintf(stderr, "extract information from loop/if structures\n"); 
    // extract information from loop/if structures
    std::vector<bool> processed(n_dim, false);
    std::vector<std::string> vars_to_be_reversed;
    itn = ir_stmt[loc];

    while (itn->parent != NULL) { // keep heading upward 
      itn = itn->parent;
      
      switch (itn->content->type()) {
      case IR_CONTROL_LOOP: {
        debug_fprintf(stderr, "loop.cc l 462  IR_CONTROL_LOOP\n"); 
        IR_Loop *lp = static_cast<IR_Loop *>(itn->content);
        Variable_ID v = r.set_var(itn->payload + 1);
        int c;
        
        try {
          c = lp->step_size();
          //debug_fprintf(stderr, "step size %d\n", c); 
          if (c > 0) {
            CG_outputRepr *lb = lp->lower_bound();

            exp2formula(this, ir, r, f_root, freevar, lb, v, 's',
                        IR_COND_GE, true, uninterpreted_symbols[loc],
                        uninterpreted_symbols_stringrepr[loc],
                        unin_rel[loc]);
            // TODO Anand's return a vector - Usage?
            CG_outputRepr *ub = lp->upper_bound();

            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_LT || cond == IR_COND_LE)
              exp2formula(this, ir, r, f_root, freevar, ub,
                          v, 's', cond, true,
                          uninterpreted_symbols[loc],
                          uninterpreted_symbols_stringrepr[loc],
                          unin_rel[loc]);
            else
              throw ir_error("loop condition not supported");
            
            
            if ((ir->QueryExpOperation(lp->lower_bound())
                 == IR_OP_ARRAY_VARIABLE)
                && (ir->QueryExpOperation(lp->lower_bound())
                    == ir->QueryExpOperation(
                                             lp->upper_bound()))) {
              
              debug_fprintf(stderr, "loop.cc lower and upper are both IR_OP_ARRAY_VARIABLE?\n"); 
              
              std::vector<CG_outputRepr *> v =
                ir->QueryExpOperand(lp->lower_bound());
              IR_ArrayRef *ref =
                static_cast<IR_ArrayRef *>(ir->Repr2Ref(v[0]));
              std::string s0 = ref->name();
              std::vector<CG_outputRepr *> v2 =
                ir->QueryExpOperand(lp->upper_bound());
              IR_ArrayRef *ref2 =
                static_cast<IR_ArrayRef *>(ir->Repr2Ref(v2[0]));
              std::string s1 = ref2->name();
              
              if (s0 == s1) {
                insp_lb.push_back(s0);
                insp_ub.push_back(s1);
              }
            }
          } else if (c < 0) {
            CG_outputBuilder *ocg = ir->builder();
            CG_outputRepr *lb = lp->lower_bound();
            lb = ocg->CreateMinus(NULL, lb);
            exp2formula(this, ir, r, f_root, freevar, lb, v, 's',
                        IR_COND_GE, true, uninterpreted_symbols[loc],
                        uninterpreted_symbols_stringrepr[loc],
                        unin_rel[loc]);
            CG_outputRepr *ub = lp->upper_bound();
            ub = ocg->CreateMinus(NULL, ub);
            IR_CONDITION_TYPE cond = lp->stop_cond();
            if (cond == IR_COND_GE)
              exp2formula(this, ir, r, f_root, freevar, ub, v,
                          's', IR_COND_LE, true,
                          uninterpreted_symbols[loc],
                          uninterpreted_symbols_stringrepr[loc],
                          unin_rel[loc]);
            else if (cond == IR_COND_GT)
              exp2formula(this, ir, r, f_root, freevar, ub, v,
                          's', IR_COND_LT, true,
                          uninterpreted_symbols[loc],
                          uninterpreted_symbols_stringrepr[loc],
                          unin_rel[loc]);
            else
              throw ir_error("loop condition not supported");
            
            vars_to_be_reversed.push_back(lp->index()->name());
          } else
            throw ir_error("loop step size zero");
        } catch (const ir_error &e) {
          actual_code[loc] =
            static_cast<IR_Block *>(ir_stmt[loc]->content)->extract();
          for (int i = 0; i < itn->children.size(); i++)
            delete itn->children[i];
          itn->children = std::vector<ir_tree_node *>();
          itn->content = itn->content->convert();
          return false;
        }
        
        // check for loop increment or decrement that is not 1
        //debug_fprintf(stderr, "abs(c)\n"); 
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
          exp2formula(this, ir, r, f_and, freevar, lb, e, 's',
                      IR_COND_EQ, true, uninterpreted_symbols[loc],
                      uninterpreted_symbols_stringrepr[loc],
                      unin_rel[loc]);
        }
        
        processed[itn->payload] = true;
        break;
      }
        
        
      case IR_CONTROL_IF: {

        debug_fprintf(stderr, "IR_CONTROL_IF\n"); 
        IR_If *theif = static_cast<IR_If *>(itn->content);
        
// Mahdi: In current form, following only supports one condition in the if-statement
//        something like following is not supported: if( A && B )

        CG_outputRepr *cond =
          static_cast<IR_If *>(itn->content)->condition();

        try {
          if (itn->payload % 2 == 1)
            exp2constraint(this, ir, r, f_root, freevar, cond, true,
                           uninterpreted_symbols[loc],
                           uninterpreted_symbols_stringrepr[loc],
                           unin_rel[loc]);
          else {
            F_Not *f_not = f_root->add_not();
            F_And *f_and = f_not->add_and();
            exp2constraint(this, ir, r, f_and, freevar, cond, true,
                           uninterpreted_symbols[loc],
                           uninterpreted_symbols_stringrepr[loc],
                           unin_rel[loc]);
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
        //debug_fprintf(stderr, "default?\n"); 
        for (int i = 0; i < itn->children.size(); i++)
          delete itn->children[i];
        itn->children = std::vector<ir_tree_node *>();
        itn->content = itn->content->convert();
        return false;
      }
    }
    
    
    //debug_fprintf(stderr, "add information for missing loops   n_dim(%d)\n", n_dim);
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

          exp2formula(this, ir, r, f_root, freevar, lb, v, 's',
                      IR_COND_EQ, false, uninterpreted_symbols[loc],
                      uninterpreted_symbols_stringrepr[loc],
                      unin_rel[loc]);

        } else { // loc > max_loc
          
          CG_outputBuilder *ocg = ir->builder();
          CG_outputRepr *ub =
            static_cast<IR_Loop *>(itn->content)->upper_bound();

          exp2formula(this, ir, r, f_root, freevar, ub, v, 's',
                      IR_COND_EQ, false, uninterpreted_symbols[loc],
                      uninterpreted_symbols_stringrepr[loc],
                      unin_rel[loc]);

        }
      }
    r.setup_names();
    r.simplify();

    // THIS IS MISSING IN PROTONU's
    for (int j = 0; j < insp_lb.size(); j++) {
      
      std::string lb = insp_lb[j] + "_";
      std::string ub = lb + "_";
      
      Global_Var_ID u, l;
      bool found_ub = false;
      bool found_lb = false;
      for (DNF_Iterator di(copy(r).query_DNF()); di; di++)
        for (Constraint_Iterator ci = (*di)->constraints(); ci; ci++)
          
          for (Constr_Vars_Iter cvi(*ci); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            if (v->kind() == Global_Var)
              if (v->get_global_var()->arity() > 0) {
                
                std::string name =
                  v->get_global_var()->base_name();
                if (name == lb) {
                  l = v->get_global_var();
                  found_lb = true;
                } else if (name == ub) {
                  u = v->get_global_var();
                  found_ub = true;
                }
              }
            
          }
      
      if (found_lb && found_ub) {
        Relation known_(copy(r).n_set());
        known_.copy_names(copy(r));
        known_.setup_names();
        Variable_ID index_lb = known_.get_local(l, Input_Tuple);
        Variable_ID index_ub = known_.get_local(u, Input_Tuple);
        F_And *fr = known_.add_and();
        GEQ_Handle g = fr->add_GEQ();
        g.update_coef(index_ub, 1);
        g.update_coef(index_lb, -1);
        g.update_const(-1);
        addKnown(known_);
        
      }
      
    }
    
    
    debug_fprintf(stderr, "loop.cc L441 insert the statement\n");
    // insert the statement
    CG_outputBuilder *ocg = ir->builder();
    CG_stringBuilder ocg_s = ir->builder_s();
    std::vector<CG_outputRepr *> reverse_expr;
    for (int j = 0; j < vars_to_be_reversed.size(); j++) {
      CG_outputRepr *repl = ocg->CreateIdent(vars_to_be_reversed[j]);
      repl = ocg->CreateMinus(NULL, repl);
      reverse_expr.push_back(repl->clone());
      CG_outputRepr *repl_ = ocg_s.CreateIdent(vars_to_be_reversed[j]);
      repl_ = ocg_s.CreateMinus(NULL, repl_);
      std::vector<CG_outputRepr *> reverse_;

      reverse_.push_back(repl_->clone());
      int pos;
      for (int k = 1; k <= r.n_set(); k++)
        if (vars_to_be_reversed[j] == r.set_var(k)->name())
          pos = k;

//			for(int k =0; k < ir_stmt.size(); k++ ){
      for (std::map<std::string, std::vector<omega::CG_outputRepr *> >::iterator it =
          uninterpreted_symbols[loc].begin();
           it != uninterpreted_symbols[loc].end(); it++) {

        for (int l = 0; l < freevar.size(); l++) {
          int arity;
          if (freevar[l]->base_name() == it->first) {
            arity = freevar[l]->arity();
            if (arity >= pos) {
              std::vector<CG_outputRepr *> tmp = it->second;
              std::vector<CG_outputRepr *> reverse;
              reverse.push_back(repl->clone());
              std::vector<std::string> tmp2;
              tmp2.push_back(vars_to_be_reversed[j]);
              tmp[pos - 1] = ocg->CreateSubstitutedStmt(0,
                                                        tmp[pos - 1]->clone(), tmp2, reverse);

              it->second = tmp;
              break;
            }
          }

        }

      }

      for (std::map<std::string, std::vector<omega::Relation> >::iterator it =
          unin_rel[loc].begin(); it != unin_rel[loc].end(); it++) {

        if (it->second.size() > 0) {
          for (int l = 0; l < freevar.size(); l++) {
            int arity;
            if (freevar[l]->base_name() == it->first) {
              arity = freevar[l]->arity();
              if (arity >= pos) {

                break;

                std::vector<omega::Relation> reprs_ = it->second;
                std::vector<omega::Relation> reprs_2;
                //					for (int k = 0; k < reprs_.size(); k++) {
                omega::Relation r(reprs_[pos - 1].n_inp(), 1);

                omega::F_And *root = r.add_and();
                omega::EQ_Handle h1 = root->add_EQ();

                h1.update_coef(r.output_var(1), 1);
                for (omega::EQ_Iterator e(
                    reprs_[pos - 1].single_conjunct()->EQs());
                     e; e++) {
                  for (omega::Constr_Vars_Iter c(*e); c;
                       c++) {

                    if ((*e).get_const() > 0)
                      h1.update_const((*e).get_const());
                    else
                      h1.update_const(
                          -((*e).get_const()));

                    omega::Variable_ID v = c.curr_var();
                    switch (v->kind()) {
                      case omega::Input_Var: {
                        int coef = c.curr_coef();
                        if (coef < 0)
                          coef *= (-1);
                        h1.update_coef(
                            r.input_var(
                                v->get_position()),
                            coef);

                        break;
                      }
                      case omega::Wildcard_Var: {
                        omega::F_Exists *f_exists =
                            root->add_exists();
                        int coef = c.curr_coef();
                        if (coef < 0)
                          coef *= (-1);
                        std::map<omega::Variable_ID,
                            omega::Variable_ID> exists_mapping;
                        omega::Variable_ID v2 =
                            replicate_floor_definition(
                                copy(
                                    reprs_[pos
                                           - 1]),
                                v, r, f_exists,
                                root,
                                exists_mapping);
                        h1.update_coef(v2, coef);
                        break;
                      }
                      default:
                        break;

                        //h1.update_const(result1.first.get_const());

                    }

                    //	}

                    //}

                  }

                  reprs_[pos - 1] = r;
                  break;
                  //}

                  it->second = reprs_;

                }
              }

            }

          }

        }
      }
    }
    debug_fprintf(stderr, "loop.cc before extract\n"); 
    CG_outputRepr *code =
      static_cast<IR_Block *>(ir_stmt[loc]->content)->extract();
    code = ocg->CreateSubstitutedStmt(0, code, vars_to_be_reversed,
                                      reverse_expr);

    stmt[loc].code = code;
    stmt[loc].IS = r;
    
    //Anand: Add Information on uninterpreted function constraints to
    //Known relation
    
    debug_fprintf(stderr, "loop.cc stmt[%d].loop_level has size n_dim %d\n", loc, n_dim); 

    stmt[loc].loop_level = std::vector<LoopLevel>(n_dim);
    stmt[loc].ir_stmt_node = ir_stmt[loc];
    stmt[loc].has_inspector = false;
    debug_fprintf(stderr, "for int i < n_dim(%d)\n", n_dim); 
    for (int ii = 0; ii < n_dim; ii++) {
      stmt[loc].loop_level[ii].type = LoopLevelOriginal;
      stmt[loc].loop_level[ii].payload = ii;
      stmt[loc].loop_level[ii].parallel_level = 0;
      stmt[loc].loop_level[ii].segreducible = false;
    }
    debug_fprintf(stderr, "whew\n"); 
    
    stmt_nesting_level[loc] = -1;
  }
  debug_begin
    dump();
    debug_fprintf(stderr, "                                        loop.cc   Loop::init_loop() END\n\n");
  debug_end

  return true;
}


// Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch
// buildIS is basically suppose to replace init_loop in Tuowens branch, and init_loop commented out
// however since Tuowen may want to keep somethings from init_loop I am leaving it there for now
std::string index_name(int level) {
  std::string iname = ("chill_idx"+to_string(level));
  return iname;
}

void Loop::buildIS(std::vector<ir_tree_node*> &ir_tree,std::vector<int> &lexicalOrder,std::vector<ir_tree_node*> &ctrls, int level) {
  for (int i = 0; i < ir_tree.size(); i++) {
    switch (ir_tree[i]->content->type()) {
      case IR_CONTROL_BLOCK: {
        // A new stmt
        // Setting up basic variables
        // Needs cleanup
        ir_stmt.push_back(ir_tree[i]);
        stmt.push_back(Statement());
        uninterpreted_symbols.emplace_back();
        uninterpreted_symbols_stringrepr.emplace_back();
        unin_rel.emplace_back();
        stmt_nesting_level_.push_back(level);
        int loc = ir_stmt.size() - 1;
        // Setup the IS holder
        Relation r(num_dep_dim);
        F_And *f_root = r.add_and();
        // Setup inspector bounds
        std::vector<std::string> insp_lb;
        std::vector<std::string> insp_ub;

        int current = 0;

        // Processing information from containing control structs
        for (int j = 0; j < ctrls.size(); ++j) {
          switch (ctrls[j]->content->type()) {
            case IR_CONTROL_LOOP: {
              IR_Loop *lp = static_cast<IR_Loop *>(ctrls[j]->content);
              ++current;
              Variable_ID v = r.set_var(current);

              // Create index replacement
              std::string iname = index_name(current);
              r.name_set_var(current,iname);

              int step = lp->step_size();
              CG_outputRepr *lb = lp->lower_bound();
              CG_outputRepr *ub = lp->upper_bound();
              IR_CONDITION_TYPE cond = lp->stop_cond();
              exp2formula(this, ir,r,f_root,freevar,lb,v,'s',IR_COND_GE,false,uninterpreted_symbols[loc],uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
              if (cond == IR_COND_LT || cond == IR_COND_LE)
                exp2formula(this, ir,r,f_root,freevar,ub,v,'s',cond,false,uninterpreted_symbols[loc],uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
              else throw ir_error("loop condition not supported");

              // strided
              if (step != 1) {
                F_Exists *f_exists = f_root->add_exists();
                Variable_ID  e = f_exists -> declare();
                F_And *f_and = f_exists->add_and();
                Stride_Handle h = f_and->add_stride(step);
                h.update_coef(e ,1);
                h.update_coef(v, -1);
                // Here is using substituted lowerbound
                exp2formula(this,ir, r, f_and, freevar, lb,e,'s',IR_COND_EQ, false, uninterpreted_symbols[loc],uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
              }
              if ((ir->QueryExpOperation(lp->lower_bound())
                   == IR_OP_ARRAY_VARIABLE)
                  && (ir->QueryExpOperation(lp->lower_bound())
                      == ir->QueryExpOperation(
                  lp->upper_bound()))) {
                std::vector<CG_outputRepr *> v =
                    ir->QueryExpOperand(lp->lower_bound());
                IR_ArrayRef *ref =
                    static_cast<IR_ArrayRef *>(ir->Repr2Ref(
                        v[0]));
                std::string s0 = ref->name();
                std::vector<CG_outputRepr *> v2 =
                    ir->QueryExpOperand(lp->upper_bound());
                IR_ArrayRef *ref2 =
                    static_cast<IR_ArrayRef *>(ir->Repr2Ref(
                        v2[0]));
                std::string s1 = ref2->name();

                if (s0 == s1) {
                  insp_lb.push_back(s0);
                  insp_ub.push_back(s1);

                }
              }
              break;
            }
            case IR_CONTROL_IF: {
              IR_If *ip = static_cast<IR_If *>(ctrls[j]->content);
              CG_outputRepr *cond = ip->condition();
              if (ctrls[j]->payload & 1)
                exp2constraint(this,ir,r,f_root,freevar,cond,false, uninterpreted_symbols[loc],uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
              else {
                F_Not *f_not = f_root->add_not();
                F_And *f_and = f_not->add_and();
                exp2constraint(this,ir,r,f_and,freevar,cond,false, uninterpreted_symbols[loc],uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
              }
              break;
            }
            default:
              throw ir_error("unknown ir type"); // should never happen
          }
        }

        // add information for missing loops
        for (int j = level; j < num_dep_dim; j++) {
          Variable_ID v = r.set_var(j + 1);
          EQ_Handle e = f_root->add_EQ();
          e.update_coef(v, 1);
        }
        r.setup_names();
        r.simplify();
        // Inspector initialization
        for (int j = 0; j < insp_lb.size(); j++) {

          std::string lb = insp_lb[j] + "_";
          std::string ub = lb + "_";

          Global_Var_ID u, l;
          bool found_ub = false;
          bool found_lb = false;
          for (DNF_Iterator di(copy(r).query_DNF()); di; di++)
            for (Constraint_Iterator ci = (*di)->constraints(); ci; ci++)

              for (Constr_Vars_Iter cvi(*ci); cvi; cvi++) {
                Variable_ID v = cvi.curr_var();
                if (v->kind() == Global_Var)
                  if (v->get_global_var()->arity() > 0) {

                    std::string name =
                        v->get_global_var()->base_name();
                    if (name == lb) {
                      l = v->get_global_var();
                      found_lb = true;
                    } else if (name == ub) {
                      u = v->get_global_var();
                      found_ub = true;
                    }
                  }

              }

          if (found_lb && found_ub) {
            Relation known_(copy(r).n_set());
            known_.copy_names(copy(r));
            known_.setup_names();
            Variable_ID index_lb = known_.get_local(l, Input_Tuple);
            Variable_ID index_ub = known_.get_local(u, Input_Tuple);
            F_And *fr = known_.add_and();
            GEQ_Handle g = fr->add_GEQ();
            g.update_coef(index_ub, 1);
            g.update_coef(index_lb, -1);
            g.update_const(-1);
            addKnown(known_);
          }
        }
        // Write back
        stmt[loc].code = static_cast<IR_Block*>(ir_stmt[loc]->content)->extract();
        stmt[loc].IS = r;
        stmt[loc].loop_level = std::vector<LoopLevel>(num_dep_dim);
        stmt[loc].has_inspector = false;
        stmt[loc].ir_stmt_node = ir_tree[i];
        for (int ii = 0; ii < num_dep_dim; ++ii) {
          stmt[loc].loop_level[ii].type = LoopLevelOriginal;
          stmt[loc].loop_level[ii].payload = ii;
          stmt[loc].loop_level[ii].parallel_level = 0;
        }
        // Lexical ordering
        stmt[loc].xform = Relation(num_dep_dim, 2 * num_dep_dim + 1);
        F_And *f_xform = stmt[loc].xform.add_and();

        for (int j = 1; j <= num_dep_dim; j++) {
          EQ_Handle h = f_xform->add_EQ();
          h.update_coef(stmt[loc].xform.output_var(2 * j), 1);
          h.update_coef(stmt[loc].xform.input_var(j), -1);
        }
        for (int j = 1; j <= 2 * num_dep_dim + 1; j += 2) {
          EQ_Handle h = f_xform->add_EQ();
          h.update_coef(stmt[loc].xform.output_var(j), 1);
          if (j/2 < lexicalOrder.size())
          h.update_const(-lexicalOrder[j/2]);
        }
        stmt[loc].xform.simplify();
        // Update lexical ordering for next statement
        lexicalOrder[lexicalOrder.size()-1]++;
        break;
      }
      case IR_CONTROL_LOOP: {
        ir_tree[i]->payload = level;
        ctrls.push_back(ir_tree[i]);
        try {
          lexicalOrder.push_back(0);
          buildIS(ir_tree[i]->children, lexicalOrder, ctrls, level +1);
          lexicalOrder.pop_back();
        } catch (ir_error &e) {
          for (int j =0;j<ir_tree[i]->children.size(); ++j)
            delete ir_tree[i]->children[j];
          ir_tree[i]->children = std::vector<ir_tree_node*>();
          ir_tree[i]->content = ir_tree[i]->content->convert();
          throw chill::error::build("converted ir_tree_node");
        }
        ctrls.pop_back();
        // Update lexical ordering for next statement
        lexicalOrder[lexicalOrder.size()-1]++;
        break;
      }
      case IR_CONTROL_IF: {
        // need to change condition to align loop vars
        ctrls.push_back(ir_tree[i]);
        try {
          buildIS(ir_tree[i]->children, lexicalOrder, ctrls, level);
        } catch (ir_error &e) {
          for (int j =0;j<ir_tree[i]->children.size(); ++j)
            delete ir_tree[i]->children[j];
          ir_tree[i]->children = std::vector<ir_tree_node*>();
          ir_tree[i]->content = ir_tree[i]->content->convert();
          throw chill::error::build("converted ir_tree_node");
        }
        ctrls.pop_back();
        // if statement shouldn't update the lexical ordering on its own.
        break;
      }
      default:
        throw std::invalid_argument("invalid ir tree");
    }
  }
}

int find_depth(std::vector<ir_tree_node *> &ir_tree) {
  int maxd = 0;
  for (int i = 0; i < ir_tree.size(); i++)
    switch (ir_tree[i]->content->type()) {
      case IR_CONTROL_BLOCK:
        // A new stmt
        break;
      case IR_CONTROL_LOOP:
        maxd = max(maxd,find_depth(ir_tree[i]->children)+1);
        break;
      case IR_CONTROL_IF:
        maxd = max(maxd,find_depth(ir_tree[i]->children));
        break;
      default:
        throw std::invalid_argument("invalid ir tree");
    }
  return maxd;
}

void Loop::align_loops(std::vector<ir_tree_node*> &ir_tree, std::vector<std::string> &vars_to_be_replaced, std::vector<CG_outputRepr*> &vars_replacement,int level) {
  for (int i = 0; i < ir_tree.size(); i++) {
    CG_outputBuilder *ocg = ir->builder();
    switch (ir_tree[i]->content->type()) {
      case IR_CONTROL_BLOCK: {
        IR_Block *bp = static_cast<IR_Block *>(ir_tree[i]->content);
        ocg->CreateSubstitutedStmt(0,bp->extract(),vars_to_be_replaced,vars_replacement,false);
        break;
      }
      case IR_CONTROL_LOOP: {
        IR_chillLoop *clp = static_cast<IR_chillLoop *>(ir_tree[i]->content);
        if (!clp->well_formed) {
          for (int j = 0; j < ir_tree[i]->children.size(); ++j)
            delete ir_tree[i]->children[j];
          ir_tree[i]->children = std::vector<ir_tree_node *>();
          ir_tree[i]->content = ir_tree[i]->content->convert();
        } else {
          clp->chilllowerbound = ocg->CreateSubstitutedStmt(0,clp->chilllowerbound,vars_to_be_replaced,vars_replacement,false);
          clp->chillupperbound = ocg->CreateSubstitutedStmt(0,clp->chillupperbound,vars_to_be_replaced,vars_replacement,false);
          std::string iname = index_name(level);
          CG_outputRepr *ivar = ocg->CreateIdent(iname);
          vars_to_be_replaced.push_back(clp->index()->name());
          vars_replacement.push_back(ivar);
          // FIXME: this breaks abstraction
          if (clp->step_size()<0) {
            IR_CONDITION_TYPE cond = clp->conditionoperator;
            if (cond == IR_COND_GE) cond = IR_COND_LE;
            else if (cond == IR_COND_GT) cond = IR_COND_LT;
            clp->conditionoperator = cond;
            clp->chilllowerbound = ocg->CreateMinus(NULL, clp->chilllowerbound);
            clp->chillupperbound = ocg->CreateMinus(NULL, clp->chillupperbound);
            clp->step_size_ = -clp->step_size_;
            CG_outputRepr *inv = ocg->CreateMinus(NULL,ivar);
            vars_to_be_replaced.push_back(iname);
            vars_replacement.push_back(inv);
            clp->chillforstmt->cond = new chillAST_BinaryOperator(((CG_chillRepr*)(clp->chillupperbound))->chillnodes[0],((chillAST_BinaryOperator*)(clp->chillforstmt->getCond()))->getOp()
                ,((CG_chillRepr*)ivar)->chillnodes[0]);
          } else
            clp->chillforstmt->cond = new chillAST_BinaryOperator(((CG_chillRepr*)ivar)->chillnodes[0],((chillAST_BinaryOperator*)(clp->chillforstmt->getCond()))->getOp()
                ,((CG_chillRepr*)(clp->chillupperbound))->chillnodes[0]);
          clp->chillforstmt->init = new chillAST_BinaryOperator(((CG_chillRepr*)ivar)->chillnodes[0],"=",((CG_chillRepr*)(clp->chilllowerbound))->chillnodes[0]);
          clp->chillforstmt->incr = new chillAST_BinaryOperator(((CG_chillRepr*)ivar)->chillnodes[0],"+=",new chillAST_IntegerLiteral(clp->step_size_));
          // Ready to recurse
          align_loops(ir_tree[i]->children,vars_to_be_replaced,vars_replacement,level+1);
        }
        break;
      }
      case IR_CONTROL_IF: {
        IR_If *ip = static_cast<IR_If *>(ir_tree[i]->content);
        ocg->CreateSubstitutedStmt(0,ip->condition(),vars_to_be_replaced,vars_replacement,false);
        // Ready to recurse
        align_loops(ir_tree[i]->children,vars_to_be_replaced,vars_replacement,level);
        break;
      }
      default:
        throw std::invalid_argument("invalid ir tree");
    }
  }
}



Loop::Loop(const IR_Control *control) {
  
  debug_fprintf(stderr, "\nLoop::Loop(const IR_Control *control)\n");
  debug_fprintf(stderr, "control type is %d   ", control->type()); 
  echocontroltype(control);
  
  // Mahdi: A temporary hack for getting dependence extraction changes integrated       
  replaceCode_ind = 1;

  last_compute_cgr_ = NULL;
  last_compute_cg_ = NULL;
  debug_fprintf(stderr, "2set last_compute_cg_ = NULL; \n"); 

  ir = const_cast<IR_Code *>(control->ir_); // point to the CHILL IR that this loop came from
  if (!ir) {
    debug_fprintf(stderr, "ir gotten from control = 0x%x\n", (long)ir);
    debug_fprintf(stderr, "loop.cc GONNA DIE SOON *******************************\n\n");
  }
  
  init_code = NULL;
  cleanup_code = NULL;
  tmp_loop_var_name_counter = 1;
  overflow_var_name_counter = 1;
  known = Relation::True(0);
  
  debug_fprintf(stderr, "in Loop::Loop, calling  build_ir_tree()\n"); 
  debug_fprintf(stderr, "\nloop.cc, Loop::Loop() about to clone control\n"); 
  ir_tree = build_ir_tree(control->clone(), NULL);
  //debug_fprintf(stderr,"in Loop::Loop. ir_tree has %ld parts\n", ir_tree.size()); 
  
  //    std::vector<ir_tree_node *> ir_stmt;
  //debug_fprintf(stderr, "loop.cc after build_ir_tree() %ld statements\n",  stmt.size()); 
  
// Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch
/*
  while (!init_loop(ir_tree, ir_stmt)) {
    //debug_fprintf(stderr, "count %d\n", count++); 
  }
  debug_fprintf(stderr, "after init_loop, %d freevar\n", (int)freevar.size()); 
  
  
  debug_fprintf(stderr, "loop.cc after init_loop, %d statements\n",  (int)stmt.size()); 
*/
  {
    std::vector<std::string> vars_to_be_relaced;
    std::vector<CG_outputRepr*> vars_replacement;
    align_loops(ir_tree, vars_to_be_relaced,vars_replacement,/*loop_index_start*/1);
  }
  bool trybuild = true;

  while (trybuild)
  {
    uninterpreted_symbols_stringrepr.clear();
    uninterpreted_symbols.clear();
    unin_rel.clear();
    stmt.clear();
    ir_stmt.clear();
    stmt_nesting_level_.clear();
    num_dep_dim = find_depth(ir_tree);
    std::vector<int> lexicalOrder;
    std::vector<ir_tree_node*> ctrls;
    lexicalOrder.push_back(0);
    trybuild = false;
    try{
      buildIS(ir_tree,lexicalOrder,ctrls,0);
    } catch (chill::error::build &e) {
      debug_printf("Retry: %s", e.what());
      trybuild=true;
    }
  }

  for (int i = 0; i < stmt.size(); i++) {
    std::map<int, CG_outputRepr*>::iterator it = replace.find(i);
    
    if (it != replace.end())
      stmt[i].code = it->second;
    else
      stmt[i].code = stmt[i].code;

    // TODO replace_set_var_as_another_set_var
    // Relation r = parseExpWithWhileToRel(stmt[i].code, stmt[i].IS, i);
    // r.simplify();
  }
  
  if (stmt.size() != 0)
    dep = DependenceGraph(stmt[0].IS.n_set());
  else
    dep = DependenceGraph(0);
  // init the dependence graph
  for (int i = 0; i < stmt.size(); i++)
    dep.insert();

  for (int i = 0; i < stmt.size(); i++) {
    stmt[i].reduction = 0; // Manu -- initialization
    for (int j = i; j < stmt.size(); j++) {
      std::pair<std::vector<DependenceVector>,
                std::vector<DependenceVector> > dv = test_data_dependences(this,
                                                                           ir, 
                                                                           stmt[i].code, 
                                                                           stmt[i].IS, 
                                                                           stmt[j].code, 
                                                                           stmt[j].IS,
                                                                           freevar, 
                                                                           index, 
                                                                           stmt_nesting_level_[i],
                                                                           stmt_nesting_level_[j],
                                                                           uninterpreted_symbols[i],
                                                                           uninterpreted_symbols_stringrepr[i], unin_rel[i], dep_relation);
      
      debug_fprintf(stderr, "dv.first.size() %d\n", (int)dv.first.size()); 
      for (int k = 0; k < dv.first.size(); k++) {
        debug_fprintf(stderr, "k1 %d\n", k); 
        if (is_dependence_valid(ir_stmt[i], ir_stmt[j], dv.first[k],
                                true))
          dep.connect(i, j, dv.first[k]);
        else {
          dep.connect(j, i, dv.first[k].reverse());
        }
        
      }
      
      for (int k = 0; k < dv.second.size(); k++) { 
        debug_fprintf(stderr, "k2 %d\n", k); 
        if (is_dependence_valid(ir_stmt[j], ir_stmt[i], dv.second[k],
                                false))
          dep.connect(j, i, dv.second[k]);
        else {
          dep.connect(i, j, dv.second[k].reverse());
        }
      }
    }
  }

  debug_fprintf(stderr, "\n\n*** LOTS OF REDUCTIONS ***\n\n"); 
  
  // TODO: Reduction check
  // Manu:: Initial implementation / algorithm
  std::set<int> reducCand = std::set<int>();
  std::vector<int> canReduce = std::vector<int>();
  debug_fprintf(stderr, "\ni range %d\n", stmt.size()); 
  for (int i = 0; i < stmt.size(); i++) {
    debug_fprintf(stderr, "i %d\n", i); 
    if (!dep.hasEdge(i, i)) {
      continue;
    }
    debug_fprintf(stderr, "dep.hasEdge(%d, %d)\n", i, i);

    // for each statement check if it has all the three dependences (RAW, WAR, WAW)
    // If there is such a statement, it is a reduction candidate. Mark all reduction candidates.
    std::vector<DependenceVector> tdv = dep.getEdge(i, i);
    debug_fprintf(stderr, "tdv size %d\n", tdv.size());
    for (int j = 0; j < tdv.size(); j++) {
      debug_fprintf(stderr, "ij %d %d\n", i, j);
      if (tdv[j].is_reduction_cand) {
        debug_fprintf(stderr, "reducCand.insert( %d )\n", i);
        reducCand.insert(i);
      }
    }
  }
  
  debug_fprintf(stderr, "loop.cc reducCand.size() %d\n", reducCand.size()); 
  bool reduc;
  std::set<int>::iterator it;
  int counter = 0; 
  for (it = reducCand.begin(); it != reducCand.end(); it++) {
    reduc = true;
    for (int j = 0; j < stmt.size(); j++) {
      debug_fprintf(stderr, "j %d\n", j); 
      if ((*it != j)
          && (stmt_nesting_level_[*it] < stmt_nesting_level_[j])) {
        if (dep.hasEdge(*it, j) || dep.hasEdge(j, *it)) {
          reduc = false;
          break;
        }
      }
    }
    
    if (reduc) {
      debug_fprintf(stderr, "canReduce.push_back()\n"); 
      canReduce.push_back(*it);
      stmt[*it].reduction = 2; // First, assume that reduction is possible with some processing
    }
  }
  
  
  // If reduction is possible without processing, update the value of the reduction variable to 1
  debug_fprintf(stderr, "loop.cc canReduce.size() %d\n", canReduce.size()); 
  for (int i = 0; i < canReduce.size(); i++) {
    // Here, assuming that stmtType returns 1 when there is a single statement within stmt[i]
    if (stmtType(ir, stmt[canReduce[i]].code) == 1) {
      stmt[canReduce[i]].reduction = 1;
      IR_OPERATION_TYPE opType;
      opType = getReductionOperator(ir, stmt[canReduce[i]].code);
      stmt[canReduce[i]].reductionOp = opType;
    }
  }
  
  // printing out stuff for debugging
  
  if (DEP_DEBUG) {
    std::cout << "STATEMENTS THAT CAN BE REDUCED: \n";
    for (int i = 0; i < canReduce.size(); i++) {
      std::cout << "------- " << canReduce[i] << " ------- "
                << stmt[canReduce[i]].reduction << "\n";
      ir->printStmt(stmt[canReduce[i]].code); // Manu
      if (stmt[canReduce[i]].reductionOp == IR_OP_PLUS)
        std::cout << "Reduction type:: + \n";
      else if (stmt[canReduce[i]].reductionOp == IR_OP_MINUS)
        std::cout << "Reduction type:: - \n";
      else if (stmt[canReduce[i]].reductionOp == IR_OP_MULTIPLY)
        std::cout << "Reduction type:: * \n";
      else if (stmt[canReduce[i]].reductionOp == IR_OP_DIVIDE)
        std::cout << "Reduction type:: / \n";
      else
        std::cout << "Unknown reduction type\n";
    }
  }


// Mahdi: Commented to correct embedded iteration space: from Tuowen's topdown branch
/*
  // cleanup the IR tree
  
  // init transformations adding auxiliary indices e.g. [i, j] -> [ 0, i, 0, j, 0]
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
  //debug_fprintf(stderr, "done with dumb\n");
  
  if (stmt.size() != 0)
    num_dep_dim = stmt[0].IS.n_set();
  else
    num_dep_dim = 0;
*/
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
  
  this->invalidateCodeGen();
  
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
        throw loop_error("incorrect loop level information for statement "
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

void Loop::debugRelations() const { 
  const int m = stmt.size(); 
  {
    std::vector<Relation> IS(m);
    std::vector<Relation> xforms(m);
    
    for (int i = 0; i < m; i++) {
      IS[i] = stmt[i].IS;
      xforms[i] = stmt[i].xform;  // const stucks
    }
    
    printf("\nxforms:\n"); 
    for (int i = 0; i < m; i++) { xforms[i].print();  printf("\n"); }
    printf("\nIS:\n"); 
    for (int i = 0; i < m; i++) {      IS[i].print();  printf("\n"); }
    fflush(stdout); 
  }
}


CG_outputRepr *Loop::getCode(int effort) const {
  debug_fprintf(stderr,"\nloop.cc Loop::getCode(  effort %d )\n", effort ); 
  const int m = stmt.size();
  if (m == 0)
    return NULL;
  const int n = stmt[0].xform.n_out();
  // if the omega code generator has never been computed, initialize last_compute_cg_
  if (last_compute_cg_ == NULL) {
    debug_fprintf(stderr, "Loop::getCode() last_compute_cg_ == NULL\n"); 
    
    std::vector<Relation> IS(m);
    std::vector<Relation> xforms(m);
    for (int i = 0; i < m; i++) {
      IS[i] = stmt[i].IS;
      xforms[i] = stmt[i].xform;
    }
    Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
    debug_begin
      debugRelations();
      printf("\nknown:\n"); known.print(); printf("\n\n"); fflush(stdout);
    debug_end
    last_compute_cg_ = new CodeGen(xforms, IS, known);
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  else {
    debug_fprintf(stderr, "Loop::getCode() last_compute_cg_ NOT NULL\n"); 
  }
  // TODO: add omp pragmas to last_compute_cg_ here.
  // if codegen result ast has never been computed, create it from last_compute_cg
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  this->omp_apply_pragmas();

  // Copy loop statements
  std::vector<CG_outputRepr *> stmts(m);
  debug_fprintf(stderr, "%d stmts\n", m);
  for (int i = 0; i < m; i++)
    stmts[i] = stmt[i].code;
  CG_outputBuilder *ocg = ir->builder();

  // Get generate code
  debug_fprintf(stderr, "calling last_compute_cgr_->printRepr()\n"); 
  CG_outputRepr *repr = last_compute_cgr_->printRepr(ocg, stmts, 
                                                     uninterpreted_symbols);
  // Add init and cleanup code.
  if (init_code != NULL)
    repr = ocg->StmtListAppend(init_code->clone(), repr);
  if (cleanup_code != NULL)
    repr = ocg->StmtListAppend(repr, cleanup_code->clone());
  debug_fprintf(stderr,"\nloop.cc Loop::getCode( effort %d )   DONE\n", effort ); 
  return repr;
}




void Loop::printCode(int effort) const {
  debug_fprintf(stderr,"\nloop.cc Loop::printCode(  effort %d )\n", effort );
  const int m = stmt.size();
  if (m == 0)
    return;
  const int n = stmt[0].xform.n_out();
  
  if (last_compute_cg_ == NULL) {
    debug_fprintf(stderr, "Loop::printCode(), last_compute_cg_ == NULL\n"); 
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
  else debug_fprintf(stderr, "Loop::printCode(), last_compute_cg_ NOT NULL\n"); 
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }

  std::string repr = last_compute_cgr_->printString(
                                                    uninterpreted_symbols_stringrepr);
  debug_fprintf(stderr, "leaving Loop::printCode()\n"); 
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

std::vector<Relation> Loop::getNewIS() const {
  const int m = stmt.size();
  
  std::vector<Relation> new_IS(m);
  for (int i = 0; i < m; i++)
    new_IS[i] = getNewIS(i);
  
  return new_IS;
}

// pragmas are tied to loops only ???
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
  std::vector<ir_tree_node*> loop_order;
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
      //while (itn->content->type() != IR_CONTROL_LOOP && itn != NULL)
      //  itn = itn->parent;
      
      while ((itn != NULL) && (itn->payload != level - 1 || itn->content->type()!= IR_CONTROL_LOOP)) {
        itn = itn->parent;
      }
      
      if (itn == NULL)
        not_nested_at_this_level.insert(*it);
      else {
        std::map<ir_tree_node*, std::set<int> >::iterator it2 =
          sorted_by_loop.find(itn);
        
        if (it2 != sorted_by_loop.end())
          it2->second.insert(*it);
        else {
          loop_order.push_back(itn);
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
    for (auto itn: loop_order)
      to_return.push_back(sorted_by_loop[itn]);
  }
  return to_return;
}

void update_successors(int n, 
                       int node_num[], 
                       int cant_fuse_with[],
                       Graph<std::set<int>, bool> &g, 
                       std::list<int> &work_list,
                       std::list<bool> &type_list, 
                       std::vector<bool> types) {
  
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
    if (!types[m]) {
      cant_fuse_with[m] = std::max(cant_fuse_with[m], cant_fuse_with[n]);
    } else {
      if (has_bad_edge_path)
        cant_fuse_with[m] = std::max(cant_fuse_with[m], node_num[n]);
      else
        cant_fuse_with[m] = std::max(cant_fuse_with[m], cant_fuse_with[n]);
    }
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
    
    if (no_incoming_edges) {
      work_list.push_back(*i);
      type_list.push_back(types[*i]);
    }
  }
}



int Loop::getMinLexValue(std::set<int> stmts, int level) {
  
  int min;
  
  std::set<int>::iterator it = stmts.begin();
  min = getLexicalOrder(*it, level);
  
  for (; it != stmts.end(); it++) {
    int curr = getLexicalOrder(*it, level);
    if (curr < min)
      min = curr;
  }
  
  return min;
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



std::vector<std::set<int> > Loop::typed_fusion(Graph<std::set<int>, bool> g,
                                               std::vector<bool> &types) {
  
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
  std::list<bool> type_list;
  int cant_fuse_with[g.vertex.size()];
  int fused = 0;
  int lastfused = 0;
  int lastnum = 0;
  std::vector<std::set<int> > s;
  //Each Fused set's representative node
  
  int node_to_fused_nodes[g.vertex.size()];
  int node_num[g.vertex.size()];
  int next[g.vertex.size()];
  
  for (int i = 0; i < g.vertex.size(); i++) {
    if (roots[i] == true) {
      work_list.push_back(i);
      type_list.push_back(types[i]);
    }
    cant_fuse_with[i] = 0;
    node_to_fused_nodes[i] = 0;
    node_num[i] = -1;
    next[i] = 0;
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
    bool type = type_list.front();
    //int n_ = g.vertex[n].first;
    work_list.pop_front();
    type_list.pop_front();
    int node;
    /*if (cant_fuse_with[n] == 0)
      node = 0;
      else
      node = cant_fuse_with[n];
    */
    int p;
    if (type) {
      //if ((fused_nodes_counter != 0) && (node != fused_nodes_counter)) {
      if (cant_fuse_with[n] == 0)
        p = fused;
      else
        p = next[cant_fuse_with[n]];
      
      if (p != 0) {
        int rep_node = node_to_fused_nodes[p];
        node_num[n] = node_num[rep_node];
        
        try {
          update_successors(n, node_num, cant_fuse_with, g, work_list,
                            type_list, types);
        } catch (const loop_error &e) {
          
          throw loop_error(
                           "statements cannot be fused together due to negative dependence");
          
        }
        for (std::set<int>::iterator it = g.vertex[n].first.begin();
             it != g.vertex[n].first.end(); it++)
          s[node_num[n] - 1].insert(*it);
      } else {
        //std::set<int> new_node;
        //new_node.insert(n_);
        s.push_back(g.vertex[n].first);
        lastnum = lastnum + 1;
        node_num[n] = lastnum;
        node_to_fused_nodes[node_num[n]] = n;
        
        if (lastfused == 0) {
          fused = lastnum;
          lastfused = fused;
        } else {
          next[lastfused] = lastnum;
          lastfused = lastnum;
          
        }
        
        try {
          update_successors(n, node_num, cant_fuse_with, g, work_list,
                            type_list, types);
        } catch (const loop_error &e) {
          
          throw loop_error(
                           "statements cannot be fused together due to negative dependence");
          
        }
        fused_nodes_counter++;
      }
      
    } else {
      s.push_back(g.vertex[n].first);
      lastnum = lastnum + 1;
      node_num[n] = lastnum;
      node_to_fused_nodes[node_num[n]] = n;
      
      try {
        update_successors(n, node_num, cant_fuse_with, g, work_list,
                          type_list, types);
      } catch (const loop_error &e) {
        
        throw loop_error(
                         "statements cannot be fused together due to negative dependence");
        
      }
      //fused_nodes_counter++;
      
    }
    
  }
  
  return s;
}




void Loop::setLexicalOrder(int dim, const std::set<int> &active,
                           int starting_order, std::vector<std::vector<std::string> > idxNames) {
  debug_fprintf(stderr, "Loop::setLexicalOrder()  %d idxNames     active size %d  starting_order %d\n", idxNames.size(), active.size(), starting_order); 
  if (active.size() == 0)
    return;

  for (int i=0; i< idxNames.size(); i++) { 
    std::vector<std::string> what = idxNames[i];
    for (int j=0; j<what.size(); j++) { 
      debug_fprintf(stderr, "%2d %2d %s\n", i,j, what[j].c_str()); 
    }
  }

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
  
  // separate statements by current loop level types
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
      } else { // recurse ! 
        debug_fprintf(stderr, "Loop:setLexicalOrder() recursing\n"); 
        setLexicalOrder(dim, cur_scc, order, idxNames);
        order += sz;
      }
    }
  }
  else { // set lexical order separating single iteration statements and loops

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
        
        std::vector<bool> types;
        for (int i = 0; i < s.size(); i++)
          types.push_back(true);
        
        Graph<std::set<int>, bool> g = construct_induced_graph_at_level(
                                                                        s, dep, dep_dim);
        s = typed_fusion(g, types);
      }
      int order = starting_order;
      for (int i = 0; i < s.size(); i++) {
        
        for (std::set<int>::iterator it = s[i].begin();
             it != s[i].end(); it++) {
          assign_const(stmt[*it].xform, dim, order);
          stmt[*it].xform.simplify();
        }
        
        if ((dim + 2) <= (stmt[ref_stmt_num].xform.n_out() - 1)) {  // recurse ! 
          debug_fprintf(stderr, "Loop:setLexicalOrder() recursing\n"); 
          setLexicalOrder(dim + 2, s[i], order, idxNames);
        }
        
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
           i++) {
        assign_const(stmt[*i].xform, dim, dummy_order++);
        stmt[*i].xform.simplify();
      }
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

  debug_fprintf(stderr, "LEAVING Loop::setLexicalOrder()  %d idxNames\n", idxNames.size()); 
  for (int i=0; i< idxNames.size(); i++) { 
    std::vector<std::string> what = idxNames[i];
    for (int j=0; j<what.size(); j++) { 
      debug_fprintf(stderr, "%2d %2d %s\n", i,j, what[j].c_str()); 
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
  debug_fprintf(stderr, "apply_xform( %d )\n", stmt_num); 
  std::set<int> active;
  active.insert(stmt_num);
  apply_xform(active);
}

void Loop::apply_xform(std::set<int> &active) {
  debug_fprintf(stderr, "loop.cc apply_xform( set )\n");
  
  int max_n = 0;
  
  omega::CG_outputBuilder *ocg = ir->builder();
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].loop_level.size();
    if (n > max_n)
      max_n = n;
    
    std::vector<int> lex = getLexicalOrder(*i);
    
    omega::Relation mapping(2 * n + 1, n);
    omega::F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
      omega::EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(2 * j), -1);
    }
    mapping = omega::Composition(mapping, stmt[*i].xform);
    mapping.simplify();
    
    // match omega input/output variables to variable names in the code
    for (int j = 1; j <= stmt[*i].IS.n_set(); j++)
      mapping.name_input_var(j, stmt[*i].IS.set_var(j)->name());
    for (int j = 1; j <= n; j++)
      mapping.name_output_var(j,
                              tmp_loop_var_name_prefix
                              + omega::to_string(
                                                 tmp_loop_var_name_counter + j - 1));
    mapping.setup_names();
    debug_begin
      mapping.print();   //   "{[I] -> [_t1] : I = _t1 }
      fflush(stdout);
    debug_end
    
    omega::Relation known = Extend_Set(copy(this->known),
                                       mapping.n_out() - this->known.n_set());
    //stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out(), NULL));
    
    omega::CG_outputBuilder *ocgr = ir->builder(); 
    
    
    //this is probably CG_chillBuilder; 
    
    omega::CG_stringBuilder *ocgs = new omega::CG_stringBuilder;
    if (uninterpreted_symbols[*i].size() == 0) {
      
      
      std::set<std::string> globals;
      
      for (omega::DNF_Iterator di(stmt[*i].IS.query_DNF()); di; di++) {
        
        for (omega::Constraint_Iterator e(*di); e; e++) {
          for (omega::Constr_Vars_Iter cvi(*e); cvi; cvi++) {
            omega::Variable_ID v = cvi.curr_var();
            if (v->kind() == omega::Global_Var
                && v->get_global_var()->arity() > 0
                && globals.find(v->name()) == globals.end()) {
              omega::Global_Var_ID g = v->get_global_var();
              globals.insert(v->name());
              std::vector<omega::CG_outputRepr *> reprs;
              std::vector<omega::CG_outputRepr *> reprs2;
              
              for (int l = 1; l <= g->arity(); l++) {
                omega::CG_outputRepr *temp = ocgr->CreateIdent(
                                                               stmt[*i].IS.set_var(l)->name());
                omega::CG_outputRepr *temp2 = ocgs->CreateIdent(
                                                                stmt[*i].IS.set_var(l)->name());
                
                reprs.push_back(temp);
                reprs2.push_back(temp2);
              }
              uninterpreted_symbols[*i].insert(
                                     std::pair<std::string,
                                           std::vector<omega::CG_outputRepr *> >(
                                                                     (const char*)(v->get_global_var()->base_name()),
                                                                     reprs));
              uninterpreted_symbols_stringrepr[*i].insert(
                                     std::pair<std::string,
                                           std::vector<omega::CG_outputRepr *> >(
                                                                     (const char*)(v->get_global_var()->base_name()),
                                                                     reprs2));
            }
          }
        }
      }
    }
    
    std::vector<std::string> loop_vars;
    for (int j = 1; j <= stmt[*i].IS.n_set(); j++) {
      loop_vars.push_back(stmt[*i].IS.set_var(j)->name());
    }
    for (int j = 0; j<loop_vars.size(); j++) { 
      debug_fprintf(stderr, "loop vars %d %s\n", j, loop_vars[j].c_str()); 
    }
    std::vector<CG_outputRepr *> subs = output_substitutions(ocg,
                                                             Inverse(copy(mapping)),
                                                             std::vector<std::pair<CG_outputRepr *, int> >(
                                                                                                           mapping.n_out(),
                                                                                                           std::make_pair(
                                                                                                                          static_cast<CG_outputRepr *>(NULL), 0)),
                                                             uninterpreted_symbols[*i]);
    
    std::vector<CG_outputRepr *> subs2;
    for (int l = 0; l < subs.size(); l++)
      subs2.push_back(subs[l]->clone());
    
    debug_fprintf(stderr, "%d uninterpreted symbols\n", (int)uninterpreted_symbols.size());
    for (int j = 0; j<loop_vars.size(); j++) {
      debug_fprintf(stderr, "loop vars %d %s\n", j, loop_vars[j].c_str()); 
    } 
    
    
    int count = 0; 
    for (std::map<std::string, std::vector<CG_outputRepr *> >::iterator it =
           uninterpreted_symbols[*i].begin();
         it != uninterpreted_symbols[*i].end(); it++) {
      debug_fprintf(stderr, "\ncount %d\n", count); 
      
      std::vector<CG_outputRepr *> reprs_ = it->second;
      debug_fprintf(stderr, "%d reprs_\n", (int)reprs_.size()); 
      
      std::vector<CG_outputRepr *> reprs_2;
      for (int k = 0; k < reprs_.size(); k++) {
        debug_fprintf(stderr, "k %d\n", k); 
        std::vector<CG_outputRepr *> subs;
        for (int l = 0; l < subs2.size(); l++) {
          debug_fprintf(stderr, "l %d\n", l); 
          subs.push_back(subs2[l]->clone());
        }
        
        debug_fprintf(stderr, "clone\n");
        CG_outputRepr *c =  reprs_[k]->clone(); 
        c->dump(); fflush(stdout); 
        
        debug_fprintf(stderr, "createsub\n"); 
        CG_outputRepr *s = ocgr->CreateSubstitutedStmt(0, c,
                                                       loop_vars, subs, true);
        
        debug_fprintf(stderr, "push back\n"); 
        reprs_2.push_back( s ); 
        
      }
      
      it->second = reprs_2;
      count++;
      debug_fprintf(stderr, "bottom\n"); 
    }
     
    std::vector<CG_outputRepr *> subs3 = output_substitutions(
                                                              ocgs, Inverse(copy(mapping)),
                                                              std::vector<std::pair<CG_outputRepr *, int> >(
                                                                                                            mapping.n_out(),
                                                                                                            std::make_pair(
                                                                                                                           static_cast<CG_outputRepr *>(NULL), 0)),
                                                              uninterpreted_symbols_stringrepr[*i]);
    
    for (std::map<std::string, std::vector<CG_outputRepr *> >::iterator it =
           uninterpreted_symbols_stringrepr[*i].begin();
         it != uninterpreted_symbols_stringrepr[*i].end(); it++) {
      
      std::vector<CG_outputRepr *> reprs_ = it->second;
      std::vector<CG_outputRepr *> reprs_2;
      for (int k = 0; k < reprs_.size(); k++) {
        std::vector<CG_outputRepr *> subs;
        /*  for (int l = 0; l < subs3.size(); l++)
            subs.push_back(subs3[l]->clone());
            reprs_2.push_back(
            ocgs->CreateSubstitutedStmt(0, reprs_[k]->clone(),
            loop_vars, subs));
        */
        reprs_2.push_back(subs3[k]->clone());
      }
      
      it->second = reprs_2;
      
    }
    
    
    debug_fprintf(stderr, "loop.cc stmt[*i].code =\n"); 
    //stmt[*i].code->dump(); 
    //debug_fprintf(stderr, "\n"); 
    stmt[*i].code = ocg->CreateSubstitutedStmt(0, stmt[*i].code, loop_vars,
                                               subs);
    //debug_fprintf(stderr, "loop.cc substituted code =\n"); 
    //stmt[*i].code->dump(); 
    //debug_fprintf(stderr, "\n"); 
    
    stmt[*i].IS = omega::Range(Restrict_Domain(mapping, stmt[*i].IS));
    stmt[*i].IS.simplify();
    
    // replace original transformation relation with straight 1-1 mapping
    //debug_fprintf(stderr, "replace original transformation relation with straight 1-1 mapping\n"); 
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
    
    //debug_fprintf(stderr, "\ncode is: \n"); 
    //stmt[*i].code->dump(); 
    //debug_fprintf(stderr, "\n\n"); 
    
  }
  
  tmp_loop_var_name_counter += max_n;
  fflush(stdout); 
  debug_fprintf(stderr, "loop.cc LEAVING apply_xform( set )\n\n"); 
  //for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
  //  debug_fprintf(stderr, "\nloop.cc stmt[i].code =\n"); 
  //  stmt[*i].code->dump(); 
  //  debug_fprintf(stderr, "\n\n"); 
  //} 
  
}




void Loop::addKnown(const Relation &cond) {
  
  // invalidate saved codegen computation
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  debug_fprintf(stderr, "Loop::addKnown(), SETTING last_compute_cg_ = NULL\n");
  
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
  debug_fprintf(stderr, "Loop::nonsingular(), SETTING last_compute_cg_ = NULL\n");

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

// Manu:: reduction operation

void Loop::scalar_expand(int stmt_num, const std::vector<int> &levels,
                         std::string arrName, int memory_type, int padding_alignment,
                         int assign_then_accumulate, int padding_stride) {
   
  //std::cout << "In scalar_expand function: " << stmt_num << ", " << arrName << "\n";
  //std::cout.flush(); 

  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  // check for sanity of parameters
  bool found_non_constant_size_dimension = false;
  
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument(
                                "invalid statement number " + to_string(stmt_num));
  //Anand: adding check for privatized levels
  //if (arrName != "RHS")
  //  throw std::invalid_argument(
  //      "invalid 3rd argument: only 'RHS' supported " + arrName);
  for (int i = 0; i < levels.size(); i++) {
    if (levels[i] <= 0 || levels[i] > stmt[stmt_num].loop_level.size())
      throw std::invalid_argument(
                                  "1invalid loop level " + to_string(levels[i]));
    
    if (i > 0) {
      if (levels[i] < levels[i - 1])
        throw std::invalid_argument(
                                    "loop levels must be in ascending order");
    }
  }
  //end --adding check for privatized levels
  
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  debug_fprintf(stderr, "Loop::scalar_expand(), SETTING last_compute_cg_ = NULL\n");

  debug_fprintf(stderr, "\nloop.cc finding array accesses in stmt %d of the code\n",stmt_num ); 
  std::vector<IR_ArrayRef *> access = ir->FindArrayRef(stmt[stmt_num].code);
  debug_fprintf(stderr, "loop.cc L2726  %d access\n", access.size()); 

  IR_ArraySymbol *sym = NULL;
  debug_fprintf(stderr, "arrName %s\n", arrName.c_str()); 
  if (arrName == "RHS") { 
    debug_fprintf(stderr, "sym RHS\n"); 
    sym = access[0]->symbol();
  }
  else {
    debug_fprintf(stderr, "looking for array %s in access\n", arrName.c_str()); 
    for (int k = 0; k < access.size(); k++) { // BUH

      //debug_fprintf(stderr, "access[%d] = %s ", k, access[k]->getTypeString()); access[k]->print(0,stderr); debug_fprintf(stderr, "\n"); 

      string name = access[k]->symbol()->name();
      //debug_fprintf(stderr, "comparing %s to %s\n", name.c_str(), arrName.c_str()); 

      if (access[k]->symbol()->name() == arrName) {
        debug_fprintf(stderr, "found it   sym access[ k=%d ]\n", k); 
        sym = access[k]->symbol();
      }      
    }
  }
  if (!sym) debug_fprintf(stderr, "DIDN'T FIND IT\n"); 
  debug_fprintf(stderr, "sym %p\n", sym); 

  // collect array references by name
  std::vector<int> lex = getLexicalOrder(stmt_num);
  int dim = 2 * levels[levels.size() - 1] - 1;
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  //Anand: shifting this down
  //  assign_const(stmt[newStmt_num].xform, 2*level+1, 1);
  
  //  std::cout << " before temp array name \n ";
  // create a temporary variable
  IR_Symbol *tmp_sym;
  
  // get the loop upperbound, that would be the size of the temp array.
  omega::coef_t lb[levels.size()], ub[levels.size()], size[levels.size()];
  
  //Anand Adding apply xform so that tiled loop bounds are reflected
  debug_fprintf(stderr, "Adding apply xform so that tiled loop bounds are reflected\n");
  apply_xform(same_loop);
  debug_fprintf(stderr, "loop.cc, back from apply_xform()\n"); 
  
  //Anand commenting out the folowing 4 lines
  /*  copy(stmt[stmt_num].IS).query_variable_bounds(
      copy(stmt[stmt_num].IS).set_var(level), lb, ub);
      std::cout << "Upper Bound = " << ub << "\n";
      std::cout << "lower Bound = " << lb << "\n";
  */
  // testing testing -- Manu ////////////////////////////////////////////////
  /*
  // int n_dim = sym->n_dim();
  // std::cout << "------- n_dim ----------- " << n_dim << "\n";
  std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(stmt[stmt_num].IS, stmt[stmt_num].IS.set_var(level));
  omega::coef_t  index_stride;
  if (result.second != NULL) {
  index_stride = abs(result.first.get_coef(result.second))/gcd(abs(result.first.get_coef(result.second)), abs(result.first.get_coef(stmt[stmt_num].IS.set_var(level))));
  std::cout << "simplest_stride :: " << index_stride << ", " << result.first.get_coef(result.second) << ", " << result.first.get_coef(stmt[stmt_num].IS.set_var(level))<< "\n";
  }
  Relation bound;
  // bound = get_loop_bound(stmt[stmt_num].IS, level);
  bound = SimpleHull(stmt[stmt_num].IS,true, true);
  bound.print();
  
  bound = copy(stmt[stmt_num].IS);
  for (int i = 1; i < level; i++) {
  bound = Project(bound, i, Set_Var);
  std::cout << "-------------------------------\n";
  bound.print();
  }
  
  bound.simplify();
  bound.print();
  // bound = get_loop_bound(bound, level);
  
  copy(bound).query_variable_bounds(copy(bound).set_var(level), lb, ub);
  std::cout << "Upper Bound = " << ub << "\n";
  std::cout << "lower Bound = " << lb << "\n";
  
  result = find_simplest_stride(bound, bound.set_var(level));
  if (result.second != NULL)
  index_stride = abs(result.first.get_coef(result.second))/gcd(abs(result.first.get_coef(result.second)), abs(result.first.get_coef(bound.set_var(level))));
  else
  index_stride = 1;
  std::cout << "simplest_stride 11:: " << index_stride << "\n";
  */
  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// copied datacopy code here /////////////////////////////////////////////

  //std::cout << "In scalar_expand function 2: " << stmt_num << ", " << arrName << "\n";
  //std::cout.flush(); 

  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 



  int n_dim = levels.size();
  Relation copy_is = copy(stmt[stmt_num].IS);
  // extract temporary array information
  CG_outputBuilder *ocg1 = ir->builder();
  std::vector<CG_outputRepr *> index_lb(n_dim); // initialized to NULL
  std::vector<coef_t> index_stride(n_dim);
  std::vector<bool> is_index_eq(n_dim, false);
  std::vector<std::pair<int, CG_outputRepr *> > index_sz(0);
  Relation reduced_copy_is = copy(copy_is);
  std::vector<CG_outputRepr *> size_repr;
  std::vector<int> size_int;
  Relation xform = copy(stmt[stmt_num].xform);
  for (int i = 0; i < n_dim; i++) {
    
    dim = 2 * levels[i] - 1;
    //Anand: Commenting out the lines below: not required
    //    if (i != 0)
    //    reduced_copy_is = Project(reduced_copy_is, level - 1 + i, Set_Var);
    Relation bound = get_loop_bound(copy(reduced_copy_is), levels[i] - 1);
    
    // extract stride
    std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(bound,
                                                                    bound.set_var(levels[i]));
    if (result.second != NULL)
      index_stride[i] = abs(result.first.get_coef(result.second))
        / gcd(abs(result.first.get_coef(result.second)),
              abs(
                  result.first.get_coef(
                                        bound.set_var(levels[i]))));
    else
      index_stride[i] = 1;
    //  std::cout << "simplest_stride 11:: " << index_stride[i] << "\n";
    
    // check if this array index requires loop
    Conjunct *c = bound.query_DNF()->single_conjunct();
    for (EQ_Iterator ei(c->EQs()); ei; ei++) {
      if ((*ei).has_wildcards())
        continue;
      
      int coef = (*ei).get_coef(bound.set_var(levels[i]));
      if (coef != 0) {
        int sign = 1;
        if (coef < 0) {
          coef = -coef;
          sign = -1;
        }
        
        CG_outputRepr *op = NULL;
        for (Constr_Vars_Iter ci(*ei); ci; ci++) {
          switch ((*ci).var->kind()) {
          case Input_Var: {
            if ((*ci).var != bound.set_var(levels[i]))
              if ((*ci).coef * sign == 1)
                op = ocg1->CreateMinus(op,
                                       ocg1->CreateIdent((*ci).var->name()));
              else if ((*ci).coef * sign == -1)
                op = ocg1->CreatePlus(op,
                                      ocg1->CreateIdent((*ci).var->name()));
              else if ((*ci).coef * sign > 1) { 
                op = ocg1->CreateMinus(op,
                                       ocg1->CreateTimes(
                                                         ocg1->CreateInt(
                                                                         abs((*ci).coef)),
                                                         ocg1->CreateIdent(
                                                                           (*ci).var->name())));
              }
              else
                // (*ci).coef*sign < -1
                op = ocg1->CreatePlus(op,
                                      ocg1->CreateTimes(
                                                        ocg1->CreateInt(
                                                                        abs((*ci).coef)),
                                                        ocg1->CreateIdent(
                                                                          (*ci).var->name())));
            break;
          }
          case Global_Var: {
            Global_Var_ID g = (*ci).var->get_global_var();
            if ((*ci).coef * sign == 1)
              op = ocg1->CreateMinus(op,
                                     ocg1->CreateIdent(g->base_name()));
            else if ((*ci).coef * sign == -1)
              op = ocg1->CreatePlus(op,
                                    ocg1->CreateIdent(g->base_name()));
            else if ((*ci).coef * sign > 1)
              op = ocg1->CreateMinus(op,
                                     ocg1->CreateTimes(
                                                       ocg1->CreateInt(abs((*ci).coef)),
                                                       ocg1->CreateIdent(g->base_name())));
            else
              // (*ci).coef*sign < -1
              op = ocg1->CreatePlus(op,
                                    ocg1->CreateTimes(
                                                      ocg1->CreateInt(abs((*ci).coef)),
                                                      ocg1->CreateIdent(g->base_name())));
            break;
          }
          default:
            throw loop_error("unsupported array index expression");
          }
        }
        if ((*ei).get_const() != 0)
          op = ocg1->CreatePlus(op,
                                ocg1->CreateInt(-sign * ((*ei).get_const())));
        if (coef != 1)
          op = ocg1->CreateIntegerFloor(op, ocg1->CreateInt(coef));
        
        index_lb[i] = op;
        is_index_eq[i] = true;
        break;
      }
    }
    if (is_index_eq[i])
      continue;
    
    // separate lower and upper bounds
    std::vector<GEQ_Handle> lb_list, ub_list;
    std::set<Variable_ID> excluded_floor_vars;
    excluded_floor_vars.insert(bound.set_var(levels[i]));
    for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
      int coef = (*gi).get_coef(bound.set_var(levels[i]));
      if (coef != 0 && (*gi).has_wildcards()) {
        bool clean_bound = true;
        GEQ_Handle h;
        for (Constr_Vars_Iter cvi(*gi, true); gi; gi++)
          if (!find_floor_definition(bound, (*cvi).var,
                                     excluded_floor_vars).first) {
            clean_bound = false;
            break;
          }
          else 
            h= find_floor_definition(bound, (*cvi).var,
                                     excluded_floor_vars).second;
        
        if (!clean_bound)
          continue;
        else{
          if (coef > 0)
            lb_list.push_back(h);
          else if (coef < 0)
            ub_list.push_back(h);
          continue;  
        }    
        
      }
      
      if (coef > 0)
        lb_list.push_back(*gi);
      else if (coef < 0)
        ub_list.push_back(*gi);
    }
    if (lb_list.size() == 0 || ub_list.size() == 0)
      throw loop_error("failed to calcuate array footprint size");
    
    // build lower bound representation
    std::vector<CG_outputRepr *> lb_repr_list;
    /*     for (int j = 0; j < lb_list.size(); j++){
           if(this->known.n_set() == 0)
           lb_repr_list.push_back(output_lower_bound_repr(ocg1, lb_list[j], bound.set_var(level-1+i+1), result.first, result.second, bound, Relation::True(bound.n_set()), std::vector<std::pair<CG_outputRepr *, int> >(bound.n_set(), std::make_pair(static_cast<CG_outputRepr *>(NULL), 0))));
           else
           lb_repr_list.push_back(output_lower_bound_repr(ocg1, lb_list[j], bound.set_var(level-1+i+1), result.first, result.second, bound, this->known, std::vector<std::pair<CG_outputRepr *, int> >(bound.n_set(), std::make_pair(static_cast<CG_outputRepr *>(NULL), 0))));
           
           }
    */
    if (lb_repr_list.size() > 1)
      index_lb[i] = ocg1->CreateInvoke("max", lb_repr_list);
    else if (lb_repr_list.size() == 1)
      index_lb[i] = lb_repr_list[0];
    
    // build temporary array size representation
    {
      Relation cal(copy_is.n_set(), 1);
      F_And *f_root = cal.add_and();
      for (int j = 0; j < ub_list.size(); j++)
        for (int k = 0; k < lb_list.size(); k++) {
          GEQ_Handle h = f_root->add_GEQ();
          
          for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              h.update_coef(cal.input_var(pos), (*ci).coef);
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = cal.get_local(g);
              else
                v = cal.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error(
                               "cannot calculate temporay array size statically");
            }
          }
          h.update_const(ub_list[j].get_const());
          
          for (Constr_Vars_Iter ci(lb_list[k]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var: {
              int pos = (*ci).var->get_position();
              h.update_coef(cal.input_var(pos), (*ci).coef);
              break;
            }
            case Global_Var: {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = cal.get_local(g);
              else
                v = cal.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error(
                               "cannot calculate temporay array size statically");
            }
          }
          h.update_const(lb_list[k].get_const());
          
          h.update_const(1);
          h.update_coef(cal.output_var(1), -1);
        }
      
      cal = Restrict_Domain(cal, copy(copy_is));
      for (int j = 1; j <= cal.n_inp(); j++) {
        cal = Project(cal, j, Input_Var);
      }
      cal.simplify();
      
      // pad temporary array size
      // TODO: for variable array size, create padding formula
      //int padding_stride = 0;
      Conjunct *c = cal.query_DNF()->single_conjunct();
      bool is_index_bound_const = false;
      if (padding_stride != 0 && i == n_dim - 1) {
        //size = (size + index_stride[i] - 1) / index_stride[i];
        size_repr.push_back(ocg1->CreateInt(padding_stride));
      } else {
        for (GEQ_Iterator gi(c->GEQs()); gi && !is_index_bound_const;
             gi++)
          if ((*gi).is_const(cal.output_var(1))) {
            coef_t size = (*gi).get_const()
              / (-(*gi).get_coef(cal.output_var(1)));
            
            if (padding_alignment > 1 && i == n_dim - 1) { // align to boundary for data packing
              int residue = size % padding_alignment;
              if (residue)
                size = size + padding_alignment - residue;
            }
            
            index_sz.push_back(
                               std::make_pair(i, ocg1->CreateInt(size)));
            is_index_bound_const = true;
            size_int.push_back(size);
            size_repr.push_back(ocg1->CreateInt(size));
            
            //  std::cout << "============================== size :: "
            //      << size << "\n";
            
          }
        
        if (!is_index_bound_const) {
          
          found_non_constant_size_dimension = true;
          Conjunct *c = bound.query_DNF()->single_conjunct();
          for (GEQ_Iterator gi(c->GEQs());
               gi && !is_index_bound_const; gi++) {
            int coef = (*gi).get_coef(bound.set_var(levels[i]));
            if (coef < 0) {
              
              size_repr.push_back(
                                  ocg1->CreatePlus(
                                                   output_upper_bound_repr(ocg1, *gi,
                                                                           bound.set_var(levels[i]),
                                                                           bound,
                                                                           std::vector<
                                                                             std::pair<
                                                                               CG_outputRepr *,
                                                                               int> >(
                                                                                      bound.n_set(),
                                                                                      std::make_pair(
                                                                                                     static_cast<CG_outputRepr *>(NULL),
                                                                                                     0)),
                                                                           uninterpreted_symbols[stmt_num]),
                                                   ocg1->CreateInt(1)));
              
              /*CG_outputRepr *op = NULL;
                for (Constr_Vars_Iter ci(*gi); ci; ci++) {
                if ((*ci).var != cal.output_var(1)) {
                switch ((*ci).var->kind()) {
                case Global_Var: {
                Global_Var_ID g =
                (*ci).var->get_global_var();
                if ((*ci).coef == 1)
                op = ocg1->CreatePlus(op,
                ocg1->CreateIdent(
                g->base_name()));
                else if ((*ci).coef == -1)
                op = ocg1->CreateMinus(op,
                ocg1->CreateIdent(
                g->base_name()));
                else if ((*ci).coef > 1)
                op =
                ocg1->CreatePlus(op,
                ocg1->CreateTimes(
                ocg1->CreateInt(
                (*ci).coef),
                ocg1->CreateIdent(
                g->base_name())));
                else
                // (*ci).coef < -1
                op =
                ocg1->CreateMinus(op,
                ocg1->CreateTimes(
                ocg1->CreateInt(
                -(*ci).coef),
                ocg1->CreateIdent(
                g->base_name())));
                break;
                }
                default:
                throw loop_error(
                "failed to generate array index bound code");
                }
                }
                }
                int c = (*gi).get_const();
                if (c > 0)
                op = ocg1->CreatePlus(op, ocg1->CreateInt(c));
                else if (c < 0)
                op = ocg1->CreateMinus(op, ocg1->CreateInt(-c));
              */
              /*            if (padding_stride != 0) {
                            if (i == fastest_changing_dimension) {
                            coef_t g = gcd(index_stride[i], static_cast<coef_t>(padding_stride));
                            coef_t t1 = index_stride[i] / g;
                            if (t1 != 1)
                            op = ocg->CreateIntegerFloor(ocg->CreatePlus(op, ocg->CreateInt(t1-1)), ocg->CreateInt(t1));
                            coef_t t2 = padding_stride / g;
                            if (t2 != 1)
                            op = ocg->CreateTimes(op, ocg->CreateInt(t2));
                            }
                            else if (index_stride[i] != 1) {
                            op = ocg->CreateIntegerFloor(ocg->CreatePlus(op, ocg->CreateInt(index_stride[i]-1)), ocg->CreateInt(index_stride[i]));
                            }
                            }
              */
              //index_sz.push_back(std::make_pair(i, op));
              //break;
            }
          }
        }
      }
    }
    //size[i] = ub[i];
    
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  
  //Anand: Creating IS of new statement
  
  //for(int l = dim; l < stmt[stmt_num].xform.n_out(); l+=2)
  //std::cout << "In scalar_expand function 3: " << stmt_num << ", " << arrName << "\n";
  //std::cout.flush(); 

  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  
  shiftLexicalOrder(lex, dim + 1, 1);
  Statement s = stmt[stmt_num];
  s.ir_stmt_node = NULL;
  int newStmt_num = stmt.size();

  debug_fprintf(stderr, "loop.cc L3249 adding stmt %d\n", stmt.size()); 
  stmt.push_back(s);
  
  debug_fprintf(stderr, "uninterpreted_symbols.push_back()  newStmt_num %d\n", newStmt_num); 
  uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
  uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
  stmt[newStmt_num].code = stmt[stmt_num].code->clone();
  stmt[newStmt_num].IS = copy(stmt[stmt_num].IS);
  stmt[newStmt_num].xform = xform;
  stmt[newStmt_num].reduction = stmt[stmt_num].reduction;
  stmt[newStmt_num].reductionOp = stmt[stmt_num].reductionOp;


  //debug_fprintf(stderr, "\nafter clone, %d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 



  //assign_const(stmt[newStmt_num].xform, stmt[stmt_num].xform.n_out(), 1);//Anand: change from 2*level + 1 to stmt[stmt_num].xform.size()
  //Anand-End creating IS of new statement
  
  CG_outputRepr * tmpArrSz;
  CG_outputBuilder *ocg = ir->builder();
  
  //for(int k =0; k < levels.size(); k++ )
  //    size_repr.push_back(ocg->CreateInt(size[k]));//Anand: copying apply_xform functionality to prevent IS modification
  //due to side effects with uninterpreted function symbols and failures in omega
  
  //int n = stmt[stmt_num].loop_level.size();
  
  /*Relation mapping(2 * n + 1, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(j), 1);
    h.update_coef(mapping.input_var(2 * j), -1);
    }
    mapping = Composition(mapping, copy(stmt[stmt_num].xform));
    mapping.simplify();
    
    // match omega input/output variables to variable names in the code
    for (int j = 1; j <= stmt[stmt_num].IS.n_set(); j++)
    mapping.name_input_var(j, stmt[stmt_num].IS.set_var(j)->name());
    for (int j = 1; j <= n; j++)
    mapping.name_output_var(j,
    tmp_loop_var_name_prefix
    + to_string(tmp_loop_var_name_counter + j - 1));
    mapping.setup_names();
    
    Relation size_ = omega::Range(Restrict_Domain(mapping, copy(stmt[stmt_num].IS)));
    size_.simplify();
  */
  
  //Anand -commenting out tmp sym creation as symbol may have more than one dimension
  //tmp_sym = ir->CreateArraySymbol(tmpArrSz, sym);
  std::vector<CG_outputRepr *> lhs_index;
  CG_outputRepr *arr_ref_repr;
  arr_ref_repr = ocg->CreateIdent(
                                  stmt[stmt_num].IS.set_var(levels[levels.size() - 1])->name());
  
  CG_outputRepr *total_size = size_repr[0];
  debug_fprintf(stderr, "total_size = ");   total_size->dump(); fflush(stdout); 

  for (int i = 1; i < size_repr.size(); i++) {
    debug_fprintf(stderr, "total_size now "); total_size->dump(); fflush(stdout); debug_fprintf(stderr, " times  something\n\n"); 

    total_size = ocg->CreateTimes(total_size->clone(),
                                  size_repr[i]->clone());
    
  }
  
  // COMMENT NEEDED 
  //debug_fprintf(stderr, "\nloop.cc COMMENT NEEDED\n"); 
  for (int k = levels.size() - 2; k >= 0; k--) {
    CG_outputRepr *temp_repr =ocg->CreateIdent(stmt[stmt_num].IS.set_var(levels[k])->name());
    for (int l = k + 1; l < levels.size(); l++) { 
      //debug_fprintf(stderr, "\nloop.cc CREATETIMES\n"); 
      temp_repr = ocg->CreateTimes(temp_repr->clone(),
                                   size_repr[l]->clone());
    }
    
    //debug_fprintf(stderr, "\nloop.cc CREATEPLUS\n"); 
    arr_ref_repr = ocg->CreatePlus(arr_ref_repr->clone(),
                                   temp_repr->clone());
  }
  

  //debug_fprintf(stderr, "loop.cc, about to die\n"); 
  std::vector<CG_outputRepr *> to_push;
  to_push.push_back(total_size);

  if (!found_non_constant_size_dimension) { 
    debug_fprintf(stderr, "constant size dimension\n"); 
    tmp_sym = ir->CreateArraySymbol(sym, to_push, memory_type);
  }
  else {
    debug_fprintf(stderr, "NON constant size dimension?\n"); 
    //tmp_sym = ir->CreatePointerSymbol(sym, to_push);
    tmp_sym = ir->CreatePointerSymbol(sym, to_push);

    static_cast<IR_PointerSymbol *>(tmp_sym)->set_size(0, total_size); // ?? 
    ptr_variables.push_back(static_cast<IR_PointerSymbol *>(tmp_sym));
    debug_fprintf(stderr, "ptr_variables now has %d entries\n", ptr_variables.size()); 
  }
  
  // add tmp_sym to Loop symtables ??
  

  //  std::cout << " temp array name == " << tmp_sym->name().c_str() << "\n";
  
  // get loop index variable at the given "level"
  // Relation R = omega::Range(Restrict_Domain(copy(stmt[stmt_num].xform), copy(stmt[stmt_num].IS)));
  //  stmt[stmt_num].IS.print();
  //stmt[stmt_num].IS.
  //  std::cout << stmt[stmt_num].IS.n_set() << std::endl;
  //  std::string v = stmt[stmt_num].IS.set_var(level)->name();
  //  std::cout << "loop index variable is '" << v.c_str() << "'\n";
  
  // create a reference for the temporary array
  debug_fprintf(stderr, "create a reference for the temporary array\n"); 
  //std::cout << "In scalar_expand function 4: " << stmt_num << ", " << arrName << "\n";
  //std::cout.flush(); 

  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  

  std::vector<CG_outputRepr *> to_push2;
  to_push2.push_back(arr_ref_repr); // can have only one entry

  //lhs_index[0] = ocg->CreateIdent(v);


  IR_ArrayRef *tmp_array_ref;
  IR_PointerArrayRef * tmp_ptr_array_ref;  // was IR_PointerArrayref

  if (!found_non_constant_size_dimension) {
    debug_fprintf(stderr, "constant size\n");

    tmp_array_ref = ir->CreateArrayRef(
                                       static_cast<IR_ArraySymbol *>(tmp_sym), to_push2);
  }
  else { 
    debug_fprintf(stderr, "NON constant size\n"); 
    tmp_ptr_array_ref = ir->CreatePointerArrayRef(
                                       static_cast<IR_PointerSymbol *>(tmp_sym), to_push2);
    // TODO static_cast<IR_PointerSymbol *>(tmp_sym), to_push2);
  }
  fflush(stdout); 

  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 


  //std::string stemp;
  //stemp = tmp_array_ref->name();
  //std::cout << "Created array reference --> " << stemp.c_str() << "\n";
  
  // get the RHS expression
  debug_fprintf(stderr, "get the RHS expression   arrName %s\n", arrName.c_str()); 

  CG_outputRepr *rhs;
  if (arrName == "RHS") {
    rhs = ir->GetRHSExpression(stmt[stmt_num].code);
    
    std::vector<IR_ArrayRef *> symbols = ir->FindArrayRef(rhs);
  }
  std::set<std::string> sym_names;
  
  //for (int i = 0; i < symbols.size(); i++)
  //  sym_names.insert(symbols[i]->symbol()->name());
  
  fflush(stdout); 

  //debug_fprintf(stderr, "\nbefore if (arrName == RHS)\n%d statements\n", stmt.size()); // problem is after here 
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  if (arrName == "RHS") {
    
    std::vector<IR_ArrayRef *> symbols = ir->FindArrayRef(rhs);
    
    for (int i = 0; i < symbols.size(); i++)
      sym_names.insert(symbols[i]->symbol()->name());
  } 
  else {

    debug_fprintf(stderr, "finding array refs in stmt_num %d\n", stmt_num); 
    //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
    //for (int i=0; i<stmt.size(); i++) { 
    //  debug_fprintf(stderr, "%2d   ", i); 
    //  ((CG_chillRepr *)stmt[i].code)->Dump();
    //} 
    //debug_fprintf(stderr, "\n"); 

    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[stmt_num].code);
    debug_fprintf(stderr, "\n%d refs\n", refs.size()); 

    
    bool found = false;

    for (int j = 0; j < refs.size(); j++) {
      CG_outputRepr* to_replace;

      debug_fprintf(stderr, "j %d   build new assignment statement with temporary array\n",j); 
      // build new assignment statement with temporary array
      if (!found_non_constant_size_dimension) {
        to_replace = tmp_array_ref->convert();
      } else {
        to_replace = tmp_ptr_array_ref->convert();
      }
      //debug_fprintf(stderr, "to_replace  %p\n", to_replace); 
      //CG_chillRepr *CR = (CG_chillRepr *) to_replace;
      //CR->Dump(); 

      if (refs[j]->name() == arrName) {
        fflush(stdout); 
        debug_fprintf(stderr, "loop.cc L353\n");  // problem is after here 
        //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
        //for (int i=0; i<stmt.size(); i++) { 
        //  debug_fprintf(stderr, "%2d   ", i); 
        //  ((CG_chillRepr *)stmt[i].code)->Dump();
        //} 
        //debug_fprintf(stderr, "\n"); 
        

        sym_names.insert(refs[j]->symbol()->name());
        
        if (!found) {
          if (!found_non_constant_size_dimension) { 
            debug_fprintf(stderr, "constant size2\n"); 
            omega::CG_outputRepr * t =  tmp_array_ref->convert();
            omega::CG_outputRepr * r = refs[j]->convert()->clone();
            //CR = (CG_chillRepr *) t;
            //CR->Dump(); 
            //CR = (CG_chillRepr *) r;
            //CR->Dump(); 

            //debug_fprintf(stderr, "lhs t %p   lhs r %p\n", t, r); 
            stmt[newStmt_num].code =
              ir->builder()->CreateAssignment(0,
                                              t, // tmp_array_ref->convert(),
                                              r); // refs[j]->convert()->clone()
          }
          else { 
            debug_fprintf(stderr, "NON constant size2\n"); 
            omega::CG_outputRepr * t =  tmp_ptr_array_ref->convert(); // this fails
            omega::CG_outputRepr * r = refs[j]->convert()->clone();

            //omega::CG_chillRepr *CR = (omega::CG_chillRepr *) t;
            //CR->Dump(); 
            //CR = (omega::CG_chillRepr *) r;
            //CR->Dump(); 

            //debug_fprintf(stderr, "lhs t %p   lhs r %p\n", t, r); 
            stmt[newStmt_num].code =
              ir->builder()->CreateAssignment(0,
                                              t, // tmp_ptr_array_ref->convert(),
                                              r ); // refs[j]->convert()->clone());
          }
          found = true;
          
        }
        
        // refs[j] has no parent?
        debug_fprintf(stderr, "replacing refs[%d]\n", j ); 
        ir->ReplaceExpression(refs[j], to_replace);
      }
      
    }
    
  }
  //ToDo need to update the dependence graph
  //Anand adding dependence graph update
  debug_fprintf(stderr, "adding dependence graph update\n");   // problem is before here 
  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  dep.insert();
  
  //Anand:Copying Dependence checks from datacopy code, might need to be a separate function/module
  // in the future
  
  /*for (int i = 0; i < newStmt_num; i++) {
    std::vector<std::vector<DependenceVector> > D;
    
    for (DependenceGraph::EdgeList::iterator j =
    dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();
    ) {
    if (same_loop.find(i) != same_loop.end()
    && same_loop.find(j->first) == same_loop.end()) {
    std::vector<DependenceVector> dvs1, dvs2;
    for (int k = 0; k < j->second.size(); k++) {
    DependenceVector dv = j->second[k];
    if (dv.sym != NULL
    && sym_names.find(dv.sym->name()) != sym_names.end()
    && (dv.type == DEP_R2R || dv.type == DEP_R2W))
    dvs1.push_back(dv);
    else
    dvs2.push_back(dv);
    }
    j->second = dvs2;
    if (dvs1.size() > 0)
    dep.connect(newStmt_num, j->first, dvs1);
    } else if (same_loop.find(i) == same_loop.end()
    && same_loop.find(j->first) != same_loop.end()) {
    std::vector<DependenceVector> dvs1, dvs2;
    for (int k = 0; k < j->second.size(); k++) {
    DependenceVector dv = j->second[k];
    if (dv.sym != NULL
    && sym_names.find(dv.sym->name()) != sym_names.end()
    && (dv.type == DEP_R2R || dv.type == DEP_W2R))
    dvs1.push_back(dv);
    else
    dvs2.push_back(dv);
    }
    j->second = dvs2;
    if (dvs1.size() > 0)
    D.push_back(dvs1);
    }
    
    if (j->second.size() == 0)
    dep.vertex[i].second.erase(j++);
    else
    j++;
    }
    
    for (int j = 0; j < D.size(); j++)
    dep.connect(i, newStmt_num, D[j]);
    }
  */
  //Anand--end dependence check
  if (arrName == "RHS") {
    
    // build new assignment statement with temporary array
    if (!found_non_constant_size_dimension) {
      if (assign_then_accumulate) {
        stmt[newStmt_num].code = ir->builder()->CreateAssignment(0,
                                                                 tmp_array_ref->convert(), rhs);
        debug_fprintf(stderr, "ir->ReplaceRHSExpression( stmt_ num %d )\n", stmt_num); 
        ir->ReplaceRHSExpression(stmt[stmt_num].code, tmp_array_ref);
      } else {
        CG_outputRepr *temp = tmp_array_ref->convert()->clone();
        if (ir->QueryExpOperation(stmt[stmt_num].code)
            != IR_OP_PLUS_ASSIGNMENT)
          throw ir_error(
                         "Statement is not a += accumulation statement");

        debug_fprintf(stderr, "replacing in a +=\n"); 
        stmt[newStmt_num].code = ir->builder()->CreatePlusAssignment(0,
                                                                     temp->clone(), rhs);
        
        CG_outputRepr * lhs = ir->GetLHSExpression(stmt[stmt_num].code);
        
        CG_outputRepr *assignment = ir->builder()->CreateAssignment(0,
                                                                    lhs, temp->clone());
        Statement init_ = stmt[newStmt_num]; // copy ??
        init_.ir_stmt_node = NULL;
        
        init_.code = stmt[newStmt_num].code->clone();
        init_.IS = copy(stmt[newStmt_num].IS);
        init_.xform = copy(stmt[newStmt_num].xform);
        init_.has_inspector = false; // ?? 

        Relation mapping(init_.IS.n_set(), init_.IS.n_set());
        
        F_And *f_root = mapping.add_and();
        
        for (int i = 1; i <= mapping.n_inp(); i++) {
          EQ_Handle h = f_root->add_EQ();
          //if (i < levels[0]) {
          if (i <= levels[levels.size() - 1]) {
            h.update_coef(mapping.input_var(i), 1);
            h.update_coef(mapping.output_var(i), -1);
          } else {
            h.update_const(-1);
            h.update_coef(mapping.output_var(i), 1);
          }
          
          /*else {
            int j;
            for (j = 0; j < levels.size(); j++)
            if (i == levels[j])
            break;
            
            if (j == levels.size()) {
            
            h.update_coef(mapping.output_var(i), 1);
            h.update_const(-1);
            
            } else {
            
            
            h.update_coef(mapping.input_var(i), 1);
            h.update_coef(mapping.output_var(i), -1);
            
            
            }
          */
          //}
        }
        
        mapping.simplify();
        // match omega input/output variables to variable names in the code
        for (int j = 1; j <= init_.IS.n_set(); j++)
          mapping.name_output_var(j, init_.IS.set_var(j)->name());
        for (int j = 1; j <= init_.IS.n_set(); j++)
          mapping.name_input_var(j, init_.IS.set_var(j)->name());
        
        mapping.setup_names();
        
        init_.IS = omega::Range(
                                omega::Restrict_Domain(mapping, init_.IS));
        std::vector<int> lex = getLexicalOrder(newStmt_num);
        int dim = 2 * levels[0] - 1;
        //init_.IS.print();
        //  init_.xform.print();
        //stmt[newStmt_num].xform.print();
        //  shiftLexicalOrder(lex, dim + 1, 1);
        shiftLexicalOrder(lex, dim + 1, 1);
        init_.reduction = stmt[newStmt_num].reduction;
        init_.reductionOp = stmt[newStmt_num].reductionOp;
        
        init_.code = ir->builder()->CreateAssignment(0, temp->clone(),
                                                     ir->builder()->CreateInt(0));

        debug_fprintf(stderr, "loop.cc L3693 adding stmt %d\n", stmt.size()); 
        stmt.push_back(init_);

        uninterpreted_symbols.push_back(uninterpreted_symbols[newStmt_num]);
        uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[newStmt_num]);
        stmt[stmt_num].code = assignment;
      }
    } else {
      if (assign_then_accumulate) {
        stmt[newStmt_num].code = ir->builder()->CreateAssignment(0,
                                                                 tmp_ptr_array_ref->convert(), rhs);
        ir->ReplaceRHSExpression(stmt[stmt_num].code,
                                 tmp_ptr_array_ref);
      } else {
        CG_outputRepr *temp = tmp_ptr_array_ref->convert()->clone();
        if (ir->QueryExpOperation(stmt[stmt_num].code)
            != IR_OP_PLUS_ASSIGNMENT)
          throw ir_error(
                         "Statement is not a += accumulation statement");
        stmt[newStmt_num].code = ir->builder()->CreatePlusAssignment(0,
                                                                     temp->clone(), rhs);
        
        CG_outputRepr * lhs = ir->GetLHSExpression(stmt[stmt_num].code);
        
        CG_outputRepr *assignment = ir->builder()->CreateAssignment(0,
                                                                    lhs, temp->clone());
        
        stmt[stmt_num].code = assignment;
      }
      // call function to replace rhs with temporary array
    }
  }
  
  //std::cout << "End of scalar_expand function!! \n";
  
  //  if(arrName == "RHS"){
  DependenceVector dv;
  std::vector<DependenceVector> E;
  dv.lbounds = std::vector<omega::coef_t>(4);
  dv.ubounds = std::vector<omega::coef_t>(4);
  dv.type = DEP_W2R;
  
  for (int k = 0; k < 4; k++) {
    dv.lbounds[k] = 0;
    dv.ubounds[k] = 0;
    
  }
  
  //std::vector<IR_ArrayRef*> array_refs = ir->FindArrayRef(stmt[newStmt_num].code);
  dv.sym = tmp_sym->clone();
  
  E.push_back(dv);
  
  dep.connect(newStmt_num, stmt_num, E);
  // }
  
}




std::pair<Relation, Relation> createCSRstyleISandXFORM(CG_outputBuilder *ocg,
                                                       std::vector<Relation> &outer_loop_bounds, std::string index_name,
                                                       std::map<int, Relation> &zero_loop_bounds,
                                                       std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                                                       std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_string,
                                                       Loop *this_loop) {
  
  Relation IS(outer_loop_bounds.size() + 1 + zero_loop_bounds.size());
  Relation XFORM(outer_loop_bounds.size() + 1 + zero_loop_bounds.size(),
                 2 * (outer_loop_bounds.size() + 1 + zero_loop_bounds.size()) + 1);
  
  F_And * f_r_ = IS.add_and();
  F_And * f_root = XFORM.add_and();
  
  if (outer_loop_bounds.size() > 0) {
    for (int it = 0; it < IS.n_set(); it++) {
      IS.name_set_var(it + 1,
                      const_cast<Relation &>(outer_loop_bounds[0]).set_var(it + 1)->name());
      XFORM.name_input_var(it + 1,
                           const_cast<Relation &>(outer_loop_bounds[0]).set_var(it + 1)->name());
      
    }
  } else if (zero_loop_bounds.size() > 0) {
    for (int it = 0; it < IS.n_set(); it++) {
      IS.name_set_var(it + 1,
                      const_cast<Relation &>(zero_loop_bounds.begin()->second).set_var(
                                                                                       it + 1)->name());
      XFORM.name_input_var(it + 1,
                           const_cast<Relation &>(zero_loop_bounds.begin()->second).set_var(
                                                                                            it + 1)->name());
      
    }
    
  }
  
  for (int i = 0; i < outer_loop_bounds.size(); i++)
    IS = replace_set_var_as_another_set_var(IS, outer_loop_bounds[i], i + 1,
                                            i + 1);
  
  int count = 1;
  for (std::map<int, Relation>::iterator i = zero_loop_bounds.begin();
       i != zero_loop_bounds.end(); i++, count++)
    IS = replace_set_var_as_another_set_var(IS, i->second,
                                            outer_loop_bounds.size() + 1 + count, i->first);
  
  if (outer_loop_bounds.size() > 0) {
    Free_Var_Decl *lb = new Free_Var_Decl(index_name + "_", 1); // index_
    Variable_ID csr_lb = IS.get_local(lb, Input_Tuple);
    
    Free_Var_Decl *ub = new Free_Var_Decl(index_name + "__", 1); // index__
    Variable_ID csr_ub = IS.get_local(ub, Input_Tuple);
    
    //lower bound
    
    F_And * f_r = IS.and_with_and();
    GEQ_Handle lower_bound = f_r->add_GEQ();
    lower_bound.update_coef(csr_lb, -1);
    lower_bound.update_coef(IS.set_var(outer_loop_bounds.size() + 1), 1);
    
    //upper bound
    
    GEQ_Handle upper_bound = f_r->add_GEQ();
    upper_bound.update_coef(csr_ub, 1);
    upper_bound.update_coef(IS.set_var(outer_loop_bounds.size() + 1), -1);
    upper_bound.update_const(-1);
    
    omega::CG_stringBuilder *ocgs = new CG_stringBuilder;
    
    std::vector<omega::CG_outputRepr *> reprs;
    std::vector<omega::CG_outputRepr *> reprs2;
    
    std::vector<omega::CG_outputRepr *> reprs3;
    std::vector<omega::CG_outputRepr *> reprs4;
    
    reprs.push_back(
                    ocg->CreateIdent(IS.set_var(outer_loop_bounds.size())->name()));
    reprs2.push_back(
                     ocgs->CreateIdent(
                                       IS.set_var(outer_loop_bounds.size())->name()));
    uninterpreted_symbols.insert(
                                 std::pair<std::string, std::vector<CG_outputRepr *> >(
                                                                                       index_name + "_", reprs));
    uninterpreted_symbols_string.insert(
                                        std::pair<std::string, std::vector<CG_outputRepr *> >(
                                                                                              index_name + "_", reprs2));
    
    std::string arg = "(" + IS.set_var(outer_loop_bounds.size())->name()
      + ")";
    std::vector< std::string > argvec;
    argvec.push_back( arg ); 
    
    CG_outputRepr *repr = ocg->CreateArrayRefExpression(index_name,
                                                        ocg->CreateIdent(IS.set_var(outer_loop_bounds.size())->name()));
    
    //debug_fprintf(stderr, "( VECTOR _)\n"); 
    //debug_fprintf(stderr, "loop.cc calling CreateDefineMacro( %s, argvec, repr)\n", (index_name + "_").c_str()); 
    this_loop->ir->CreateDefineMacro(index_name + "_", argvec, repr);
    
    Relation known_(copy(IS).n_set());
    known_.copy_names(copy(IS));
    known_.setup_names();
    Variable_ID index_lb = known_.get_local(lb, Input_Tuple);
    Variable_ID index_ub = known_.get_local(ub, Input_Tuple);
    F_And *fr = known_.add_and();
    GEQ_Handle g = fr->add_GEQ();
    g.update_coef(index_ub, 1);
    g.update_coef(index_lb, -1);
    g.update_const(-1);
    this_loop->addKnown(known_);
    
    reprs3.push_back(
                     
                     ocg->CreateIdent(IS.set_var(outer_loop_bounds.size())->name()));
    reprs4.push_back(
                     
                     ocgs->CreateIdent(IS.set_var(outer_loop_bounds.size())->name()));
    
    CG_outputRepr *repr2 = ocg->CreateArrayRefExpression(index_name,
                                                         ocg->CreatePlus(
                                                                         ocg->CreateIdent(
                                                                                          IS.set_var(outer_loop_bounds.size())->name()),
                                                                         ocg->CreateInt(1)));
    
    //debug_fprintf(stderr, "( VECTOR __)\n"); 
    //debug_fprintf(stderr, "loop.cc calling CreateDefineMacro( %s, argvec, repr)\n", (index_name + "__").c_str()); 
    
    this_loop->ir->CreateDefineMacro(index_name + "__", argvec, repr2);
    
    uninterpreted_symbols.insert(
                                 std::pair<std::string, std::vector<CG_outputRepr *> >(
                                                                                       index_name + "__", reprs3));
    uninterpreted_symbols_string.insert(
                                        std::pair<std::string, std::vector<CG_outputRepr *> >(
                                                                                              index_name + "__", reprs4));
  } else {
    Free_Var_Decl *ub = new Free_Var_Decl(index_name);
    Variable_ID csr_ub = IS.get_local(ub);
    F_And * f_r = IS.and_with_and();
    GEQ_Handle upper_bound = f_r->add_GEQ();
    upper_bound.update_coef(csr_ub, 1);
    upper_bound.update_coef(IS.set_var(outer_loop_bounds.size() + 1), -1);
    upper_bound.update_const(-1);
    
    GEQ_Handle lower_bound = f_r->add_GEQ();
    lower_bound.update_coef(IS.set_var(outer_loop_bounds.size() + 1), 1);
    
  }
  
  for (int j = 1; j <= XFORM.n_inp(); j++) {
    omega::EQ_Handle h = f_root->add_EQ();
    h.update_coef(XFORM.output_var(2 * j), 1);
    h.update_coef(XFORM.input_var(j), -1);
  }
  
  for (int j = 1; j <= XFORM.n_out(); j += 2) {
    omega::EQ_Handle h = f_root->add_EQ();
    h.update_coef(XFORM.output_var(j), 1);
  }
  
  if (_DEBUG_) {
    IS.print();
    XFORM.print();
    
  }
  
  return std::pair<Relation, Relation>(IS, XFORM);
  
}

std::pair<Relation, Relation> construct_reduced_IS_And_XFORM(IR_Code *ir,
                                                             const Relation &is, const Relation &xform, const std::vector<int> loops,
                                                             std::vector<int> &lex_order, Relation &known,
                                                             std::map<std::string, std::vector<CG_outputRepr *> > &uninterpreted_symbols) {
  
  Relation IS(loops.size());
  Relation XFORM(loops.size(), 2 * loops.size() + 1);
  int count_ = 1;
  std::map<int, int> pos_mapping;
  
  int n = is.n_set();
  Relation is_and_known = Intersection(copy(is),
                                       Extend_Set(copy(known), n - known.n_set()));
  
  for (int it = 0; it < loops.size(); it++, count_++) {
    IS.name_set_var(count_,
                    const_cast<Relation &>(is).set_var(loops[it])->name());
    XFORM.name_input_var(count_,
                         const_cast<Relation &>(xform).input_var(loops[it])->name());
    XFORM.name_output_var(2 * count_,
                          const_cast<Relation &>(xform).output_var((loops[it]) * 2)->name());
    XFORM.name_output_var(2 * count_ - 1,
                          const_cast<Relation &>(xform).output_var((loops[it]) * 2 - 1)->name());
    pos_mapping.insert(std::pair<int, int>(count_, loops[it]));
  }
  
  XFORM.name_output_var(2 * loops.size() + 1,
                        const_cast<Relation &>(xform).output_var(is.n_set() * 2 + 1)->name());
  
  F_And * f_r = IS.add_and();
  for (std::map<int, int>::iterator it = pos_mapping.begin();
       it != pos_mapping.end(); it++)
    IS = replace_set_var_as_another_set_var(IS, is_and_known, it->first,
                                            it->second);
  /*
    for (std::map<std::string, std::vector<CG_outputRepr *> >::iterator it2 =
    uninterpreted_symbols.begin();
    it2 != uninterpreted_symbols.end(); it2++) {
    std::vector<CG_outputRepr *> reprs_ = it2->second;
    //std::vector<CG_outputRepr *> reprs_2;
    
    for (int k = 0; k < reprs_.size(); k++) {
    std::vector<IR_ScalarRef *> refs = ir->FindScalarRef(reprs_[k]);
    bool exception_found = false;
    for (int m = 0; m < refs.size(); m++){
    
    if (refs[m]->name()
    == const_cast<Relation &>(is).set_var(it->second)->name())
    try {
    ir->ReplaceExpression(refs[m],
    ir->builder()->CreateIdent(
    IS.set_var(it->first)->name()));
    } catch (ir_error &e) {
    
    reprs_[k] = ir->builder()->CreateIdent(
    IS.set_var(it->first)->name());
    exception_found = true;
    }
    if(exception_found)
    break;
    }
    
    }
    it2->second = reprs_;
    }
    
    }
  */
  if (_DEBUG_) {
    std::cout << "relation debug" << std::endl;
    IS.print();
  }
  
  F_And *f_root = XFORM.add_and();
  
  count_ = 1;
  
  for (int j = 1; j <= loops.size(); j++) {
    omega::EQ_Handle h = f_root->add_EQ();
    h.update_coef(XFORM.output_var(2 * j), 1);
    h.update_coef(XFORM.input_var(j), -1);
  }
  for (int j = 0; j < loops.size(); j++, count_++) {
    omega::EQ_Handle h = f_root->add_EQ();
    h.update_coef(XFORM.output_var(count_ * 2 - 1), 1);
    h.update_const(-lex_order[count_ * 2 - 2]);
  }
  
  omega::EQ_Handle h = f_root->add_EQ();
  h.update_coef(XFORM.output_var((loops.size()) * 2 + 1), 1);
  h.update_const(-lex_order[xform.n_out() - 1]);
  
  if (_DEBUG_) {
    std::cout << "relation debug" << std::endl;
    IS.print();
    XFORM.print();
  }
  
  return std::pair<Relation, Relation>(IS, XFORM);
  
}

std::set<std::string> inspect_repr_for_scalars(IR_Code *ir,
                                               CG_outputRepr * repr, std::set<std::string> ignore) {
  
  std::vector<IR_ScalarRef *> refs = ir->FindScalarRef(repr);
  std::set<std::string> loop_vars;
  
  for (int i = 0; i < refs.size(); i++)
    if (ignore.find(refs[i]->name()) == ignore.end())
      loop_vars.insert(refs[i]->name());
  
  return loop_vars;
  
}

std::set<std::string> inspect_loop_bounds(IR_Code *ir, const Relation &R,
                                          int pos,
                                          std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols) {
  
  if (!R.is_set())
    throw loop_error("Input R has to be a set not a relation!");
  
  std::set<std::string> vars;
  
  std::vector<CG_outputRepr *> refs;
  Variable_ID v = const_cast<Relation &>(R).set_var(pos);
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
      if ((*gi).get_coef(v) != 0 && (*gi).is_const_except_for_global(v)) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() > 0) {
              
              std::string s = g->base_name();
              std::copy(
                        uninterpreted_symbols.find(s)->second.begin(),
                        uninterpreted_symbols.find(s)->second.end(),
                        back_inserter(refs));
              
            }
            
            break;
          }
          default:
            break;
          }
        }
        
      }
    }
  }
  
  for (int i = 0; i < refs.size(); i++) {
    std::vector<IR_ScalarRef *> refs_ = ir->FindScalarRef(refs[i]);
    
    for (int j = 0; j < refs_.size(); j++)
      vars.insert(refs_[j]->name());
    
  }
  return vars;
}

CG_outputRepr * create_counting_loop_body(IR_Code *ir, const Relation &R,
                                          int pos, CG_outputRepr * count,
                                          std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols) {
  
  if (!R.is_set())
    throw loop_error("Input R has to be a set not a relation!");
  
  CG_outputRepr *ub, *lb;
  ub = NULL;
  lb = NULL;
  std::vector<CG_outputRepr *> refs;
  Variable_ID v = const_cast<Relation &>(R).set_var(pos);
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
      if ((*gi).get_coef(v) != 0 && (*gi).is_const_except_for_global(v)) {
        bool same_ge_1 = false;
        bool same_ge_2 = false;
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() > 0) {
              
              std::string s = g->base_name();
              
              if ((*gi).get_coef(v) > 0) {
                if (ub != NULL)
                  throw ir_error(
                                 "bound expression too complex!");
                
                ub = ir->builder()->CreateInvoke(s,
                                                 uninterpreted_symbols.find(s)->second);
                //ub = ir->builder()->CreateMinus(ub->clone(), ir->builder()->CreateInt(-(*gi).get_const()));
                same_ge_1 = true;
                
              } else {
                if (lb != NULL)
                  throw ir_error(
                                 "bound expression too complex!");
                lb = ir->builder()->CreateInvoke(s,
                                                 uninterpreted_symbols.find(s)->second);
                same_ge_2 = true;
                
              }
            }
            
            break;
          }
          default:
            break;
          }
        }
        
        if (same_ge_1 && same_ge_2)
          lb = ir->builder()->CreatePlus(lb->clone(),
                                         ir->builder()->CreateInt(-(*gi).get_const()));
        else if (same_ge_1)
          ub = ir->builder()->CreatePlus(ub->clone(),
                                         ir->builder()->CreateInt(-(*gi).get_const()));
        else if (same_ge_2)
          lb = ir->builder()->CreatePlus(lb->clone(),
                                         ir->builder()->CreateInt(-(*gi).get_const()));
      }
    }
    
  }
  
  return ir->builder()->CreatePlusAssignment(0, count,
                                             ir->builder()->CreatePlus(
                                                                       ir->builder()->CreateMinus(ub->clone(), lb->clone()),
                                                                       ir->builder()->CreateInt(1)));
}



std::map<std::string, std::vector<std::string> > recurse_on_exp_for_arrays(
                                                                           IR_Code * ir, CG_outputRepr * exp) {
  
  std::map<std::string, std::vector<std::string> > arr_index_to_ref;
  switch (ir->QueryExpOperation(exp)) {
    
  case IR_OP_ARRAY_VARIABLE: {
    IR_ArrayRef *ref = dynamic_cast<IR_ArrayRef *>(ir->Repr2Ref(exp));
    IR_PointerArrayRef *ref_ =
      dynamic_cast<IR_PointerArrayRef *>(ir->Repr2Ref(exp));
    if (ref == NULL && ref_ == NULL)
      throw loop_error("Array symbol unidentifiable!");
    
    if (ref != NULL) {
      std::vector<std::string> s0;
      
      for (int i = 0; i < ref->n_dim(); i++) {
        CG_outputRepr * index = ref->index(i);
        std::map<std::string, std::vector<std::string> > a0 =
          recurse_on_exp_for_arrays(ir, index);
        std::vector<std::string> s;
        for (std::map<std::string, std::vector<std::string> >::iterator j =
               a0.begin(); j != a0.end(); j++) {
          if (j->second.size() != 1 && (j->second)[0] != "")
            throw loop_error(
                             "indirect array references not allowed in guard!");
          s.push_back(j->first);
        }
        std::copy(s.begin(), s.end(), back_inserter(s0));
      }
      arr_index_to_ref.insert(
                              std::pair<std::string, std::vector<std::string> >(
                                                                                ref->name(), s0));
    } else {
      std::vector<std::string> s0;
      for (int i = 0; i < ref_->n_dim(); i++) {
        CG_outputRepr * index = ref_->index(i);
        std::map<std::string, std::vector<std::string> > a0 =
          recurse_on_exp_for_arrays(ir, index);
        std::vector<std::string> s;
        for (std::map<std::string, std::vector<std::string> >::iterator j =
               a0.begin(); j != a0.end(); j++) {
          if (j->second.size() != 1 && (j->second)[0] != "")
            throw loop_error(
                             "indirect array references not allowed in guard!");
          s.push_back(j->first);
        }
        std::copy(s.begin(), s.end(), back_inserter(s0));
      }
      arr_index_to_ref.insert(
                              std::pair<std::string, std::vector<std::string> >(
                                                                                ref_->name(), s0));
    }
    break;
  }
  case IR_OP_PLUS:
  case IR_OP_MINUS:
  case IR_OP_MULTIPLY:
  case IR_OP_DIVIDE: {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(exp);
    std::map<std::string, std::vector<std::string> > a0 =
      recurse_on_exp_for_arrays(ir, v[0]);
    std::map<std::string, std::vector<std::string> > a1 =
      recurse_on_exp_for_arrays(ir, v[1]);
    arr_index_to_ref.insert(a0.begin(), a0.end());
    arr_index_to_ref.insert(a1.begin(), a1.end());
    break;
    
  }
  case IR_OP_POSITIVE:
  case IR_OP_NEGATIVE: {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(exp);
    std::map<std::string, std::vector<std::string> > a0 =
      recurse_on_exp_for_arrays(ir, v[0]);
    
    arr_index_to_ref.insert(a0.begin(), a0.end());
    break;
    
  }
  case IR_OP_VARIABLE: {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(exp);
    IR_ScalarRef *ref = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v[0]));
    
    std::string s = ref->name();
    std::vector<std::string> to_insert;
    to_insert.push_back("");
    arr_index_to_ref.insert(
                            std::pair<std::string, std::vector<std::string> >(s,
                                                                              to_insert));
    break;
  }
  case IR_OP_CONSTANT:
    break;
    
  default: {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(exp);
    
    for (int i = 0; i < v.size(); i++) {
      std::map<std::string, std::vector<std::string> > a0 =
        recurse_on_exp_for_arrays(ir, v[i]);
      
      arr_index_to_ref.insert(a0.begin(), a0.end());
    }
    
    break;
  }
  }
  return arr_index_to_ref;
}



std::vector<CG_outputRepr *> find_guards(IR_Code *ir, IR_Control *code) {
  debug_fprintf(stderr, "find_guards()\n"); 
  std::vector<CG_outputRepr *> guards;
  switch (code->type()) {
  case IR_CONTROL_IF: {
    debug_fprintf(stderr, "find_guards() it's an if\n"); 
    CG_outputRepr *cond = dynamic_cast<IR_If*>(code)->condition();
    
    std::vector<CG_outputRepr *> then_body;
    std::vector<CG_outputRepr *> else_body;
    IR_Block *ORTB = dynamic_cast<IR_If*>(code)->then_body(); 
    if (ORTB != NULL) {
      debug_fprintf(stderr, "recursing on then\n"); 
      then_body = find_guards(ir, ORTB); 
      //dynamic_cast<IR_If*>(code)->then_body());
    }
    if (dynamic_cast<IR_If*>(code)->else_body() != NULL) {
      debug_fprintf(stderr, "recursing on then\n"); 
      else_body = find_guards(ir,
                              dynamic_cast<IR_If*>(code)->else_body());
    }
    
    guards.push_back(cond);
    if (then_body.size() > 0)
      std::copy(then_body.begin(), then_body.end(),
                back_inserter(guards));
    if (else_body.size() > 0)
      std::copy(else_body.begin(), else_body.end(),
                back_inserter(guards));
    break;
  }
  case IR_CONTROL_BLOCK: {
    debug_fprintf(stderr, "find_guards() it's a control block\n"); 
    IR_Block*  IRCB = dynamic_cast<IR_Block*>(code);
    debug_fprintf(stderr, "find_guards() calling ir->FindOneLevelControlStructure(IRCB);\n"); 
    std::vector<IR_Control *> stmts = ir->FindOneLevelControlStructure(IRCB); 

    for (int i = 0; i < stmts.size(); i++) {
      std::vector<CG_outputRepr *> stmt_repr = find_guards(ir, stmts[i]);
      std::copy(stmt_repr.begin(), stmt_repr.end(),
                back_inserter(guards));
    }
    break;
  }
  case IR_CONTROL_LOOP: {
    debug_fprintf(stderr, "find_guards() it's a control loop\n"); 
    std::vector<CG_outputRepr *> body = find_guards(ir,
                                                    dynamic_cast<IR_Loop*>(code)->body());
    if (body.size() > 0)
      std::copy(body.begin(), body.end(), back_inserter(guards));
    break;
  } // loop 
  } // switch 
  return guards;
}

bool sort_helper(std::pair<std::string, std::vector<std::string> > i,
                 std::pair<std::string, std::vector<std::string> > j) {
  int c1 = 0;
  int c2 = 0;
  for (int k = 0; k < i.second.size(); k++)
    if (i.second[k] != "")
      c1++;
  
  for (int k = 0; k < j.second.size(); k++)
    if (j.second[k] != "")
      c2++;
  return (c1 < c2);
  
}

bool sort_helper_2(std::pair<int, int> i, std::pair<int, int> j) {
  
  return (i.second < j.second);
  
}

std::vector<std::string> construct_iteration_order(
                                                   std::map<std::string, std::vector<std::string> > & input) {
  std::vector<std::string> arrays;
  std::vector<std::string> scalars;
  std::vector<std::pair<std::string, std::vector<std::string> > > input_aid;
  
  for (std::map<std::string, std::vector<std::string> >::iterator j =
         input.begin(); j != input.end(); j++)
    input_aid.push_back(
                        std::pair<std::string, std::vector<std::string> >(j->first,
                                                                          j->second));
  
  std::sort(input_aid.begin(), input_aid.end(), sort_helper);
  
  for (int j = 0; j < input_aid[input_aid.size() - 1].second.size(); j++)
    if (input_aid[input_aid.size() - 1].second[j] != "") {
      arrays.push_back(input_aid[input_aid.size() - 1].second[j]);
      
    }
  
  if (arrays.size() > 0) {
    for (int i = input_aid.size() - 2; i >= 0; i--) {
      
      int max_count = 0;
      for (int j = 0; j < input_aid[i].second.size(); j++)
        if (input_aid[i].second[j] != "") {
          max_count++;
        }
      if (max_count > 0) {
        for (int j = 0; j < max_count; j++) {
          std::string s = input_aid[i].second[j];
          bool found = false;
          for (int k = 0; k < max_count; k++)
            if (s == arrays[k])
              found = true;
          if (!found)
            throw loop_error("guard condition not solvable");
        }
      } else {
        bool found = false;
        for (int k = 0; k < arrays.size(); k++)
          if (arrays[k] == input_aid[i].first)
            found = true;
        if (!found)
          arrays.push_back(input_aid[i].first);
      }
    }
  } else {
    
    for (int i = input_aid.size() - 1; i >= 0; i--) {
      arrays.push_back(input_aid[i].first);
    }
  }
  return arrays;
}



void Loop::compact(int stmt_num, int level, std::string new_array, int zero,
                   std::string data_array) {
  
  Relation equalities_in_xform;
  CG_outputRepr *x_check = NULL;
  //equalities_in_xform = replicate_complex_equalities_in_xform(
  //    stmt[stmt_num].xform);
  apply_xform(stmt_num);
  std::map<int, Relation> inner_loops;
  
  // 1a. Identify set of loops enclosed by level (store inner loops' bounds)
  
  for (int i = level + 1; i < stmt[stmt_num].loop_level.size(); i++) {
    
    Relation bound = get_loop_bound(stmt[stmt_num].IS, i, this->known);
    inner_loops.insert(std::pair<int, Relation>(i, bound));
    if (_DEBUG_)
      bound.print();
  }
  
  // 1b. Extra Check for Padding, in case N does not divide C
  //      i.   Get Loop Index of interior loop
  //  ii.  Check for arrays being subscripted by interior loop
  //  iii. Check for other indices in subscript
  Relation copy_IS = copy(stmt[stmt_num].IS);
  
  for (std::map<int, Relation>::iterator it = inner_loops.begin();
       it != inner_loops.end(); it++) {
    std::string index = stmt[stmt_num].IS.set_var(it->first)->name();
    
    Relation bound = it->second;
    int max = -1;
    for (DNF_Iterator di(bound.query_DNF()); di; di++)
      for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++)
        if (((*gi).get_coef(bound.set_var(it->first)) == -1)
            && (*gi).is_const(bound.set_var(it->first)))
          max = (*gi).get_const();
    
    Relation r(copy_IS.n_set());
    r.copy_names(copy_IS);
    r.setup_names();
    
    F_And *root = r.add_and();
    GEQ_Handle h = root->add_GEQ();
    h.update_coef(r.set_var(it->first), -1);
    h.update_const(max);
    
    GEQ_Handle g = root->add_GEQ();
    g.update_coef(r.set_var(it->first), 1);
    
    copy_IS = and_with_relation_and_replace_var(copy_IS,
                                                copy_IS.set_var(it->first), r);
  }
  
  for (std::map<int, Relation>::iterator it = inner_loops.begin();
       it != inner_loops.end(); it++) {
    std::vector<IR_ArrayRef *> refs2 = ir->FindArrayRef(
                                                        stmt[stmt_num].code);
    std::string index = stmt[stmt_num].IS.set_var(it->first)->name();
    
    std::set<IR_ArrayRef *> subscripts;
    for (int i = 0; i < refs2.size(); i++) {
      assert(refs2[i]->n_dim() == 1);
      std::vector<IR_ScalarRef *> scalar_refs = ir->FindScalarRef(
                                                                  refs2[i]->index(0));
      for (int j = 0; j < scalar_refs.size(); j++)
        if (scalar_refs[j]->name() == index)
          subscripts.insert(refs2[i]);
    }
    
    for (std::set<IR_ArrayRef *>::iterator i = subscripts.begin();
         i != subscripts.end(); i++) {
      CG_outputRepr *repr = (*i)->index(0);
      
      Relation mapping(stmt[stmt_num].IS.n_set(),
                       stmt[stmt_num].IS.n_set() + 1);
      for (int k = 1; k <= mapping.n_inp(); k++)
        mapping.name_input_var(k, stmt[stmt_num].IS.set_var(k)->name());
      mapping.setup_names();
      F_And *f_root = mapping.add_and();
      for (int k = 1; k <= mapping.n_inp(); k++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(k), 1);
        h.update_coef(mapping.output_var(k), -1);
      }
      
      exp2formula(this, ir, mapping, f_root, freevar, repr->clone(),
                  mapping.output_var(stmt[stmt_num].IS.n_set() + 1), 'w',
                  IR_COND_EQ, false, uninterpreted_symbols[stmt_num],
                  uninterpreted_symbols_stringrepr[stmt_num], unin_rel[stmt_num]);
      
      Relation r = omega::Range(
                                Restrict_Domain(mapping,
                                                Intersection(copy(copy_IS),
                                                             Extend_Set(copy(this->known),
                                                                        copy_IS.n_set()
                                                                        
                                                                        - this->known.n_set()))));
      
      if (_DEBUG_)
        r.print();
      
      for (int j = 1; j <= stmt[stmt_num].IS.n_set(); j++) {
        r = Project(r, j, Input_Var);
        r.simplify();
      }
      r.simplify();
      
      Conjunct *c = r.query_DNF()->single_conjunct();
      int size = 0;
      
      if (_DEBUG_)
        r.print();
      for (GEQ_Iterator gi(c->GEQs()); gi; gi++)
        if ((*gi).is_const(r.set_var(stmt[stmt_num].IS.n_set() + 1))
            && (*gi).get_coef(
                              r.set_var(stmt[stmt_num].IS.n_set() + 1)) < 0)
          size = (*gi).get_const()
            
            / (-(*gi).get_coef(
                               r.set_var(stmt[stmt_num].IS.n_set() + 1)));
      assert(size != 0);
      
      if (array_dims.find((*i)->name()) != array_dims.end()) {
        int decl_size = array_dims.find((*i)->name())->second;
        if (size + 1 > decl_size) {
          
          std::vector<CG_outputRepr *> size_repr2;
          size_repr2.push_back(ir->builder()->CreateInt(size + 1));
          IR_PointerSymbol *sub = ir->CreatePointerSymbol(
                                                          (*i)->symbol()->elem_type(), size_repr2);
          x_check = ir->builder()->StmtListAppend(x_check,
                                                  ir->CreateMalloc((*i)->symbol()->elem_type(),
                                                                   sub->name(), size_repr2[0]->clone()));
          ptr_variables.push_back(
                                  static_cast<IR_PointerSymbol *>(sub));
          cleanup_code = ir->builder()->StmtListAppend(cleanup_code,
                                                       ir->CreateFree(
                                                                      ir->builder()->CreateIdent(sub->name())));
          IR_ScalarSymbol * iter = ir->CreateScalarSymbol(
                                                          IR_CONSTANT_INT, 0, "");
          Relation new_rel(1);
          new_rel.name_set_var(1, iter->name());
          F_And *g_root = new_rel.add_and();
          
          GEQ_Handle g = g_root->add_GEQ();
          g.update_coef(new_rel.set_var(1), 1);
          
          GEQ_Handle g1 = g_root->add_GEQ();
          g1.update_coef(new_rel.set_var(1), -1);
          g1.update_const(decl_size - 1);
          
          std::map<std::string, std::vector<omega::CG_outputRepr *> > uninterpreted_symbols;
          CG_outputRepr *loop = output_loop(ir->builder(), new_rel, 1,
                                            Relation::True(1),
                                            std::vector<std::pair<CG_outputRepr *, int> >(1,
                                                                                          std::make_pair(
                                                                                                         static_cast<CG_outputRepr *>(NULL),
                                                                                                         0)), uninterpreted_symbols);
          CG_outputBuilder *ocg = ir->builder();
          CG_outputRepr *body = ocg->CreateAssignment(0,
                                                      ocg->CreateArrayRefExpression(sub->name(),
                                                                                    ocg->CreateIdent(iter->name())),
                                                      ocg->CreateArrayRefExpression((*i)->name(),
                                                                                    ocg->CreateIdent(iter->name())));
          
          body = ir->builder()->CreateLoop(0, loop, body);
          
          x_check = ir->builder()->StmtListAppend(x_check, body);
          
          ir->ReplaceExpression(*i,
                                ir->builder()->CreateArrayRefExpression(sub->name(),
                                                                        repr->clone()));
          
        }
        
      }
      
      // get_bounds
      //  iv.  Compute max data footprint
      //  v.   Check if max data exceeds array_dims
      //  vi   If so extend and copy  (ReplaceExpression)
      
    }
  }
  
  // 2. Identify inner guard, if more than one throw error for now
  
  IR_Control *code = ir->GetCode(stmt[stmt_num].code->clone());
  debug_fprintf(stderr, "loop.cc back from ir->GetCode()\n"); 
  std::vector<CG_outputRepr *> guards = find_guards(ir, code);
  debug_fprintf(stderr, "loop.cc  back from find_guards()\n"); 
  
  if (guards.size() > 1)
    throw loop_error(
                     "Support for compaction with multiple guards currently not supported!");
  
  if (guards.size() == 0)
    throw loop_error("No guards found within compact command");
  
  // 3. Exp2formula/constraint for  guard
  std::map<std::string, std::vector<std::string> > res =
    recurse_on_exp_for_arrays(ir, guards[0]->clone());
  
  if (_DEBUG_)
    for (std::map<std::string, std::vector<std::string> >::iterator j =
           res.begin(); j != res.end(); j++) {
      std::cout << j->first << std::endl;
      for (int i = 0; i < j->second.size(); i++)
        if ((j->second)[i] != "")
          std::cout << (j->second)[i] << std::endl;
      
      std::cout << std::endl;
    }
  
  std::vector<std::string> variable_order = construct_iteration_order(res);
  
  if (_DEBUG_) {
    for (int i = 0; i < variable_order.size(); i++)
      std::cout << variable_order[i] << std::endl;
  }
  
  Relation r(variable_order.size());
  
  for (int i = 0; i < variable_order.size(); i++)
    r.name_set_var(i + 1, variable_order[i]);
  
  std::vector<std::pair<int, int> > name_to_pos;
  for (int j = 0; j < variable_order.size(); j++) {
    bool found = false;
    for (int i = level; i <= stmt[stmt_num].IS.n_set(); i++)
      if (stmt[stmt_num].IS.set_var(i)->name() == variable_order[j]) {
        name_to_pos.push_back(std::pair<int, int>(j + 1, i));
        found = true;
      }
    if (!found)
      throw loop_error("guard condition too complex to compact");
  }
  std::map<int, int> dont_project;
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  for (std::vector<std::pair<int, int> >::iterator it = name_to_pos.begin();
       it != name_to_pos.end(); it++) {
    Relation R = get_loop_bound(stmt[stmt_num].IS, it->second, this->known);
    bool ignore = false;
    
    for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        GEQ_Handle h = f_root->add_GEQ();
        if ((*gi).get_coef(R.set_var(it->second)) != 0) {
          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
              
            case Global_Var: {
              
              Global_Var_ID g = v->get_global_var();
              
              if (g->arity() > 0)
                ignore = true;
              
              break;
              
            }
            default:
              break;
            }
          }
          
        }
      }
    if (ignore)
      dont_project.insert(std::pair<int, int>(it->first, it->second));
  }
  
  //assert(dont_project.size() == 1);
  std::vector<Relation> actual_bounds;
  std::map<int, int> pos_map;
  for (std::vector<std::pair<int, int> >::iterator it = name_to_pos.begin();
       it != name_to_pos.end(); it++) {
    bool skip = false;
    //for ( i = 0; i < dont_project.size(); i++)
    if (dont_project.find(it->first) != dont_project.end())
      skip = true;
    
    if (!skip) {
      actual_bounds.push_back(
                              replace_set_var_as_another_set_var(r,
                                                                 get_loop_bound(stmt[stmt_num].IS, it->second,
                                                                                this->known), it->first, it->second));
      
    }
    // pos_map.insert(std::pair<int,int>(it->second, it->first));
  }
  
  if (_DEBUG_) {
    //  r.print();
    for (std::map<int, int>::iterator i = dont_project.begin();
         i != dont_project.end(); i++)
      std::cout << i->first << std::endl;
  }
  F_And *f_root_ = r.and_with_and();
  std::vector<omega::Free_Var_Decl*> freevars;
  exp2constraint(this, ir, r, f_root_, freevars, guards[0]->clone(), true,
                 uninterpreted_symbols[stmt_num],
                 uninterpreted_symbols_stringrepr[stmt_num], unin_rel[stmt_num]);
  r.simplify();
  if (_DEBUG_)
    r.print();
  
  // 4. Project from  innermost to outermost and
  //    back in to see which loops may be cancelled(store cancelled loops (build and store their expressions) )
  
  std::sort(name_to_pos.begin(), name_to_pos.end(), sort_helper_2);
  if (_DEBUG_)
    for (int i = 0; i < name_to_pos.size(); i++)
      std::cout << name_to_pos[i].first << std::endl;
  Relation update = copy(r);
  update.copy_names(r);
  update.setup_names();
  std::stack<Relation> to_eliminate;
  std::vector<std::pair<int, CG_outputRepr *> > pos_to_repr;
  
  std::vector<bool> already_simplified;
  
  for (int i = name_to_pos.size() - 1; i >= 0; i--)
    already_simplified.push_back(false);
  
  int max_count = -1;
  std::map<int, CG_outputRepr *> most_reprs;
  for (int k = 0; k < name_to_pos.size(); k++) {
    int project_count = 0;
    std::map<int, CG_outputRepr *> reprs;
    for (int i = name_to_pos.size() - 1; i >= 0; i--)
      already_simplified[i] = false;
    std::stack<Relation> to_eliminate;
    to_eliminate.push(update);
    std::vector<Relation> bounds = actual_bounds;
    for (int i = k; i < name_to_pos.size() && !to_eliminate.empty(); i++) {
      Relation update = to_eliminate.top();
      to_eliminate.pop();
      
      // int project_count = 0;
      
      std::pair<Relation, bool> temp;
      temp.first = copy(update);
      temp.first.copy_names(update);
      temp.first.copy_names(update);
      temp.first.setup_names();
      temp.second = false;
      std::pair<EQ_Handle, int> eq_handle;
      bool more_variables_left = false;
      if (dont_project.find(name_to_pos[i].first) == dont_project.end()) {
        //Projection Active
        for (int j = name_to_pos.size() - 1; j > i; j--) {
          if (dont_project.find(name_to_pos[j].first)
              == dont_project.end() && !already_simplified[j]) {
            //temp = Project(temp, name_to_pos[j].first,
            //    Set_Var);
            temp = replace_set_var_as_existential(temp.first,
                                                  name_to_pos[j].first, bounds);
            more_variables_left = true;
          }
        }
        for (int j = i - 1; j >= 0; j--) {
          if (dont_project.find(name_to_pos[j].first)
              == dont_project.end() && !already_simplified[j]) {
            //temp = Project(temp, name_to_pos[j].first,
            //    Set_Var);
            temp = replace_set_var_as_existential(temp.first,
                                                  name_to_pos[j].first, bounds);
            more_variables_left = true;
          }
          
        }
        if (temp.second || !more_variables_left) {
          
          eq_handle = find_simplest_assignment(temp.first,
                                               temp.first.set_var(name_to_pos[i].first),
                                               std::vector<std::pair<CG_outputRepr *, int> >(
                                                                                             temp.first.n_set(),
                                                                                             std::make_pair(
                                                                                                            static_cast<CG_outputRepr *>(NULL),
                                                                                                            0)));
          if (eq_handle.second == INT_MAX) {
            
            std::set<Variable_ID> excluded_floor_vars;
            std::pair<bool, GEQ_Handle> geq_handle =
              find_floor_definition(temp.first,
                                    temp.first.set_var(
                                                       name_to_pos[i].first),
                                    excluded_floor_vars);
            
            if (geq_handle.first) {
              
              already_simplified[i] = true;
              project_count++;
              
              reprs.insert(
                           std::pair<int, CG_outputRepr *>(
                                                           name_to_pos[i].second,
                                                           construct_int_floor(ir->builder(),
                                                                               temp.first,
                                                                               geq_handle.second,
                                                                               temp.first.set_var(
                                                                                                  name_to_pos[i].first),
                                                                               std::vector<
                                                                                 std::pair<
                                                                                   CG_outputRepr *,
                                                                                   int> >(
                                                                                          temp.first.n_set(),
                                                                                          std::make_pair(
                                                                                                         static_cast<CG_outputRepr *>(NULL),
                                                                                                         0)),
                                                                               uninterpreted_symbols[stmt_num])));
              
              //  reprs.push_back(result.second.first);
              Relation update_prime = replace_set_var_as_Global(
                                                                update, name_to_pos[i].first, bounds);
              
              std::vector<Relation>::iterator it;
              bool erase = false;
              for (it = bounds.begin(); it != bounds.end();
                   it++) {
                for (DNF_Iterator di(it->query_DNF()); di;
                     di++) {
                  for (GEQ_Iterator gi((*di)->GEQs()); gi;
                       gi++) {
                    
                    if ((*gi).get_coef(
                                       it->set_var(
                                                   name_to_pos[i].first))
                        != 0
                        && (*gi).is_const_except_for_global(
                                                            it->set_var(
                                                                        name_to_pos[i].first))) {
                      erase = true;
                      break;
                    }
                    if (erase)
                      break;
                  }
                  if (erase)
                    break;
                }
                if (erase)
                  break;
              }
              if (erase)
                bounds.erase(it);
              
              //  update_prime.copy_names(update);
              update_prime.setup_names();
              //update_prime.and_with_GEQ(geq_handle.second);
              //update_prime = and_with_relation_and_replace_var(
              //    update,
              //    update.set_var(name_to_pos[i].first),
              //      update_prime);
              to_eliminate.push(update_prime);
              
            } else {
              bool found = false;
              for (int j = i + 1; j < name_to_pos.size(); j++) {
                if (dont_project.find(name_to_pos[j].first)
                    == dont_project.end()
                    && !already_simplified[j]) {
                  //temp = Project(temp, name_to_pos[j].first,
                  //    Set_Var);
                  Relation temp = replace_set_var_as_Global(
                                                            update, name_to_pos[j].first,
                                                            bounds);
                  eq_handle =
                    find_simplest_assignment(temp,
                                             temp.set_var(
                                                          name_to_pos[i].first),
                                             std::vector<
                                               std::pair<
                                                 CG_outputRepr *,
                                                 int> >(
                                                        temp.n_set(),
                                                        std::make_pair(
                                                                       static_cast<CG_outputRepr *>(NULL),
                                                                       0)));
                  
                  bool erase = false;
                  std::vector<Relation>::iterator it;
                  for (it = bounds.begin();
                       it != bounds.end(); it++) {
                    for (DNF_Iterator di(it->query_DNF());
                         di; di++) {
                      for (GEQ_Iterator gi((*di)->GEQs());
                           gi; gi++) {
                        
                        if ((*gi).get_coef(
                                           it->set_var(
                                                       name_to_pos[j].first))
                            != 0
                            && (*gi).is_const_except_for_global(
                                                                it->set_var(
                                                                            name_to_pos[j].first))) {
                          erase = true;
                          break;
                        }
                        if (erase)
                          break;
                      }
                      if (erase)
                        break;
                    }
                    if (erase)
                      break;
                  }
                  if (erase)
                    bounds.erase(it);
                  already_simplified[j] = true;
                  
                  if (eq_handle.second != INT_MAX) {
                    
                    found = true;
                    
                    already_simplified[i] = true;
                    project_count++;
                    
                    std::pair<CG_outputRepr *,
                              std::pair<CG_outputRepr *, int> > result =
                      output_assignment(ir->builder(),
                                        temp,
                                        name_to_pos[i].first,
                                        this->known,
                                        std::vector<
                                          std::pair<
                                            CG_outputRepr *,
                                            int> >(
                                                   temp.n_set(),
                                                   std::make_pair(
                                                                  static_cast<CG_outputRepr *>(NULL),
                                                                  0)),
                                        uninterpreted_symbols[stmt_num]);
                    
                    if (result.first == NULL
                        && result.second.first != NULL
                        && result.second.second
                        != INT_MAX) {
                      
                      ir->builder()->CreateAssignment(0,
                                                      ir->builder()->CreateIdent(
                                                                                 temp.set_var(
                                                                                              name_to_pos[i].first)->name()),
                                                      result.second.first);
                      
                      reprs.insert(
                                   std::pair<int,
                                             CG_outputRepr *>(
                                                              name_to_pos[i].second,
                                                              result.second.first));
                      Relation update_prime =
                        replace_set_var_as_Global(
                                                  temp,
                                                  name_to_pos[i].first,
                                                  bounds);
                      //update_prime.copy_names(update);
                      update_prime.setup_names();
                      to_eliminate.push(update_prime);
                      
                      std::vector<Relation>::iterator it;
                      bool erase = false;
                      for (it = bounds.begin();
                           it != bounds.end(); it++) {
                        for (DNF_Iterator di(
                                             it->query_DNF()); di;
                             di++) {
                          for (GEQ_Iterator gi(
                                               (*di)->GEQs()); gi;
                               gi++) {
                            
                            if ((*gi).get_coef(
                                               it->set_var(
                                                           name_to_pos[i].first))
                                != 0
                                && (*gi).is_const_except_for_global(
                                                                    it->set_var(
                                                                                name_to_pos[i].first))) {
                              erase = true;
                              break;
                            }
                            if (erase)
                              break;
                          }
                          if (erase)
                            break;
                        }
                        if (erase)
                          break;
                      }
                      if (erase)
                        bounds.erase(it);
                      
                    }
                    break;
                    //  update_prime.copy_names(update);
                    //update_prime.setup_names();
                    
                  }
                  
                }
              }
              /*  if (!found)
                  for (int j = i - 1; j >= 0; j--) {
                  if (dont_project.find(name_to_pos[j].first)
                  == dont_project.end()
                  && !already_simplified[j]) {
                  //temp = Project(temp, name_to_pos[j].first,
                  //   Set_Var);
                  Relation temp =
                  replace_set_var_as_Global(
                  update,
                  name_to_pos[j].first,
                  bounds);
                  eq_handle =
                  find_simplest_assignment(temp,
                  temp.set_var(
                  name_to_pos[i].first),
                  std::vector<
                  std::pair<
                  CG_outputRepr *,
                  int> >(
                  temp.n_set(),
                  std::make_pair(
                  static_cast<CG_outputRepr *>(NULL),
                  0)));
                  
                  bool erase = false;
                  std::vector<Relation>::iterator it;
                  for (it = bounds.begin();
                  it != bounds.end(); it++) {
                  for (DNF_Iterator di(
                  it->query_DNF()); di;
                  di++) {
                  for (GEQ_Iterator gi(
                  (*di)->GEQs()); gi;
                  gi++) {
                  
                  if ((*gi).get_coef(
                  it->set_var(
                  name_to_pos[j].first))
                  != 0
                  && (*gi).is_const_except_for_global(
                  it->set_var(
                  name_to_pos[j].first))) {
                  erase = true;
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  bounds.erase(it);
                  already_simplified[j] = true;
                  
                  if (eq_handle.second != INT_MAX) {
                  
                  found = true;
                  
                  already_simplified[i] = true;
                  project_count++;
                  
                  std::pair<CG_outputRepr *,
                  std::pair<CG_outputRepr *,
                  int> > result =
                  output_assignment(
                  ir->builder(), temp,
                  name_to_pos[i].first,
                  this->known,
                  std::vector<
                  std::pair<
                  CG_outputRepr *,
                  int> >(
                  temp.n_set(),
                  std::make_pair(
                  static_cast<CG_outputRepr *>(NULL),
                  0)),
                  uninterpreted_symbols[stmt_num]);
                  
                  if (result.first == NULL
                  && result.second.first
                  != NULL
                  && result.second.second
                  != INT_MAX) {
                  
                  ir->builder()->CreateAssignment(
                  0,
                  ir->builder()->CreateIdent(
                  temp.set_var(
                  name_to_pos[i].first)->name()),
                  result.second.first);
                  
                  reprs.insert(
                  std::pair<int,
                  CG_outputRepr *>(
                  name_to_pos[i].second,
                  result.second.first));
                  Relation update_prime =
                  replace_set_var_as_Global(
                  temp,
                  name_to_pos[i].first,
                  bounds);
                  //update_prime.copy_names(update);
                  update_prime.setup_names();
                  to_eliminate.push(update_prime);
                  
                  std::vector<Relation>::iterator it;
                  bool erase = false;
                  for (it = bounds.begin();
                  it != bounds.end();
                  it++) {
                  for (DNF_Iterator di(
                  it->query_DNF());
                  di; di++) {
                  for (GEQ_Iterator gi(
                  (*di)->GEQs());
                  gi; gi++) {
                  
                  if ((*gi).get_coef(
                  it->set_var(
                  name_to_pos[i].first))
                  != 0
                  && (*gi).is_const_except_for_global(
                  it->set_var(
                  name_to_pos[i].first))) {
                  erase = true;
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  break;
                  }
                  if (erase)
                  bounds.erase(it);
                  
                  }
                  break;
                  // update_prime.copy_names(update);
                  //update_prime.setup_names();
                  
                  }
                  }
                  
                  }*/
              
            }
          }
          
          else {
            already_simplified[i] = true;
            project_count++;
            //reprs.push_back(result.second.first);
            
            std::pair<CG_outputRepr *,
                      std::pair<CG_outputRepr *, int> > result =
              output_assignment(ir->builder(), temp.first,
                                name_to_pos[i].first, this->known,
                                std::vector<
                                  std::pair<CG_outputRepr *, int> >(temp.first.n_set(),
                                                                    std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)),
                                uninterpreted_symbols[stmt_num]);
            
            if (result.first == NULL && result.second.first != NULL
                && result.second.second != INT_MAX) {
              
              ir->builder()->CreateAssignment(0,
                                              ir->builder()->CreateIdent(temp.first.set_var(name_to_pos[i].first)->name()),
                                              result.second.first);
              
              reprs.insert(
                           std::pair<int, CG_outputRepr *>(
                                                           name_to_pos[i].second,
                                                           result.second.first));
              Relation update_prime = replace_set_var_as_Global(
                                                                update, name_to_pos[i].first, bounds);
              //update_prime.copy_names(update);
              update_prime.setup_names();
              to_eliminate.push(update_prime);
              
              std::vector<Relation>::iterator it;
              bool erase = false;
              for (it = bounds.begin(); it != bounds.end();
                   it++) {
                for (DNF_Iterator di(it->query_DNF()); di;
                     di++) {
                  for (GEQ_Iterator gi((*di)->GEQs()); gi;
                       gi++) {
                    
                    if ((*gi).get_coef(
                                       it->set_var(
                                                   name_to_pos[i].first))
                        != 0
                        && (*gi).is_const_except_for_global(it->set_var(name_to_pos[i].first))) {
                      erase = true;
                      break;
                    }
                    if (erase)
                      break;
                  }
                  if (erase)
                    break;
                }
                if (erase)
                  break;
              }
              if (erase)
                bounds.erase(it);
              
            }
            /*update_prime.and_with_EQ(eq_handle.first);
              update_prime = and_with_relation_and_replace_var(
              update,
              update.set_var(name_to_pos[i].first),
              update_prime);
            */
            
          }
        }
      }
    }
    if (project_count > max_count) {
      most_reprs = reprs;
      max_count = project_count;
      
    }
    
  }
  
  if (_DEBUG_) {
    std::cout << max_count << std::endl;
    
    for (std::map<int, CG_outputRepr *>::iterator it = most_reprs.begin();
         it != most_reprs.end(); it++) {
      //   for(int i=0; i <most_reprs.size(); i++)
      //  stmt[stmt_num].code = ir->builder()->StmtListAppend(stmt[stmt_num].code->clone(), it->second->clone());
      std::cout << it->first << std::endl;
    }
  }
  
  // 5. Identify the loop from which the loop index being compacted is being derived(as a function of this loop), identify all loops this loop index is dependent on and compute IS. compacted  being computed non affine loops(pattern match index[i] and index[i+1] to
  //    sum closed form distance. insert loop above computation loop setLexicalOrder,
  
  std::vector<int> loops_for_non_zero_block_count;
  
  if (most_reprs.find(level) == most_reprs.end())
    ; //throw loop_error("Cannot derive compacted iterator !");
  else {
    std::pair<int, CG_outputRepr *> entry = *(most_reprs.find(level));
    
    std::set<std::string> ignore;
    ignore.insert(stmt[stmt_num].IS.set_var(level)->name());
    std::set<std::string> indices = inspect_repr_for_scalars(ir,
                                                             entry.second, ignore);
    
    //assert(
    //    indices.size() == 1
    //        && *(indices.begin())
    //            == stmt[stmt_num].IS.set_var(
    //                dont_project.begin()->second)->name());
  }
  std::set<std::string> derivative_loops = inspect_loop_bounds(ir,
                                                               get_loop_bound(stmt[stmt_num].IS, dont_project.begin()->second,
                                                                              this->known), dont_project.begin()->second,
                                                               uninterpreted_symbols[stmt_num]);
  
  if (_DEBUG_) {
    
    for (std::set<std::string>::iterator it = derivative_loops.begin();
         it != derivative_loops.end(); it++) {
      //   for(int i=0; i <most_reprs.size(); i++)
      //  stmt[stmt_num].code = ir->builder()->StmtListAppend(stmt[stmt_num].code->clone(), it->second->clone());
      std::cout << *it << std::endl;
    }
  }
  
  Relation counting_IS(derivative_loops.size());
  Relation counting_XFORM(derivative_loops.size(),
                          2 * derivative_loops.size() + 1);
  
  for (std::map<int, Relation>::iterator it = inner_loops.begin();
       it != inner_loops.end(); it++) {
    bool found = false;
    for (std::map<int, int>::iterator j = dont_project.begin();
         j != dont_project.end(); j++)
      if (j->second == it->first)
        found = true;
    if (!found)
      loops_for_non_zero_block_count.push_back(it->first);
    
  }
  
  // 6. Above will give worst case count of non-zero blocks, multiply that by inner loop bounds. compute inner loop bounds
  
  CG_outputRepr *count = ir->builder()->CreateIdent("chill_count_0");
  
  CG_outputRepr *loop_body = create_counting_loop_body(ir,
                                                       get_loop_bound(stmt[stmt_num].IS, dont_project.begin()->second,
                                                                      this->known), dont_project.begin()->second, count->clone(),
                                                       uninterpreted_symbols[stmt_num]);
  
  if (_DEBUG_) {
    
    for (int it = 0;
         
         it < loops_for_non_zero_block_count.size(); it++) {
      //   for(int i=0; i <most_reprs.size(); i++)
      //  stmt[stmt_num].code = ir->builder()->StmtListAppend(stmt[stmt_num].code->clone(), it->second->clone());
      std::cout << loops_for_non_zero_block_count[it] << std::endl;
      
    }
    //stmt[stmt_num].code = ir->builder()->StmtListAppend(stmt[stmt_num].code->clone(), loop_body->clone());
  }
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::vector<int> derived_loops;
  for (std::set<std::string>::iterator it = derivative_loops.begin();
       it != derivative_loops.end(); it++) {
    for (int i = 1; i <= stmt[stmt_num].IS.n_set(); i++)
      if (stmt[stmt_num].IS.set_var(i)->name() == *it) {
        
        //   for(int i=0; i <most_reprs.size(); i++)
        //  stmt[stmt_num].code = ir->builder()->StmtListAppend(stmt[stmt_num].code->clone(), it->second->clone());
        derived_loops.push_back(i);
        break;
      }
  }
  
  std::pair<Relation, Relation> IS_and_XFORM = construct_reduced_IS_And_XFORM(ir, 
                                                                              stmt[stmt_num].IS, 
                                                                              stmt[stmt_num].xform, 
                                                                              derived_loops, 
                                                                              lex,
                                                                              this->known, 
                                                                              uninterpreted_symbols[stmt_num]);
  
  CG_outputRepr *body = loop_body;
  
  for (int i = derived_loops.size(); i >= 1; i--) {
    CG_outputRepr *loop = output_loop(ir->builder(), 
                                      IS_and_XFORM.first, 
                                      i,
                                      this->known,
                                      std::vector<std::pair<CG_outputRepr *, int> >(IS_and_XFORM.first.n_set(),
                                                                                    std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)),
                                      uninterpreted_symbols[stmt_num]);
    
    body = ir->builder()->CreateLoop(0, loop, body);
    
  }
  
  CG_outputBuilder *ocg = ir->builder();
  //init_code = ocg->StmtListAppend(init_code,
  //    ocg->CreateAssignment(0, count->clone(), ocg->CreateInt(0)));
  //init_code = ir->builder()->StmtListAppend(init_code, body);
  
  init_code = ocg->StmtListAppend(init_code, x_check);
  // 7. Malloc offset_index, new_array and explicit_index and marked based on size of kk loop.
  IR_PointerSymbol *offset_index, *explicit_index, *new_array_prime, *marked;
  IR_PointerSymbol *offset_index2, *new_array_prime2;
  //IR_CONSTANT_TYPE type;
  
  if (level > 1) {
    int ub, lb;
    Relation R = get_loop_bound(stmt[stmt_num].IS, level - 1, this->known);
    for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        if ((*gi).get_coef(R.set_var(level - 1)) < 0
            && (*gi).is_const(R.set_var(level - 1))) {
          
          ub = (*gi).get_const();
        } else if ((*gi).get_coef(R.set_var(level - 1)) > 0
                   && (*gi).is_const(R.set_var(level - 1))) {
          
          lb = (*gi).get_const();
        }
      }
    
    if (lb < 0 || ub < 0)
      throw loop_error("bounds of outer loops cannot be determined");
    
    int size = (ub - lb + 1);
    std::vector<CG_outputRepr *> size_repr;

    // this seems really dumb
    //size_repr.push_back(ocg->CreatePlus(ocg->CreateInt(size),ocg->CreateInt(1)));
    size_repr.push_back(ocg->CreateInt(size+1));
    
    // nameless. name generated in CreatePointerSymbol()
    offset_index2 = ir->CreatePointerSymbol(IR_CONSTANT_INT, size_repr);
    debug_fprintf(stderr, "IR_PointerSymbol *offset_index2 has name?\n");  

    debug_fprintf(stderr, "making a malloc of %s\n", offset_index2->name().c_str()); 
    ptr_variables.push_back(static_cast<IR_PointerSymbol *>(offset_index2));

    //die(); 
    // this didn't just make a malloc. it cast it to an int, and used that to assign the new memory to a variable
    //init_code = ocg->StmtListAppend(init_code,
    //                                ir->CreateMalloc(IR_CONSTANT_INT, 
    //                                                 offset_index2->name(),
    //                                                 size_repr[0]->clone()));


    // SHOULD be something like this  TODO why is IR so limited?
    //CG_outputRepr *theMalloc = ir->CreateMalloc(IR_CONSTANT_INT, size_repr[0]->clone()); // malloc (sizeof int) * 248) 

    //  (int *)malloc (sizeof int) * 248) 
    //CG_outputRepr *theCast = ir->CreateCast( IR_CONSTANT_INT, theMalloc); // this does not exist

    // P_DATA0 = (int *)malloc (sizeof int) * 248);
    //CG_outputRepr *assign  = ir->CreateAssign(                            // this does not exist
  



    // seemingly  P_DATA0 = (int *) malloc( sizeof(int) * (247 +1));

    offset_index2->set_size(0, size_repr[0]->clone()); // a pointer symbol that kows what size it will be malloced to? 

  }


  CG_outputRepr *count_2 = ocg->CreateIdent("chill_count_1");
  std::vector<CG_outputRepr *> size_repr;
  size_repr.push_back(ocg->CreatePlus(count_2->clone(), ocg->CreateInt(1)));
  /*  offset_index = ir->CreatePointerSymbol(IR_CONSTANT_INT, size_repr);
      ptr_variables.push_back(static_cast<IR_PointerSymbol *>(offset_index));
      init_code = ocg->StmtListAppend(init_code,
      ir->CreateMalloc(IR_CONSTANT_INT, offset_index->name(),
      size_repr[0]->clone()));
      offset_index->set_size(0, size_repr[0]->clone());
  */
  //  std::vector<CG_outputRepr *> size_repr2;
  //  size_repr2.push_back(count->clone());
  explicit_index = ir->CreatePointerSymbol(IR_CONSTANT_INT, size_repr);
  explicit_index->set_size(0, size_repr[0]->clone());
  ptr_variables.push_back(static_cast<IR_PointerSymbol *>(explicit_index));
  
  /*  init_code = ocg->StmtListAppend(init_code,
      ir->CreateMalloc(IR_CONSTANT_INT, explicit_index->name(),
      size_repr2[0]->clone()));
  */
  std::vector<CG_outputRepr *> size_repr3;
  int size = 1;
  std::vector<std::pair<int, int> > inner_loop_bounds;
  for (int i = 0.; i < loops_for_non_zero_block_count.size(); i++) {
    // Relation R = extract_upper_bound(i->second, i->second.set_var(i->first));
    Relation R = get_loop_bound(stmt[stmt_num].IS,
                                loops_for_non_zero_block_count[i], known);
    
    if (is_single_loop_iteration(R, loops_for_non_zero_block_count[i],
                                 known)) {
      inner_loop_bounds.push_back(std::pair<int, int>(0, 0));
      size *= 1;
      continue;
      
    }
    
    int lb = -1, ub = -1;
    for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        if ((*gi).get_coef(R.set_var(loops_for_non_zero_block_count[i]))
            < 0
            && (*gi).is_const(
                              R.set_var(loops_for_non_zero_block_count[i]))) {
          
          ub = (*gi).get_const();
        } else if ((*gi).get_coef(
                                  R.set_var(loops_for_non_zero_block_count[i])) > 0
                   && (*gi).is_const(
                                     R.set_var(loops_for_non_zero_block_count[i]))) {
          
          lb = (*gi).get_const();
        }
      }
    
    if (lb < 0 || ub < 0)
      throw loop_error("bounds of inner loops cannot be determined");
    
    size *= (ub - lb + 1);
    inner_loop_bounds.push_back(std::pair<int, int>(lb, ub));
    
  }
  
  std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[stmt_num].code);
  IR_CONSTANT_TYPE type;
  IR_ArrayRef* data_arr_ref;
  for (int i = 0; i < refs.size(); i++)
    if (refs[i]->name() == data_array) {
      type = refs[i]->symbol()->elem_type();
      data_arr_ref = refs[i];
    }
  
  debug_fprintf(stderr, "loop.cc 5293  calling CreateArrayType( float, %d )\n", size);


  // all this is to create a linked list data structure
  //struct a_list
  //{
  //   float data[4];
  //   int col;
  //   struct a_list *next;
  //}
  CG_outputRepr *temp_arr_data_type = ir->CreateArrayType(IR_CONSTANT_FLOAT,
                                                          ocg->CreateInt(size));
  CG_outputRepr *temp_col_type = ir->CreateScalarType(IR_CONSTANT_INT);
  
  std::vector<std::string>     class_data;
  std::vector<CG_outputRepr *> class_data_types;
  
  class_data.push_back("data");
  class_data.push_back("col");
  
  class_data_types.push_back(temp_arr_data_type);
  class_data_types.push_back(temp_col_type);
  throw std::runtime_error("about to use ROSE Builder in generic loop.cc");
  CG_outputRepr *list_type =
    static_cast<CG_chillBuilder *>(ocg)->CreateLinkedListStruct("a_list", // after exit()
                                                               class_data, class_data_types);

  size_repr3.push_back(
                       ocg->CreateTimes(count->clone(), ocg->CreateInt(size)));
  std::vector<CG_outputRepr *> to_push;
  to_push.push_back(ocg->CreateInt(0));
  new_array_prime2 = ir->CreatePointerSymbol(list_type, to_push);
  new_array_prime = ir->CreatePointerSymbol(type, size_repr3, new_array);
  new_array_prime->set_size(0, ocg->CreateTimes(count_2->clone(), ocg->CreateInt(size)));
  ptr_variables.push_back(static_cast<IR_PointerSymbol *>(new_array_prime));
  
  //init_code = ocg->StmtListAppend(init_code,
  //    ir->CreateMalloc(type, new_array_prime->name(),
  //        size_repr3[0]->clone()));
  init_code = ocg->StmtListAppend(init_code,
                                  ocg->CreateAssignment(0, ocg->CreateIdent(new_array_prime2->name()),
                                                        ocg->CreateInt(0)));
  std::vector<CG_outputRepr *> size_repr4;
  size = 1;
  
  // Relation R = extract_upper_bound(i->second, i->second.set_var(i->first));
  int lb = -1, ub = -1;
  Relation R = get_loop_bound(stmt[stmt_num].IS, level, this->known);
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      if ((*gi).get_coef(R.set_var(level)) < 0
          && (*gi).is_const(R.set_var(level))) {
        
        ub = (*gi).get_const();
      } else if ((*gi).get_coef(R.set_var(level)) > 0
                 && (*gi).is_const(R.set_var(level))) {
        
        lb = (*gi).get_const();
      }
    }
  
  if (lb < 0 || ub < 0)
    throw loop_error("bounds of inner loops cannot be determined");
  
  size *= (ub - lb + 1);
  
  std::vector<std::string> mk_data;
  std::vector<CG_outputRepr *> mk_data_types;
  mk_data.push_back("ptr");
  mk_data_types.push_back(ir->CreatePointerType(list_type));
  
  throw std::runtime_error( "about to use ROSE Builder in generic loop.cc\n");
  // TODO roseBuilder
  CG_outputRepr *mk_type = static_cast<CG_chillBuilder *>(ocg)->CreateClass( // after exit();
                                                                           "mk", mk_data, mk_data_types);
  //mk_type = ir->CreatePointerType(mk_type);
  std::vector<CG_outputRepr *> size_repr5;
  size_repr5.push_back(ocg->CreateInt(size));
  IR_PointerSymbol *marked2 = ir->CreatePointerSymbol(mk_type, size_repr5);
  init_code = ocg->StmtListAppend(init_code,
                                  ir->CreateMalloc(mk_type, marked2->name(), ocg->CreateInt(size)));
  
  /*  size_repr4.push_back(ocg->CreateInt(size));
      marked = ir->CreatePointerSymbol(IR_CONSTANT_INT, size_repr4);
      ptr_variables.push_back(static_cast<IR_PointerSymbol *>(marked));
      init_code = ocg->StmtListAppend(init_code,
      ir->CreateMalloc(IR_CONSTANT_INT, marked->name(),
      size_repr4[0]->clone()));
      marked->set_size(0, size_repr4[0]->clone());
  */
  // 8. Insert count = 0 and offset_index[0] = 0 above loop. throw error if level != 2
  
  //  CG_outputRepr *count_2 = ocg->CreateIdent("chill_count_1");
  CG_outputRepr *count_init = ocg->CreateAssignment(0, count_2->clone(),
                                                    ocg->CreateInt(0));
  
  CG_outputRepr *offset_init = NULL;
  
  if(level > 1)
    offset_init = ocg->CreateAssignment(0,
                                        ocg->CreateArrayRefExpression(offset_index2->name(),
                                                                      ocg->CreateInt(0)), ocg->CreateInt(0));
  
  //if (level == 2) {
  
  init_code = ocg->StmtListAppend(init_code, count_init);
  
  if (level > 1)
    init_code = ocg->StmtListAppend(init_code, offset_init);
  if (level > 2) {
    
    std::vector<int> outer_loops;
    
    for (int i = 1; i < level; i++)
      outer_loops.push_back(i);
    
    std::pair<Relation, Relation> IS_and_XFORM =
      construct_reduced_IS_And_XFORM(ir, stmt[stmt_num].IS,
                                     stmt[stmt_num].xform, outer_loops, lex, this->known,
                                     uninterpreted_symbols[stmt_num]);
    
    Statement s = stmt[stmt_num];
    s.IS = IS_and_XFORM.first;
    s.xform = IS_and_XFORM.second;
    s.has_inspector = false;
    s.ir_stmt_node = NULL;
    s.reduction = 0;
    
    CG_outputRepr *new_code = ocg->StmtListAppend(count_init, offset_init);
    delete s.code;
    s.code = new_code;
    std::vector<LoopLevel> ll;
    
    for (int i = 0; i < outer_loops.size(); i++)
      ll.push_back(s.loop_level[i]);
    
    s.loop_level = ll;
    
    debug_fprintf(stderr, "loop.cc L5669 adding stmt %d\n", stmt.size()); 
    stmt.push_back(s);

    uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
    uninterpreted_symbols_stringrepr.push_back(
                                               uninterpreted_symbols_stringrepr[stmt_num]);
    dep.insert();
    //shiftLexicalOrder(lex, 0, 1);
  }
  
  // 9. Cancel loops and initialize mark before computation loop distribute at level kk
  //10. Marked = 0;
  
  std::set<int> cancelled_loops;
  std::set<int> remaining_loops;
  for (std::map<int, CG_outputRepr *>::iterator it = most_reprs.begin();
       it != most_reprs.end(); it++)
    cancelled_loops.insert(it->first);
  
  for (int i = 1; i <= stmt[stmt_num].loop_level.size(); i++)
    if (cancelled_loops.find(i) == cancelled_loops.end())
      remaining_loops.insert(i);
  
  CG_outputRepr *mark_code = NULL;
  
  std::set<int> guard_vars;
  std::set<int> unprojected_vars;
  std::set<int> derived_vars;
  CG_outputRepr * new_code, *new_code2;
  for (int it = 0; it < name_to_pos.size(); it++)
    guard_vars.insert(name_to_pos[it].second);
  
  for (std::map<int, int>::iterator it = dont_project.begin();
       it != dont_project.end(); it++)
    unprojected_vars.insert(it->second);
  
  for (std::map<int, CG_outputRepr*>::iterator it = most_reprs.begin();
       it != most_reprs.end(); it++)
    derived_vars.insert(it->first);
  
  bool check_if = (guard_vars.size()
                   == (derived_vars.size() + unprojected_vars.size()));
  
  if (most_reprs.find(level) != most_reprs.end()) {
    
    CG_outputRepr * level_repr = most_reprs.find(level)->second;
    mark_code = ocg->StmtListAppend(mark_code, level_repr->clone());
    
    CG_outputRepr *mark_assign2 =
      ocg->CreateAssignment(0,
                            static_cast<CG_chillBuilder *>(ocg)->CreateDotExpression( // after exit()
                                                                                    ocg->CreateArrayRefExpression(marked2->name(),
                                                                                                                  ocg->CreateIdent(
                                                                                                                                   stmt[stmt_num].IS.set_var(level)->name())),
                                                                                    static_cast<CG_chillBuilder *>(ocg)
                                                                                        ->lookup_member_data(
                                                                                                                                           mk_type, "ptr",
                                                                                                                                           ocg->CreateIdent(marked2->name()))),
                            ocg->CreateInt(0));
    /*  CG_outputRepr *mark_assign = ocg->CreateAssignment(0,
        ocg->CreateArrayRefExpression(marked2->name(),
        ocg->CreateIdent(
        stmt[stmt_num].IS.set_var(level)->name())),
        ocg->CreateInt(-1));
        new_code = ocg->StmtListAppend(mark_code->clone(), mark_assign);
    */
    new_code2 = ocg->StmtListAppend(mark_code->clone(), mark_assign2);
    //new_code = mark_code->clone();
  } else {
    
    for (std::map<int, CG_outputRepr *>::iterator i = most_reprs.begin();
         i != most_reprs.end(); i++)
      mark_code = ocg->StmtListAppend(mark_code, i->second->clone());
    CG_outputRepr *cond = dynamic_cast<IR_If*>(code)->condition();
    assert(cond != NULL);
    /*  CG_outputRepr *mark_assign = ocg->CreateAssignment(0,
        ocg->CreateArrayRefExpression(marked2->name(),
        ocg->CreateIdent(
        stmt[stmt_num].IS.set_var(level)->name())),
        ocg->CreateInt(-1));
    */
    //debug_fprintf(stderr, "about to use ROSE Builder in generic loop.cc\n");
    exit(-1); 
    CG_outputRepr *mark_assign2 =
      ocg->CreateAssignment(0,
                            static_cast<CG_chillBuilder *>(ocg)->CreateDotExpression( // after exit()
                                                                                    ocg->CreateArrayRefExpression(marked2->name(),
                                                                                                                  ocg->CreateIdent(
                                                                                                                                   stmt[stmt_num].IS.set_var(level)->name())),
                                                                                    static_cast<CG_chillBuilder *>(ocg)
                                                                                        ->lookup_member_data(
                                                                                                                                           mk_type, "ptr",
                                                                                                                                           ocg->CreateIdent(marked2->name()))),
                            ocg->CreateInt(0));
    //CG_outputRepr * new_if = ocg->CreateIf(0, cond->clone(), mark_assign,
    //NULL);
    CG_outputRepr * new_if2 = ocg->CreateIf(0, cond->clone(), mark_assign2,
                                            NULL);
    if (mark_code != NULL && new_if2 != NULL) {
      //new_code = ocg->StmtListAppend(mark_code->clone(), new_if->clone());
      new_code2 = ocg->StmtListAppend(mark_code->clone(),
                                      new_if2->clone());
    } else if (new_if2 != NULL) {
      //new_code = new_if->clone();
      new_code2 = new_if2->clone();
    }
  }
  
  int count_rem = 1;
  
  for (std::set<int>::iterator it = remaining_loops.begin();
       it != remaining_loops.end(); it++) {
    if (*it >= level)
      break;
    count_rem++;
  }
  
  //11. Insert computation loop after cancellation, cancel guard if possible otherwise throw error
  //12. If marked == 0 increment count, initialize marked[loop_index] to count, set offset_kk[kk] = kk,(use cancelled loops)
  //    initialize inner data footprint(using set of inner loops) to zero.
  //13. Copy value from data_array into new_array.
  CG_outputRepr *marked_check, *explicit_offset_assign,
    *explicit_offset_assign2, *data_init, *data_init2, *data_assign,
    *marked_assign, *count_inc, *marked_check2;
  
  /*  marked_check = ocg->CreateEQ(
      ocg->CreateArrayRefExpression(marked->name(),
      ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name())),
      ocg->CreateInt(-1));
  */
  //debug_fprintf(stderr, "about to use ROSE Builder in generic loop.cc\n");
  exit(-1); 
  marked_check2 = ocg->CreateEQ(
                                static_cast<CG_chillBuilder *>(ocg)->CreateDotExpression( // after exit()
                                                                                        ocg->CreateArrayRefExpression(marked2->name(),
                                                                                                                      ocg->CreateIdent(
                                                                                                                                       stmt[stmt_num].IS.set_var(level)->name())),
                                                                                        static_cast<CG_chillBuilder *>
                                                                                        (ocg)->lookup_member_data(
                                                                                                                                               mk_type, "ptr", ocg->CreateIdent(marked2->name()))),
                                ocg->CreateInt(0));
  
  std::vector<IR_ScalarSymbol *> loop_iters;
  int level_coef = 1;
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    loop_iters.push_back(ir->CreateScalarSymbol(IR_CONSTANT_INT, 0));
    level_coef *= inner_loop_bounds[i].second - inner_loop_bounds[i].first
      + 1;
  }
  
  /*temp = (float_list*)malloc(sizeof(float_list));
    temp->next = a_list;
    a_list = temp;
    mk[kk].ptr = a_list;
    for (i_ = 0; i_ < R; i_++)
    for (k_ = 0; k_ < C; k_++)
    mk[kk].ptr->data[i_ * C + k_] = 0;
    mk[kk].ptr->col = kk;
    count++;
  */
  std::vector<CG_outputRepr *> size__;
  size__.push_back(ocg->CreateInt(1));
  IR_PointerSymbol *temp = ir->CreatePointerSymbol(list_type, size__);
  
  CG_outputRepr *allocate = ir->CreateMalloc(list_type, temp->name(),
                                             ocg->CreateInt(1));
  
  allocate =
    ocg->StmtListAppend(allocate,
                        ocg->CreateAssignment(0,
                                              static_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression(
                                                                                                           ocg->CreateIdent(temp->name()),
                                                                                                           static_cast<CG_chillBuilder *>(ocg)->lookup_member_data(
                                                                                                                                                                  list_type, "next",
                                                                                                                                                                  ocg->CreateIdent(temp->name()))),
                                              ocg->CreateIdent(new_array_prime2->name())));
  
  allocate = ocg->StmtListAppend(allocate,
                                 ocg->CreateAssignment(0, ocg->CreateIdent(new_array_prime2->name()),
                                                       ocg->CreateIdent(temp->name())));
  
  allocate =
    ocg->StmtListAppend(allocate,
                        ocg->CreateAssignment(0,
                                              static_cast<CG_chillBuilder *>(ocg)->CreateDotExpression(
                                                                                                      ocg->CreateArrayRefExpression(
                                                                                                                                    marked2->name(),
                                                                                                                                    ocg->CreateIdent(
                                                                                                                                                     stmt[stmt_num].IS.set_var(
                                                                                                                                                                               level)->name())),
                                                                                                      static_cast<CG_chillBuilder *>(ocg)->lookup_member_data(
                                                                                                                                                             mk_type, "ptr",
                                                                                                                                                             ocg->CreateIdent(marked2->name()))),
                                              ocg->CreateIdent(new_array_prime2->name())));
  
  CG_outputRepr *data_array_ref_2 =
    static_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression(
                                                                 static_cast<CG_chillBuilder *>(ocg)
                                                                     ->CreateDotExpression(
                                                                                                                         ocg->CreateArrayRefExpression(marked2->name(),
                                                                                                                                                       ocg->CreateIdent(
                                                                                                                                                                        stmt[stmt_num].IS.set_var(level)->name())),
                                                                                                                         static_cast<CG_chillBuilder *>(ocg)->lookup_member_data(
                                                                                                                                                                                mk_type, "ptr",
                                                                                                                                                                                ocg->CreateIdent(marked2->name()))),
                                                                 static_cast<CG_chillBuilder *>(ocg)
                                                                     ->lookup_member_data(
                                                                                                                        list_type, "data",
                                                                                                                        ocg->CreateIdent(new_array_prime2->name())));
  
  /*CG_outputRepr *data_array_ref = ocg->CreateTimes(
    ocg->CreateArrayRefExpression(marked->name(),
    ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name())),
    ocg->CreateInt(level_coef));
  */
  CG_outputRepr *temp2=NULL;
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    CG_outputRepr *current = ocg->CreateIdent(loop_iters[i]->name());
    int level_coef = 1;
    for (int j = i + 1; j < inner_loop_bounds.size(); j++)
      level_coef *= inner_loop_bounds[j].second
        - inner_loop_bounds[j].first + 1;
    
    current = ocg->CreateTimes(ocg->CreateInt(level_coef), current);
    
    //data_array_ref = ocg->CreatePlus(data_array_ref, current->clone());
    if (i == 0)
      temp2 = current->clone();
    else
      temp2 = ocg->CreatePlus(temp2, current->clone());
  }
  CG_outputRepr *data_array_ref_2_cpy = data_array_ref_2->clone();
  if(temp2 != NULL)
    data_array_ref_2 = ocg->CreateArrayRefExpression(data_array_ref_2, temp2);
  else
    data_array_ref_2 = ocg->CreateArrayRefExpression(data_array_ref_2, ocg->CreateInt(0));
  
  //data_init = ocg->CreateAssignment(0,
  //    ocg->CreateArrayRefExpression(new_array, data_array_ref->clone()),
  //    ocg->CreateInt(0));
  data_init2 = ocg->CreateAssignment(0, data_array_ref_2->clone(),
                                     ocg->CreateInt(0));
  
  for (int i = inner_loop_bounds.size() - 1; i >= 0; i--) {
    
    //CG_outputRepr *loop_inductive = ocg->CreateInductive(
    //    ocg->CreateIdent(loop_iters[i]->name()),
    //    ocg->CreateInt(inner_loop_bounds[i].first),
    //    ocg->CreateInt(inner_loop_bounds[i].second), NULL);
    CG_outputRepr *loop_inductive2 = ocg->CreateInductive(
                                                          ocg->CreateIdent(loop_iters[i]->name()),
                                                          ocg->CreateInt(inner_loop_bounds[i].first),
                                                          ocg->CreateInt(inner_loop_bounds[i].second), NULL);
    //data_init = ocg->CreateLoop(0, loop_inductive, data_init);
    data_init2 = ocg->CreateLoop(0, loop_inductive2, data_init2);
  }
  count_inc = ocg->CreatePlusAssignment(0, count_2->clone(),
                                        ocg->CreateInt(1));
  /*  marked_assign = ocg->CreateAssignment(0,
      ocg->CreateArrayRefExpression(marked->name(),
      ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name())),
      count_2->clone());
      explicit_offset_assign = ocg->CreateAssignment(0,
      ocg->CreateArrayRefExpression(explicit_index->name(),
      ocg->CreateArrayRefExpression(marked->name(),
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(level)->name()))),
      ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name()));
  */
  
  explicit_offset_assign2 =
    static_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression(
                                                                 static_cast<CG_chillBuilder *>(ocg)
                                                                     ->CreateDotExpression(
                                                                                                                         ocg->CreateArrayRefExpression(marked2->name(),
                                                                                                                                                       ocg->CreateIdent(
                                                                                                                                                                        stmt[stmt_num].IS.set_var(level)->name())),
                                                                                                                         static_cast<CG_chillBuilder *>(ocg)->lookup_member_data(
                                                                                                                                                                                mk_type, "ptr",
                                                                                                                                                                                ocg->CreateIdent(marked2->name()))),
                                                                 static_cast<CG_chillBuilder *>(ocg)
                                                                     ->lookup_member_data(
                                                                                                                        list_type, "col",
                                                                                                                        ocg->CreateIdent(new_array_prime2->name())));
  explicit_offset_assign2 = ocg->CreateAssignment(0, explicit_offset_assign2,
                                                  ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name()));
  int level_coef_ = 1;
  
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    
    level_coef_ *= inner_loop_bounds[i].second - inner_loop_bounds[i].first
      + 1;
  }
  /*CG_outputRepr *data_array_ref_ = ocg->CreateTimes(
    ocg->CreateArrayRefExpression(marked->name(),
    ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name())),
    ocg->CreateInt(level_coef_));
    //To replace original data array ref in executor (a to a_prime)
    */
  CG_outputRepr *data_prime_ref = ocg->CreateTimes(
                                                   ocg->CreateIdent(stmt[stmt_num].IS.set_var(level)->name()),
                                                   ocg->CreateInt(level_coef_));
  temp2 = NULL;
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    CG_outputRepr *current =
      ocg->CreateIdent(
                       stmt[stmt_num].IS.set_var(
                                                 loops_for_non_zero_block_count[i])->name());
    int level_coef = 1;
    for (int j = i + 1; j < inner_loop_bounds.size(); j++)
      level_coef *= inner_loop_bounds[j].second
        - inner_loop_bounds[j].first + 1;
    
    current = ocg->CreateTimes(ocg->CreateInt(level_coef), current);
    
    //  data_array_ref_ = ocg->CreatePlus(data_array_ref_, current->clone());
    data_prime_ref = ocg->CreatePlus(data_prime_ref, current->clone());
    if (i == 0)
      temp2 = current->clone();
    else
      temp2 = ocg->CreatePlus(temp2, current->clone());
  }
  CG_outputRepr *data_prime_ref2;
  if(temp2 != NULL)
    data_prime_ref2= ocg->CreateArrayRefExpression(
                                                   data_array_ref_2_cpy, temp2);
  else
    data_prime_ref2 = ocg->CreateArrayRefExpression(data_array_ref_2_cpy, ocg->CreateInt(0));
  std::vector<CG_outputRepr *> data_rhs;
  
  for (int i = 0; i < data_arr_ref->n_dim(); i++)
    data_rhs.push_back(data_arr_ref->index(i)->clone());
  
  CG_outputRepr *data_arr_repr = ocg->CreateArrayRefExpression(
                                                               data_arr_ref->name(), data_rhs[0]);
  
  for (int i = 1; i < data_rhs.size(); i++)
    data_arr_repr = ocg->CreateArrayRefExpression(data_arr_repr,
                                                  data_rhs[i]);
  
  //data_assign = ocg->CreateAssignment(0,
  //    ocg->CreateArrayRefExpression(new_array, data_array_ref_->clone()),
  //    data_arr_repr->clone());
  
  CG_outputRepr *data_assign2 = ocg->CreateAssignment(0, data_prime_ref2,
                                                      data_arr_repr->clone());
  
  //CG_outputRepr *replaced_code = NULL;
  
  CG_outputRepr *replaced_code2 = NULL;
  for (std::map<int, CG_outputRepr *>::iterator i = most_reprs.begin();
       i != most_reprs.end(); i++) {
    //replaced_code = ocg->StmtListAppend(replaced_code, i->second->clone());
    replaced_code2 = ocg->StmtListAppend(replaced_code2,
                                         i->second->clone());
  }
  //CG_outputRepr *statement_body = NULL;
  CG_outputRepr *statement_body2 = NULL;
  //statement_body = ocg->StmtListAppend(marked_assign, data_init);
  statement_body2 = ocg->StmtListAppend(allocate, data_init2);
  //statement_body = ocg->StmtListAppend(statement_body,
  //    explicit_offset_assign);
  statement_body2 = ocg->StmtListAppend(statement_body2,
                                        explicit_offset_assign2);
  //statement_body = ocg->StmtListAppend(statement_body, count_inc->clone());
  statement_body2 = ocg->StmtListAppend(statement_body2, count_inc->clone());
  //statement_body = ocg->CreateIf(0, marked_check, statement_body, NULL);
  statement_body2 = ocg->CreateIf(0, marked_check2, statement_body2, NULL);
  //statement_body = ocg->StmtListAppend(statement_body, data_assign);
  statement_body2 = ocg->StmtListAppend(statement_body2, data_assign2);
  if (!check_if) {
    CG_outputRepr *cond = dynamic_cast<IR_If*>(code)->condition();
    //statement_body = ocg->CreateIf(0, cond->clone(), statement_body, NULL);
    statement_body2 = ocg->CreateIf(0, cond, statement_body2, NULL);
  }
  
  //statement_body = ocg->StmtListAppend(replaced_code, statement_body);
  statement_body2 = ocg->StmtListAppend(replaced_code2, statement_body2);
  //14. Set offset_index[ii] = count just outside kk loop.
  CG_outputRepr *offset_index_assign = NULL;
  if (level > 1)
    offset_index_assign =
      ocg->CreateAssignment(0,
                            ocg->CreateArrayRefExpression(offset_index2->name(),
                                                          ocg->CreatePlus(
                                                                          ocg->CreateIdent(
                                                                                           stmt[stmt_num].IS.set_var(
                                                                                                                     level - 1)->name()),
                                                                          ocg->CreateInt(1))), count_2->clone());
  
  std::vector<int> remain;
  std::vector<int> remain_for_offset_index_assign;
  
  for (std::set<int>::iterator i = remaining_loops.begin();
       i != remaining_loops.end(); i++)
    if (*i < level)
      remain_for_offset_index_assign.push_back(*i);
  
  for (std::set<int>::iterator i = remaining_loops.begin();
       i != remaining_loops.end(); i++)
    remain.push_back(*i);
  
  std::pair<Relation, Relation> xform_is = construct_reduced_IS_And_XFORM(ir,
                                                                          stmt[stmt_num].IS, stmt[stmt_num].xform, remain, lex, this->known,
                                                                          uninterpreted_symbols[stmt_num]);
  
  std::pair<Relation, Relation> xform_is_2 = construct_reduced_IS_And_XFORM(
                                                                            ir, stmt[stmt_num].IS, stmt[stmt_num].xform,
                                                                            remain_for_offset_index_assign, lex, this->known,
                                                                            uninterpreted_symbols[stmt_num]);
  
  CG_outputRepr *old_code = stmt[stmt_num].code;
  Relation old_IS = stmt[stmt_num].IS;
  Relation old_XFORM = stmt[stmt_num].xform;
  stmt[stmt_num].IS = xform_is.first;
  stmt[stmt_num].xform = xform_is.second;
  //stmt[stmt_num].code = statement_body;
  stmt[stmt_num].code = statement_body2;
  std::vector<LoopLevel> ll0;
  std::vector<LoopLevel> old_loop_level = stmt[stmt_num].loop_level;
  for (std::set<int>::iterator i = remaining_loops.begin();
       i != remaining_loops.end(); i++)
    ll0.push_back(stmt[stmt_num].loop_level[*i - 1]);
  stmt[stmt_num].loop_level = ll0;
  //i. Copy Statement Xform and IS
  Statement s;
  s.IS = stmt[stmt_num].IS;
  s.xform = stmt[stmt_num].xform;
  s.has_inspector = false;
  s.ir_stmt_node = NULL;
  s.reduction = 0;
  std::vector<LoopLevel> ll;
  for (std::set<int>::iterator i = remaining_loops.begin();
       i != remaining_loops.end(); i++)
    ll.push_back(old_loop_level[*i - 1]);
  s.loop_level = ll;
  //s.code = new_code;
  s.code = new_code2;
  lex = getLexicalOrder(stmt_num);
  shiftLexicalOrder(lex, 2 * count_rem - 2, 1);
  int new_stmt_num = stmt.size();

  debug_fprintf(stderr, "loop.cc L6106 adding stmt %d\n", stmt.size()); 
  stmt.push_back(s);

  uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
  uninterpreted_symbols_stringrepr.push_back(
                                             uninterpreted_symbols_stringrepr[stmt_num]);
  dep.insert();
  std::set<int> stmts;
  
  stmts.insert(new_stmt_num);
  stmts.insert(stmt_num);
  apply_xform(stmts);
  lex = getLexicalOrder(stmt_num);
  
  //i. Copy Statement Xform and IS
  
  Statement s2;
  
  if (offset_index_assign) {
    s2.IS = xform_is_2.first;
    s2.xform = xform_is_2.second;
    s2.has_inspector = false;
    s2.ir_stmt_node = NULL;
    s2.reduction = 0;
    std::vector<LoopLevel> ll2;
    for (std::set<int>::iterator i = remaining_loops.begin();
         i != remaining_loops.end(); i++)
      if (*i < level)
        ll2.push_back(stmt[stmt_num].loop_level[*i - 1]);
    s2.loop_level = ll2;
    s2.code = offset_index_assign;
    
    assign_const(s2.xform, 2 * count_rem - 2, lex[2 * count_rem - 2] + 1);
    
    debug_fprintf(stderr, "loop.cc L6140 adding stmt %d\n", stmt.size()); 
    stmt.push_back(s2);

    dep.insert();
    uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
    uninterpreted_symbols_stringrepr.push_back(
                                               uninterpreted_symbols_stringrepr[stmt_num]);
  }
  
  //14b. Create Malloc code after exact size has been determined
  
  //assign_const(s3.xform, 0, lex[0] + 1);
  //Anand: Hack to shift lexical Order
  lex[0] += 1;
  
  shiftLexicalOrder(lex, 0, 1);
  
  Statement m_s[3];
  
  for (int i = 0; i < 3; i++) {
    //  m_s[i].IS = xform_is_2.first;
    //  s2.xform = xform_is_2.second;
    m_s[i].has_inspector = false;
    m_s[i].ir_stmt_node = NULL;
    m_s[i].reduction = 0;
    std::vector<LoopLevel> ll2;
    
    uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
    uninterpreted_symbols_stringrepr.push_back(
                                               uninterpreted_symbols_stringrepr[stmt_num]);
    dep.insert();
    
  }
  
  CG_outputRepr *malloced_stmts, *dc_stmts, *index_cp_stmt, *ll_inc_and_free;
  
  malloced_stmts = ir->CreateMalloc(IR_CONSTANT_INT, explicit_index->name(),
                                    count_2->clone());
  malloced_stmts = ocg->StmtListAppend(malloced_stmts,
                                       ir->CreateMalloc(IR_CONSTANT_FLOAT, new_array_prime->name(),
                                                        ocg->CreateTimes(count_2->clone(),
                                                                         ocg->CreateInt(level_coef_))));
  Relation IS_ms(1), IS_dc(1 + inner_loop_bounds.size()), IS_ll(1 + inner_loop_bounds.size());
  Relation xform_ms(1, 3), xform_dc(1 + inner_loop_bounds.size(),
                                    2 * (1 + inner_loop_bounds.size()) + 1), xform_ll(1+ inner_loop_bounds.size(), 2*(1+ inner_loop_bounds.size()) + 1);
  
  //prepare malloced stmts relation
  
  F_And *root_ms = IS_ms.add_and();
  EQ_Handle e1 = root_ms->add_EQ();
  e1.update_coef(IS_ms.set_var(1), 1);
  
  Free_Var_Decl *cc1 = new Free_Var_Decl("chill_count_1");
  Variable_ID tmp_fv = IS_ms.get_local(cc1);
  freevar.push_back(cc1);
  e1.update_coef(tmp_fv, 1);
  e1.update_const(-1);
  
  Relation known_(stmt[stmt_num].IS.n_set());
  known_.copy_names(copy(stmt[stmt_num].IS.n_set()));
  known_.setup_names();
  
  F_And *fr = known_.add_and();
  GEQ_Handle g = fr->add_GEQ();
  Variable_ID tmp_fv_ = known_.get_local(cc1);
  g.update_coef(tmp_fv_, 1);
  g.update_const(-1);
  this->addKnown(known_);
  
  F_And *root_ms_x = xform_ms.add_and();
  F_And *root_dc_x = xform_dc.add_and();
  for (int i = 1; i <= 3; i += 2) {
    EQ_Handle e = root_ms_x->add_EQ();
    e.update_coef(xform_ms.output_var(i), 1);
    if (i == 1)
      e.update_const(-lex[0]);
  }
  EQ_Handle e2 = root_ms_x->add_EQ();
  e2.update_coef(xform_ms.output_var(2), 1);
  e2.update_coef(xform_ms.input_var(1), -1);
  m_s[0].IS = IS_ms;
  m_s[0].xform = xform_ms;
  std::vector<LoopLevel> empty_ll;
  m_s[0].loop_level = empty_ll;
  m_s[0].code = malloced_stmts;
  
  F_And *root_dc = IS_dc.add_and();
  GEQ_Handle g1 = root_dc->add_GEQ();
  g1.update_coef(IS_dc.set_var(1), 1);
  Variable_ID tmp_fv1 = IS_dc.get_local(cc1);
  g1.update_coef(tmp_fv1, 1);
  g1.update_const(-1);
  GEQ_Handle g2 = root_dc->add_GEQ();
  g2.update_coef(IS_dc.set_var(1), -1);
  //g2.update_const(1);
  
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    int ub = inner_loop_bounds[i].second;
    int lb = inner_loop_bounds[i].first;
    
    GEQ_Handle g1 = root_dc->add_GEQ();
    g1.update_coef(IS_dc.set_var(i + 2), 1);
    g1.update_const(-lb);
    
    GEQ_Handle g2 = root_dc->add_GEQ();
    g2.update_coef(IS_dc.set_var(i + 2), -1);
    g2.update_const(ub);
    
    EQ_Handle e = root_dc_x->add_EQ();
    e.update_coef(xform_dc.output_var(2 * i + 3), 1);
    
    EQ_Handle e2 = root_dc_x->add_EQ();
    e2.update_coef(xform_dc.output_var(2 * (i + 2)), 1);
    e2.update_coef(xform_dc.input_var(i + 2), -1);
  }
  
  EQ_Handle e3 = root_dc_x->add_EQ();
  e3.update_coef(xform_dc.output_var(xform_dc.n_out()), 1);
  
  EQ_Handle e4 = root_dc_x->add_EQ();
  e4.update_coef(xform_dc.output_var(1), 1);
  e4.update_const(-lex[0]);
  
  EQ_Handle e5 = root_dc_x->add_EQ();
  e5.update_coef(xform_dc.output_var(2), 1);
  e5.update_coef(xform_dc.input_var(1), -1);
  
  F_And *root_ll = IS_ll.add_and();
  GEQ_Handle g3 = root_ll->add_GEQ();
  g3.update_coef(IS_ll.set_var(1), 1);
  Variable_ID tmp_fv3 = IS_ll.get_local(cc1);
  g3.update_coef(tmp_fv3, 1);
  g3.update_const(-1);
  
  F_And *root_ll_x = xform_ll.add_and();
  GEQ_Handle g4 = root_ll->add_GEQ();
  g4.update_coef(IS_ll.set_var(1), -1);
  
  
  
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    int ub = inner_loop_bounds[i].second;
    
    
    EQ_Handle g1 = root_ll->add_EQ();
    g1.update_coef(IS_ll.set_var(i + 2), 1);
    g1.update_const(-ub);
    
    
    EQ_Handle e = root_ll_x->add_EQ();
    e.update_coef(xform_ll.output_var(2 * i + 3), 1);
    e.update_const(-1);
    EQ_Handle e2 = root_ll_x->add_EQ();
    e2.update_coef(xform_ll.output_var(2 * (i + 2)), 1);
    e2.update_coef(xform_ll.input_var(i + 2), -1);
  }
  
  
  
  
  for (int i = 1; i <= 1; i += 1) {
    EQ_Handle e = root_ll_x->add_EQ();
    e.update_coef(xform_ll.output_var(i), 1);
    e.update_const(-lex[0]);
  }
  
  
  
  EQ_Handle e_4 = root_ll_x->add_EQ();
  e_4.update_coef(xform_ll.output_var(2), 1);
  e_4.update_coef(xform_ll.input_var(1), -1);
  
  EQ_Handle e_5 = root_ll_x->add_EQ();
  e_5.update_coef(xform_ll.output_var(xform_ll.n_out()), 1);
  
  //data copy from linked list to new_array_prime
  CG_outputRepr *new_data_array_ref_ = ocg->CreateTimes(ocg->CreateMinus(NULL, ocg->CreateIdent(IS_dc.set_var(1)->name())),
                                                        ocg->CreateInt(level_coef_));
  CG_outputRepr * temp2_=ocg->CreateInt(0);
  //0. index expression creation
  for (int i = 0; i < inner_loop_bounds.size(); i++) {
    CG_outputRepr *current = ocg->CreateIdent(IS_dc.set_var(i + 2)->name());
    int level_coef = 1;
    for (int j = i + 1; j < inner_loop_bounds.size(); j++)
      level_coef *= inner_loop_bounds[j].second
        - inner_loop_bounds[j].first + 1;
    
    current = ocg->CreateTimes(ocg->CreateInt(level_coef), current);
    
    new_data_array_ref_ = ocg->CreatePlus(new_data_array_ref_,
                                          current->clone());
    if (i == 0)
      temp2_ = current->clone();
    else
      temp2_ = ocg->CreatePlus(temp2_, current->clone());
  }
  
  //1. lhs array ref
  
  CG_outputRepr *lhs = ocg->CreateArrayRefExpression(new_array_prime->name(),
                                                     new_data_array_ref_);
  CG_outputRepr *rhs =
    dynamic_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression(new_array_prime2->name(),
                                                                  dynamic_cast<CG_chillBuilder *>(ocg)
                                                                      ->lookup_member_data(list_type,
                                                                                                                          "data",
                                                                                                                          ocg->CreateIdent(new_array_prime2->name())));
  rhs = ocg->CreateArrayRefExpression(rhs, temp2_->clone());
  dc_stmts = ocg->CreateAssignment(0, lhs->clone(), rhs->clone());
  
  // copy col
  CG_outputRepr *lhs_col = ocg->CreateArrayRefExpression(explicit_index->name(),
                                                         ocg->CreateMinus(NULL, ocg->CreateIdent(IS_ll.set_var(1)->name())));
  CG_outputRepr *rhs_col =
    dynamic_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression(new_array_prime2->name(),
                                                                  dynamic_cast<CG_chillBuilder *>(ocg)
                                                                      ->lookup_member_data(list_type,
                                                                                                                          "col",
                                                                                                                          ocg->CreateIdent(new_array_prime2->name())));
  index_cp_stmt = ocg->CreateAssignment(0, 
                                        lhs_col->clone(),
                                        rhs_col->clone());
  
  ll_inc_and_free = ocg->CreateAssignment(0, 
                                          ocg->CreateIdent(temp->name()),
                                          dynamic_cast<CG_chillBuilder *>(ocg)->CreateArrowRefExpression
                                              (new_array_prime2->name(),
                                                                                                        dynamic_cast<CG_chillBuilder *>(ocg)->lookup_member_data(
                                                                                                                                                                list_type, 
                                                                                                                                                                "next",
                                                                                                                                                                ocg->CreateIdent(new_array_prime2->name()))));

  ll_inc_and_free = ocg->StmtListAppend(ll_inc_and_free,
                                        ir->CreateFree(ocg->CreateIdent(new_array_prime2->name())));
  ll_inc_and_free = ocg->StmtListAppend(ll_inc_and_free,
                                        ocg->CreateAssignment(0, ocg->CreateIdent(new_array_prime2->name()),
                                                              ocg->CreateIdent(temp->name())));
  ll_inc_and_free = ocg->StmtListAppend(index_cp_stmt, ll_inc_and_free);
  
  m_s[1].IS = IS_dc;
  m_s[1].xform = xform_dc;
  std::vector<LoopLevel> empty_ll2;
  for (int i = 0; i <= inner_loop_bounds.size(); i++) {
    LoopLevel tmp;
    tmp.payload = i;
    tmp.type = LoopLevelOriginal;
    tmp.parallel_level = 0;
    tmp.segreducible = false;
    empty_ll2.push_back(tmp);
  }
  m_s[1].loop_level = empty_ll2;
  m_s[1].code = dc_stmts;
  
  
  m_s[2].IS = IS_ll;
  m_s[2].xform = xform_ll;
  std::vector<LoopLevel> empty_ll3;
  //for (int i = 0; i <= inner_loop_bounds.size(); i++) {
  LoopLevel tmp;
  tmp.payload = 0;
  tmp.type = LoopLevelOriginal;
  tmp.parallel_level = 0;
  tmp.segreducible = false;
  empty_ll3.push_back(tmp);
  //}
  m_s[2].loop_level = empty_ll3;
  m_s[2].code = ll_inc_and_free;
  
  for(int i=0; i < 3;i++) { 
    debug_fprintf(stderr, "loop.cc L6406 adding stmt %d\n", stmt.size()); 
    stmt.push_back(m_s[i]);
  }
  
  
  
  //15. Create Executor Code
  
  std::vector<Relation> outer_loop_bounds;
  std::map<int, Relation> zero_loop_bounds;
  for (int i = 1; i < level; i++) {
    outer_loop_bounds.push_back(get_loop_bound(old_IS, i, this->known));
    if (_DEBUG_)
      get_loop_bound(old_IS, i, this->known).print();
  }
  for (int i = 0; i < loops_for_non_zero_block_count.size(); i++) {
    zero_loop_bounds.insert(
                            std::pair<int, Relation>(loops_for_non_zero_block_count[i],
                                                     get_loop_bound(old_IS,
                                                                    loops_for_non_zero_block_count[i],
                                                                    this->known)));
    if (_DEBUG_)
      get_loop_bound(old_IS, loops_for_non_zero_block_count[i],
                     this->known).print();
  }
  
  std::string index_name;
  if (level > 1)
    index_name = offset_index2->name();
  else
    index_name = "chill_count_1";
  
  std::pair<Relation, Relation> xform_is_3 = createCSRstyleISandXFORM(ocg,
                                                                      outer_loop_bounds, index_name, zero_loop_bounds,
                                                                      uninterpreted_symbols[stmt_num],
                                                                      uninterpreted_symbols_stringrepr[stmt_num], this);
  
  assert(dynamic_cast<IR_If*>(code) != NULL);
  
  CG_outputRepr *assignment =
    dynamic_cast<IR_If*>(code)->then_body()->extract();
  
  std::vector<IR_ArrayRef *> array_refs = ir->FindArrayRef(assignment);
  
  for (int i = 0; i < array_refs.size(); i++) {
    if (array_refs[i]->name() != data_array) {
      for (int j = 0; j < array_refs[i]->n_dim(); j++) {
        std::vector<IR_ScalarRef *> scalar_refs = ir->FindScalarRef(
                                                                    array_refs[i]->index(j));
        for (int k = 0; k < scalar_refs.size(); k++)
          if (scalar_refs[k]->name() == old_IS.set_var(level)->name())
            ir->ReplaceExpression(scalar_refs[k],
                                  ocg->CreateArrayRefExpression(
                                                                explicit_index->name(),
                                                                ocg->CreateIdent(
                                                                                 old_IS.set_var(level)->name())));
      }
    } else
      ir->ReplaceExpression(array_refs[i],
                            ocg->CreateArrayRefExpression(new_array, data_prime_ref));
    
  }
  
  Statement s3;
  s3.IS = xform_is_3.first;
  s3.xform = xform_is_3.second;
  s3.has_inspector = false;
  s3.ir_stmt_node = NULL;
  s3.reduction = 0;
  std::vector<LoopLevel> ll3;
  for (int i = 0; i < level - 1; i++)
    ll3.push_back(old_loop_level[i]);
  LoopLevel tmp_;
  tmp_.payload = level - 1;
  tmp_.type = LoopLevelOriginal;
  tmp_.parallel_level = 0;
  tmp_.segreducible = false;
  ll3.push_back(tmp_);
  
  for (int i = 0; i < loops_for_non_zero_block_count.size(); i++)
    ll3.push_back(old_loop_level[loops_for_non_zero_block_count[i] - 1]);
  
  s3.loop_level = ll3;
  s3.code = assignment;
  
  assign_const(s3.xform, 0, lex[0] + 1);
  //Anand: Hack to shift lexical Order
  lex[0] += 1;
  
  shiftLexicalOrder(lex, 0, 1);
  //    int new_num = stmt.size();
  
  //  std::vector<int> lex3 = getLexicalOrder(new_num);
  
  debug_fprintf(stderr, "loop.cc L6500 adding stmt %d\n", stmt.size()); 
  stmt.push_back(s3);

  uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
  uninterpreted_symbols_stringrepr.push_back(
                                             uninterpreted_symbols_stringrepr[stmt_num]);
  dep.insert();
  
  if(level > 1) 
    cleanup_code = ocg->StmtListAppend(cleanup_code,
                                       ir->CreateFree(ocg->CreateIdent(offset_index2->name())));
  cleanup_code = ocg->StmtListAppend(cleanup_code,
                                     ir->CreateFree(ocg->CreateIdent(explicit_index->name())));
  cleanup_code = ocg->StmtListAppend(cleanup_code,
                                     ir->CreateFree(ocg->CreateIdent(new_array_prime->name())));
  cleanup_code = ocg->StmtListAppend(cleanup_code,
                                     ir->CreateFree(ocg->CreateIdent(marked2->name())));
  num_dep_dim = ll3.size();
  //lex = getLexicalOrder(stmt_num);
  
  //shiftLexicalOrder(lex, 2 * count_rem - 2, 1);
  //stmt.push_back(s);
  
  /*  apply_xform(stmt_num);
      
      IR_CONSTANT_TYPE type;
      
      IR_ArrayRef *arr_sym = NULL;
      IR_PointerArrayRef *ptr_sym = NULL;
      
      bool found = false;
      std::vector<IR_PointerArrayRef *> arrRefs = ir->FindPointerArrayRef(
      stmt[stmt_num].code);std::map<std::string, int>
      
      for (int i = 0; i < arrRefs.size(); i++)
      if (data_array == arrRefs[i]->name()) {
      type = arrRefs[i]->symbol()->elem_type();
      found = true;
      ptr_sym = arrRefs[i];
      break;
      }
      
      if (!found) {
      
      std::vector<IR_ArrayRef *> arrRefs = ir->FindArrayRef(
      stmt[stmt_num].code);
      for (int i = 0; i < arrRefs.size(); i++)
      if (data_array == arrRefs[i]->name()) {
      type = arrRefs[i]->symbol()->elem_type();
      found = true;
      arr_sym = arrRefs[i];
      break;
      }
      
      }
      if (!found)
      throw loop_error("data array " + data_array + " not found!");
      if (outer_loop_levels.size() < 2)
      throw loop_error("Compaction requires at least 2 input loop levels!");
      
      if (outer_loop_levels[0] != 1)
      throw loop_error("loop levels must start from outer most loop level");
      for (int i = 1; i < outer_loop_levels.size(); i++) {
      if (outer_loop_levels[i] != outer_loop_levels[i - 1] + 1)
      throw loop_error(
      "Input loop levels for compaction must be continuous");
      if (outer_loop_levels[i] < 0
      || outer_loop_levels[i] > stmt[stmt_num].loop_level.size())
      throw loop_error("loop levels out of bounds");
      }
      std::vector<int> zero_entity_levels;
      
      if (outer_loop_levels[outer_loop_levels.size() - 1]
      == stmt[stmt_num].loop_level.size())
      zero_entity_levels.push_back(
      outer_loop_levels[outer_loop_levels.size() - 1]);
      else
      for (int i = outer_loop_levels[outer_loop_levels.size() - 1] + 1;
      i <= stmt[stmt_num].loop_level.size(); i++)
      zero_entity_levels.push_back(i);
      
      int compressed_level = outer_loop_levels[outer_loop_levels.size() - 1];
      int row_level = outer_loop_levels[outer_loop_levels.size() - 2];
      
      //1. discover footprint of outer loops for declaring reorganized array, 
      // declare new data array, col array and index array of appropriate size
      
      Relation rows = get_loop_bound(stmt[stmt_num].IS, row_level - 1);
      Relation column = get_loop_bound(stmt[stmt_num].IS, compressed_level - 1);
      
      int nnz;
      int num_rows;
      int num_cols;
      Variable_ID v = rows.set_var(row_level);
      for (GEQ_Iterator e(const_cast<Relation &>(rows).single_conjunct()->GEQs());
      e; e++)
      if ((*e).get_coef(v) < 0)
      num_rows = (*e).get_const() + 1;
      
      Variable_ID v2 = column.set_var(compressed_level);
      for (GEQ_Iterator e(
      const_cast<Relation &>(column).single_conjunct()->GEQs()); e; e++)
      if ((*e).get_coef(v2) < 0)
      num_cols = (*e).get_const() + 1;
      
      nnz = num_rows * num_cols;
      std::vector<int> outer_loop_bounds;
      std::vector<int> zero_entity_loop_bounds;
      int data_size = nnz;
      int index_size = num_rows + 1;
      ;
      for (int i = 0; i < outer_loop_levels.size() - 2; i++) {
      Relation bound = get_loop_bound(stmt[stmt_num].IS,
      2 * outer_loop_levels[i] - 1);
      Variable_ID v = bound.set_var(outer_loop_levels[i]);
      for (GEQ_Iterator e(
      const_cast<Relation &>(bound).single_conjunct()->GEQs()); e;
      e++)
      if ((*e).get_coef(v) < 0)
      outer_loop_bounds.push_back((*e).get_const() + 1);
      
      data_size *= outer_loop_bounds[i];
      index_size *= outer_loop_bounds[i];
      }IR_PointerSymbol *offset_index, *explicit_index, *new_array_prime, *marked;
      
      if (zero_entity_levels.size() > 1) {
      for (int i = 0; i < zero_entity_levels.size(); i++) {
      Relation bound = get_loop_bound(stmt[stmt_num].IS,
      zero_entity_levels[i] - 1);
      Variable_ID v = bound.set_var(zero_entity_levels[i]);
      for (GEQ_Iterator e(
      const_cast<Relation &>(bound).single_conjunct()->GEQs()); e;
      e++)
      if ((*e).get_coef(v) < 0 && (*e).get_const() > 0) {
      zero_entity_loop_bounds.push_back((*e).get_const() + 1);
      break;
      }
      data_size *= zero_entity_loop_bounds[i];
      }
      } else
      zero_entity_loop_bounds.push_back(1);
      
      int col_size = data_size;
      
      CG_roseBuilder *ocg = dynamic_cast<CG_roseBuilder*>(ir->builder());
      
      IR_PointerSymbol *new_col;
      IR_PointerSymbol *new_data;
      IR_PointerSymbol *new_index;
      std::vector<CG_outputRepr*> dims1;
      std::vector<CG_outputRepr*> dims2;
      std::vector<CG_outputRepr*> dims3;
      dims1.push_back(ocg->CreateInt(col_size));
      dims2.push_back(ocg->CreateInt(data_size));
      dims3.push_back(ocg->CreateInt(index_size));
      new_col = ir->CreatePointerSymbol(IR_CONSTANT_INT, dims1);
      new_data = ir->CreatePointerSymbol(type, dims2);
      new_index = ir->CreatePointerSymbol(IR_CONSTANT_INT, dims3);
      new_col->set_size(0, ocg->CreateInt(col_size));
      new_data->set_size(0, ocg->CreateInt(data_size));
      new_index->set_size(0, ocg->CreateInt(index_size));
      
      //2. Loop over outer loops and copy data and generate the inspector loop
      if (zero_entity_levels.size() > 1) {
      CG_outputRepr *stmt_1, *stmt_2, *stmt_3, *stmt_4, *stmt_5, *stmt_6;
      CG_outputRepr *index_access = NULL;
      
      //i. bcsr_ind[0] = 0;
      
      CG_outputRepr* index_expr = NULL;
      
      for (int i = 0; i < outer_loop_levels.size() - 2; i++) {
      CG_outputRepr *indice = ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(outer_loop_levels[i])->name());
      for (int j = i + 1; j < outer_loop_levels.size() - 2; j++)
      indice = ocg->CreateTimes(ocg->CreateInt(outer_loop_bounds[j]),
      indice->clone());
      
      if (index_expr == NULL)
      index_expr = indice;
      else
      index_expr = ocg->CreatePlus(index_expr->clone(),
      indice->clone());
      
      }
      
      if (index_expr == NULL) {
      index_expr = ocg->CreateInt(0);
      index_access = ocg->CreateInt(0);
      } else
      index_access = index_expr->clone();
      
      stmt_1 = ocg->CreateArrayRefExpression(new_index->name(),
      index_expr->clone());
      
      Statement s1 = stmt[stmt_num];
      
      s1.xform = copy(stmt[stmt_num].xform);
      s1.code = ocg->CreateAssignment(0, stmt_1->clone(), ocg->CreateInt(0));
      for (int i = outer_loop_levels[outer_loop_levels.size() - 2];
      i <= stmt[stmt_num].loop_level.size(); i++)
      assign_const(s1.xform, 2 * i - 1, 0);
      s1.xform.simplify();
      //stmt.push_back(s1);
      
      std::vector<int> lex_ = getLexicalOrder(stmt_num);
      //ii. bcsr_ind[ii+1] = bcsr_ind[ii];
      
      CG_outputRepr *index_access2, *index_access3, *index_expr2,
      *index_expr3;
      
      index_access2 =
      ocg->CreatePlus(index_access->clone(),
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(
      outer_loop_levels[outer_loop_levels.size()
      - 2])->name()));
      index_access3 = ocg->CreatePlus(index_access2->clone(),
      ocg->CreateInt(1));
      
      index_expr2 = ocg->CreateArrayRefExpression(new_index->name(),
      index_access2->clone());
      
      index_expr3 = ocg->CreateArrayRefExpression(new_index->name(),
      index_access3->clone());
      
      stmt_2 = ocg->CreateAssignment(0, index_expr3->clone(),
      index_expr2->clone());
      
      Statement s2 = stmt[stmt_num];
      
      s2.xform = copy(stmt[stmt_num].xform);
      s2.code = stmt_2;
      for (int i = outer_loop_levels[outer_loop_levels.size() - 1];
      i <= stmt[stmt_num].loop_level.size(); i++)
      assign_const(s2.xform, 2 * i - 1, 0);
      
      assign_const(s2.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 2] - 2,
      lex_[0] + 1);
      
      //stmt.push_back(s2);
      
      ///iii. notAllzero = 0;
      
      CG_outputRepr *zero_counter = ocg->CreateIdent("notAllZero");
      
      stmt_3 = ocg->CreateAssignment(0, zero_counter->clone(),
      ocg->CreateInt(0));
      
      Statement s3 = stmt[stmt_num];
      
      s3.xform = copy(stmt[stmt_num].xform);
      //s3.xform = copy(stmt[stmt_num].xform);
      s3.code = stmt_3;
      
      for (int i = outer_loop_levels[outer_loop_levels.size() - 1] + 1;
      i <= stmt[stmt_num].loop_level.size(); i++)
      assign_const(s3.xform, 2 * i - 1, 0);
      
      assign_const(s3.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 2] - 2,
      lex_[0] + 1);
      assign_const(s3.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2,
      lex_[2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2]
      + 1);
      
      //stmt.push_back(s3);
      
      ///iv.     if (new_P1[ii*r+i][jj*c+j] != 0) {
      ///           notallzero = 1;
      ///           break;IR_PointerSymbol *offset_index, *explicit_index, *new_array_prime, *marked;
      ///       }
      
      CG_outputRepr *cond, *brk, *assn, *list;
      CG_outputRepr *orig_expr;
      std::vector<CG_outputRepr *> subscripts;
      if (arr_sym != NULL) {
      for (int i = 0; i < arr_sym->n_dim(); i++)
      subscripts.push_back(arr_sym->index(i));
      
      orig_expr = ocg->CreateArrayRefExpression(arr_sym->symbol()->name(),
      subscripts[0]->clone());
      
      for (int i = 1; i < arr_sym->n_dim(); i++)
      orig_expr = ocg->CreateArrayRefExpression(orig_expr->clone(),
      subscripts[i]->clone());
      
      } else if (ptr_sym != NULL) {
      for (int i = 0; i < ptr_sym->symbol()->n_dim(); i++)
      subscripts.push_back(ptr_sym->index(i));
      
      orig_expr = ocg->CreateArrayRefExpression(ptr_sym->symbol()->name(),
      subscripts[0]->clone());
      
      for (int i = 1; i < ptr_sym->symbol()->n_dim(); i++)
      orig_expr = ocg->CreateArrayRefExpression(orig_expr->clone(),
      subscripts[i]->clone());
      
      }
      
      cond = ocg->CreateNEQ(orig_expr->clone(), ocg->CreateInt(0));
      assn = ocg->CreateAssignment(0, zero_counter->clone(),
      ocg->CreateInt(1));
      brk = ocg->CreateBreakStatement();
      
      list = ocg->StmtListAppend(assn->clone(), brk->clone());
      
      stmt_4 = ocg->CreateIf(0, cond->clone(), list->clone(), NULL);
      
      Statement s4 = stmt[stmt_num];
      
      s4.xform = copy(stmt[stmt_num].xform);
      s4.code = stmt_4;
      
      assign_const(s4.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 2] - 2,
      lex_[0] + 1);
      assign_const(s4.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2,
      lex_[2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2]
      + 1);
      assign_const(s4.xform, 2 * zero_entity_levels[0] - 2,
      lex_[2 * zero_entity_levels[0] - 2] + 1);
      
      ///v.    if (notAllzero == 1)
      ///           break;
      
      CG_outputRepr *cond2 = ocg->CreateEQ(zero_counter->clone(),
      ocg->CreateInt(1));
      stmt_5 = ocg->CreateIf(0, cond2->clone(), brk->clone(), NULL);
      
      Statement s5 = stmt[stmt_num];
      
      s5.xform = copy(stmt[stmt_num].xform);
      
      for (int i = outer_loop_levels[outer_loop_levels.size() - 1] + 3;
      i <= stmt[stmt_num].loop_level.size(); i++)
      assign_const(s5.xform, 2 * i - 1, 0);
      
      assign_const(s5.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 2] - 2,
      lex_[0] + 1);
      assign_const(s5.xform,
      2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2,
      lex_[2 * outer_loop_levels[outer_loop_levels.size() - 1] - 2]
      + 1);
      assign_const(s5.xform, 2 * zero_entity_levels[0] - 2,
      lex_[2 * zero_entity_levels[0] - 2] + 1);
      assign_const(s5.xform, 2 * zero_entity_levels[1] - 2,
      lex_[2 * zero_entity_levels[1] - 2] + 1);
      
      if (outer_loop_levels[outer_loop_levels.size() - 1] + 2
      <= stmt[stmt_num].loop_level.size())
      assign_const(s5.xform,
      2 * (outer_loop_levels[outer_loop_levels.size() - 1] + 2)
      - 1,
      zero_entity_loop_bounds[zero_entity_levels.size() - 1]);
      s5.code = stmt_5;
      
      //stmt.push_back(s5);
      
      ///vi.
      
      
      CG_outputRepr *loop_body, *data_copy, *col_copy, *if_check, *loop1,
      *loop2, *index_inc, *data_subscript = NULL, *data_subscript2,
      *data_copy2, *col_copy2;
      
      for (int i = 0; i < outer_loop_levels.size() - 2; i++) {
      CG_outputRepr *indice = ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(outer_loop_levels[i])->name());
      for (int j = i + 1; j <= stmt[stmt_num].loop_level.size(); j++)
      indice = ocg->CreateTimes(ocg->CreateInt(j), indice->clone());
      
      if (data_subscript == NULL)
      data_subscript = indice;
      else
      data_subscript = ocg->CreatePlus(data_subscript->clone(),
      indice->clone());
      }
      
      int zero_entity_size = 1;
      for (int i = 0; i < zero_entity_loop_bounds.size(); i++)
      zero_entity_size *= zero_entity_loop_bounds[i];
      
      //if (zero_entity_size > 1)
      // data_subscript = ocg->CreateTimes(ocg->CreateInt(zero_entity_size),
      //     index_expr3->clone());
      
      if (data_subscript != NULL && zero_entity_size > 1) {
      
      data_subscript = ocg->CreatePlus(data_subscript->clone(),
      ocg->CreateTimes(ocg->CreateInt(zero_entity_size),
      index_expr3->clone()));
      data_subscript2 =
      ocg->CreatePlus(data_subscript->clone(),
      ocg->CreateTimes(ocg->CreateInt(zero_entity_size),
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(
      outer_loop_levels[outer_loop_levels.size()
      - 1])->name())));
      
      } else if (data_subscript != NULL && zero_entity_size == 1) {
      data_subscript = ocg->CreatePlus(data_subscript->clone(),
      index_expr3->clone());
      data_subscript2 = ocg->CreatePlus(data_subscript->clone(),
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(
      outer_loop_levels[outer_loop_levels.size()
      - 1])->name()));
      } else if (data_subscript == NULL && zero_entity_size > 1) {
      data_subscript = ocg->CreateTimes(ocg->CreateInt(zero_entity_size),
      index_expr3->clone());
      data_subscript2 = ocg->CreateTimes(ocg->CreateInt(zero_entity_size),
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(
      outer_loop_levels[outer_loop_levels.size()
      - 1])->name()));
      } else if (data_subscript == NULL && zero_entity_size == 1) {
      data_subscript = index_expr3->clone();
      data_subscript2 =
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(
      outer_loop_levels[outer_loop_levels.size()
      - 1])->name());
      }
      if (zero_entity_levels.size() > 1)
      for (int i = 0; i < zero_entity_levels.size(); i++) {
      CG_outputRepr *index =
      ocg->CreateIdent(
      stmt[stmt_num].IS.set_var(zero_entity_levels[i])->name());
      for (int j = i + 1; j < zero_entity_levels.size(); j++)
      index = ocg->CreateTimes(
      ocg->CreateInt(zero_entity_loop_bounds[j]),
      index->clone());
      
      data_subscript = ocg->CreatePlus(data_subscript->clone(),
      index->clone());
      data_subscript2 = ocg->CreatePlus(data_subscript2->clone(),
      index->clone());
      }
      
      data_copy = ocg->CreateArrayRefExpression(new_data->name(),
      data_subscript->clone());
      data_copy2 = ocg->CreateArrayRefExpression(new_data->name(),
      data_subscript2->clone());
      col_copy = ocg->CreateArrayRefExpression(new_col->name(),ool sort_helper_2(std::pair<int, int> i, std::pair<int, int> j) {
      
      return (i.second < j.second);
  */
}



void Loop::ELLify(int stmt_num, std::vector<std::string> arrays_to_pad,
                  int pad_to, bool dense_pad, std::string dense_pad_pos_array) {
  
  apply_xform(stmt_num);
  
  //Sanity Check if Loop is a double loop
  
  if (stmt[stmt_num].loop_level.size() != 2)
    throw loop_error("ELLify only works on doubly nested loops for now !");
  
  //Check that outer loop is normalized
  
  Relation bound = get_loop_bound(copy(stmt[stmt_num].IS), 1, this->known);
  int dim = 2;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> old_loop = getStatements(lex, dim - 1);
  if (!lowerBoundIsZero(bound, 1))
    for (std::set<int>::iterator it = old_loop.begin();
         it != old_loop.end(); it++)
      normalize(*it, 1);
  //0.If inner loop does not start from zero
  //normalize
  
  bound = get_loop_bound(copy(stmt[stmt_num].IS), 2, this->known);
  dim = 4;
  if (!lowerBoundIsZero(bound, 2))
    for (std::set<int>::iterator it = old_loop.begin();
         it != old_loop.end(); it++)
      normalize(*it, 2);
  
  //1.do a scalar expand get the original statement and the scalar expanded statements
  
  std::vector<int> loop_levels;
  loop_levels.push_back(1);
  loop_levels.push_back(2);
  
  if (dense_pad)
    assert(arrays_to_pad.size() == 2);
  
  std::string data_array_name;
  std::string pos_array_name;
  int count;
  for (int i = 0; i < arrays_to_pad.size(); i++) {
    
    if (dense_pad) {
      count = ir->getAndIncrementPointerCounter();
      if (arrays_to_pad[i] == dense_pad_pos_array)
        pos_array_name = "_P_DATA" + omega::to_string(count);
      else
        data_array_name = "_P_DATA" + omega::to_string(count);
    }
    scalar_expand(stmt_num, loop_levels, arrays_to_pad[i], 0, 0, 1, pad_to);
    
  }
  std::set<int> new_loop = getStatements(lex, dim - 1);
  
  //2.distribute the original statement and the scalar expanded statements
  distribute(new_loop, 2);
  distribute(new_loop, 1);
  std::set<int> new_stmts;
  
  for (std::set<int>::iterator it = new_loop.begin(); it != new_loop.end();
       it++)
    if (old_loop.find(*it) == old_loop.end())
      new_stmts.insert(*it);
  
  fuse(new_stmts, 1);
  fuse(new_stmts, 2);
  
  fuse(old_loop, 1);
  fuse(old_loop, 2);
  
  //3.replace the original statement with executor loop extended to pad_to
  apply_xform();
  std::vector<IR_ArrayRef *> refs_;
  for (std::set<int>::iterator it = old_loop.begin(); it != old_loop.end();
       it++) {
    
    std::vector<IR_ArrayRef *> refs2;
    
    refs2 = ir->FindArrayRef(stmt[*it].code);
    
    for (int i = 0; i < refs2.size(); i++)
      refs_.push_back(refs2[i]);
    
    Relation R(stmt[*it].IS.n_set());
    F_And *f = R.add_and();
    
    GEQ_Handle g1 = f->add_GEQ();
    g1.update_coef(R.set_var(2), -1);
    g1.update_const(pad_to - 1);
    
    GEQ_Handle g2 = f->add_GEQ();
    g2.update_coef(R.set_var(2), 1);
    
    stmt[*it].IS = and_with_relation_and_replace_var(stmt[*it].IS,
                                                     stmt[*it].IS.set_var(2), R);
    
    if (ir->QueryExpOperation(stmt[*it].code) != IR_OP_PLUS_ASSIGNMENT)
      throw ir_error("Statement is not a += accumulation statement");
    
  }
  
  //4.add to inspector loop to extend to pad_to
  
  std::set<int> final_stmts = new_stmts;
  std::set<int> finally_created_stmts;
  std::vector<IR_PointerArrayRef *> refs;
  for (std::set<int>::iterator it = new_stmts.begin(); it != new_stmts.end();
       it++) {
    
    std::vector<IR_PointerArrayRef *> refs2;
    
    refs2 = ir->FindPointerArrayRef(stmt[*it].code);
    
    for (int i = 0; i < refs2.size(); i++)
      refs.push_back(refs2[i]);
    Relation R = extract_upper_bound(stmt[*it].IS, stmt[*it].IS.set_var(2));
    
    R = Complement(R);
    
    F_And *f_root = R.and_with_and();
    
    GEQ_Handle g1 = f_root->add_GEQ();
    g1.update_coef(R.set_var(2), -1);
    g1.update_const(pad_to - 1);
    
    R.simplify();
    Relation new_IS = and_with_relation_and_replace_var(copy(stmt[*it].IS),
                                                        stmt[*it].IS.set_var(2), R);
    
    Statement s = stmt[*it];
    s.IS = new_IS;
    s.code = stmt[*it].code->clone();
    
    CG_outputRepr *lhs = ir->GetLHSExpression(s.code);
    s.code = ir->builder()->CreateAssignment(0, lhs->clone(),
                                             ir->builder()->CreateInt(0));
    
    debug_fprintf(stderr, "loop.cc L7095 adding stmt %d\n", stmt.size()); 
    stmt.push_back(s);

    uninterpreted_symbols.push_back(uninterpreted_symbols[*it]);
    uninterpreted_symbols_stringrepr.push_back(
                                               uninterpreted_symbols_stringrepr[*it]);
    dep.insert();
    final_stmts.insert(stmt.size() - 1);
    finally_created_stmts.insert(stmt.size() - 1);
    
  }
  if (dense_pad) {
    
    IR_PointerSymbol *data = NULL;
    IR_PointerSymbol *col = NULL;
    for (int i = 0; i < refs.size(); i++) {
      
      if (refs[i]->name() == data_array_name)
        data = refs[i]->symbol();
      else if (refs[i]->name() == pos_array_name)
        col = refs[i]->symbol();
      
      if (data != NULL && col != NULL)
        break;
    }
    
    //1. define a permutation based on values of col.. sigma[j] = col[j];
    //2. Generate a permutation for j[0,n] in col
    
    /// Design decisions for paper
    /*
     *  1. Use the col array to route the data array "A"
     *  2. Do the simplification on _P2 since it is a permutation remove the inner indirection x[col[j]]->x[j]
     *
     *
     *
     *
     *
     *
     */
    
    //1. Set up statement iteration spaces
    Statement s0 = stmt[*(new_stmts.begin())];
    Statement s1 = stmt[*(new_stmts.begin())];
    std::vector<int> lex0 = getLexicalOrder(*(new_stmts.begin()));
    IR_PointerSymbol *sigma;
    IR_PointerSymbol *new_P1;
    
    // 2. Create Additional array variables
    //CG_roseBuilder *ocg = dynamic_cast<CG_roseBuilder*>(ir->builder());
    CG_outputBuilder *ocg = ir->builder();
    
    std::vector<CG_outputRepr *> dims1;
    std::vector<CG_outputRepr *> dims2;
    
    for (int i = 0; i < data->n_dim(); i++) {
      dims1.push_back(ocg->CreateNullStatement());
      dims2.push_back(ocg->CreateNullStatement());
      
    }
    
    sigma = ir->CreatePointerSymbol(col, dims1);
    new_P1 = ir->CreatePointerSymbol(data, dims2);
    
    // 3. Create stmt.code for each statement
    
    CG_outputRepr *linearized_expr = ocg->CreatePlus(
                                                     ocg->CreateTimes(ocg->CreateIdent(s1.IS.set_var(1)->name()),
                                                                      ocg->CreateInt(pad_to)),
                                                     ocg->CreateIdent(s1.IS.set_var(2)->name()));
    CG_outputRepr *stmt_1, *stmt_2, *stmt_3, *stmt_4, *stmt_5;
    
    CG_outputRepr *array_exp1 = ocg->CreateArrayRefExpression(sigma->name(),
                                                              linearized_expr->clone());
    
    s0.code = ocg->CreateAssignment(0, array_exp1->clone(),
                                    ocg->CreateInt(0));
    
    assign_const(s1.xform, 2, lex0[2] + 1);
    
    CG_outputRepr *array_exp2 = ocg->CreateArrayRefExpression(col->name(),
                                                              linearized_expr->clone());
    CG_outputRepr *linearized_expr2 = ocg->CreatePlus(
                                                      ocg->CreateTimes(ocg->CreateIdent(s1.IS.set_var(1)->name()),
                                                                       ocg->CreateInt(pad_to)), array_exp2->clone());
    
    CG_outputRepr *array_exp3 = ocg->CreateArrayRefExpression(sigma->name(),
                                                              linearized_expr2->clone());
    
    CG_outputRepr *array_exp4 = ocg->CreateArrayRefExpression(
                                                              new_P1->name(), linearized_expr2->clone());
    
    CG_outputRepr *array_exp5 = ocg->CreateArrayRefExpression(data->name(),
                                                              linearized_expr->clone());
    
    stmt_1 = ocg->CreateAssignment(0, array_exp3->clone(),
                                   ocg->CreateInt(1));
    
    stmt_2 = ocg->CreateAssignment(0, array_exp4->clone(),
                                   array_exp5->clone());
    CG_outputRepr * if_cond = ocg->CreateNEQ(array_exp2->clone(),
                                             ocg->CreateInt(0));
    
    CG_outputRepr *linearized_expr3 = ocg->CreatePlus(
                                                      ocg->CreateTimes(ocg->CreateIdent(s1.IS.set_var(1)->name()),
                                                                       ocg->CreateInt(pad_to)), ocg->CreateIdent("t6"));
    
    CG_outputRepr *array_exp6 = ocg->CreateArrayRefExpression(sigma->name(),
                                                              linearized_expr3->clone());
    
    stmt_4 = ocg->CreateAssignment(0, array_exp6->clone(),
                                   ocg->CreateInt(1));
    
    CG_outputRepr *array_exp7 = ocg->CreateArrayRefExpression(
                                                              new_P1->name(), linearized_expr3->clone());
    
    CG_outputRepr *array_exp7a = ocg->CreateArrayRefExpression(
                                                               new_P1->name(), linearized_expr->clone());
    stmt_5 = ocg->CreateAssignment(0, array_exp7->clone(),
                                   array_exp5->clone());
    CG_outputRepr *stmt_6 = ocg->CreateLoop(0,
                                            ocg->CreateInductive(ocg->CreateIdent("t6"), ocg->CreateInt(0),
                                                                 ocg->CreateInt(pad_to - 1),
                                                                 NULL),
                                            ocg->CreateIf(0,
                                                          ocg->CreateEQ(array_exp6->clone(), ocg->CreateInt(1)),
                                                          ocg->StmtListAppend(ocg->StmtListAppend(stmt_4, stmt_5),
                                                                              ocg->CreateBreakStatement()), NULL));
    
    stmt_3 = ocg->CreateIf(0, if_cond, ocg->StmtListAppend(stmt_1, stmt_2),
                           stmt_6);
    s1.code = stmt_3;
    
    dep.insert();
    
    std::vector<int> lex2 = getLexicalOrder(*(new_stmts.begin()));
    
    assign_const(s1.xform, 0, lex2[0] + 1);
    lex2[0] = lex2[0] + 1;
    shiftLexicalOrder(lex2, 0, 1);
    
    debug_fprintf(stderr, "loop.cc L7236 adding stmt %d\n", stmt.size()); 
    stmt.push_back(s1);

    uninterpreted_symbols.push_back(
                                    uninterpreted_symbols[*(new_stmts.begin())]);
    uninterpreted_symbols_stringrepr.push_back(
                                               uninterpreted_symbols_stringrepr[*(new_stmts.begin())]);
    lex2 = getLexicalOrder(stmt.size() - 1);
    
    std::vector<IR_PointerArrayRef *> refs2;
    
    refs2 = ir->FindPointerArrayRef(stmt[*(old_loop.begin())].code);
    for (int i = 0; i < refs2.size(); i++)
      if (refs2[i]->name() == data_array_name)
        ir->ReplaceExpression(refs2[i], array_exp7a);
    
    std::vector<IR_ArrayRef *> refs_;
    
    refs_ = ir->FindArrayRef(stmt[*(old_loop.begin())].code);
    
    for (int i = 0; i < refs_.size(); i++) {
      if (ir->QueryExpOperation(refs_[i]->index(0))
          == IR_OP_ARRAY_VARIABLE) {
        std::vector<IR_ArrayRef *> inner_refs = ir->FindArrayRef(
                                                                 refs_[i]->index(0));
        
        for (int k = 0; k < inner_refs.size(); k++)
          if (inner_refs[k]->name() == pos_array_name) {
            CG_outputRepr *array_exp =
              ocg->CreateArrayRefExpression(refs_[i]->name(),
                                            ocg->CreateIdent(
                                                             s1.IS.set_var(2)->name()));
            ir->ReplaceExpression(refs_[i], array_exp);
            
          }
        
      }
    }
    /*
     *
     * Inspector Code
     for(i=0; i < n; i++)
     for(j=0; j < n; j++){
     if(_P2[i][j] != 0 )
     sigma[i][j] = _P2[i][j];
     
     else
     sigma[i][j] =  j
     new_P1[i][sigma[i][j]] = _P1[i][j];
     
     }
     
     }
    */
    
    // 4.set up lexical order and dependences
    //2.by right you have to compose the array access relation with the iteration and data relations, but for now just plug in the simplified relation
    // generate the code here instead of at code gen: intorduce the statement here.: new_P1[i][j]*x[j]
  }
  //  distribute(final_stmts, 2);
  //  fuse(finally_created_stmts, 2);
}

// TODO 
#define _DEBUG_ true

void Loop::make_dense(int stmt_num, int loop_level,
                      std::string new_loop_index) {
  
  debug_fprintf(stderr, "\nLoop::make_dense()\n"); 
  apply_xform(stmt_num);
  std::vector<int> lex = getLexicalOrder(stmt_num);
  //1. Identify Loop Index
  std::string loop_index = stmt[stmt_num].IS.set_var(loop_level)->name();
  if (_DEBUG_)
    std::cout << loop_index << std::endl;
  
  //2. Identify indirect access via loop index, throw error if multiple such accesses, throw error if index expression not basic
  
  std::vector<IR_ArrayRef *> arr_refs = ir->FindArrayRef(stmt[stmt_num].code);
  
  int count = 0;
  std::string inner_arr_name;
  for (int i = 0; i < arr_refs.size(); i++) {
    if (arr_refs[i]->n_dim() == 1)
      if (ir->QueryExpOperation(arr_refs[i]->index(0))
          == IR_OP_ARRAY_VARIABLE) {
        std::vector<CG_outputRepr *> inner_ref = ir->QueryExpOperand(
                                                                     arr_refs[i]->index(0));
        std::vector<IR_ArrayRef *> inner_arr_refs = ir->FindArrayRef(
                                                                     inner_ref[0]);
        if (inner_arr_refs[0]->n_dim() == 1)
          if (ir->QueryExpOperation(inner_arr_refs[0]->index(0))
              == IR_OP_VARIABLE) {
            std::vector<CG_outputRepr *> var_ref =
              ir->QueryExpOperand(
                                  inner_arr_refs[0]->index(0));
            if (dynamic_cast<IR_ScalarRef *>(ir->Repr2Ref(
                                                          var_ref[0]))->name() == loop_index) {
              if (count > 0
                  && inner_arr_refs[0]->name()
                  != inner_arr_name)
                throw loop_error(
                                 "Multiple arrays with indirect accesses not supported currently");
              count++;
              inner_arr_name = inner_arr_refs[0]->name();
              
              if (_DEBUG_)
                std::cout << inner_arr_name << std::endl;
            }
          }
      }
  }
  
  if (count == 0)
    throw loop_error("No indirect accesses found");
  
  //3. Construct new loop index  with lb and ub.. Setup Iteration Space
  
  Relation new_bound(1);
  
  //std::cout << (*it)->name() << std::endl;
  F_And *f_root = new_bound.add_and();
  new_bound.name_set_var(1, new_loop_index);
  
  GEQ_Handle lower_bound = f_root->add_GEQ();
  lower_bound.update_coef(new_bound.set_var(1), 1);
  Free_Var_Decl *lb = new Free_Var_Decl("lb");
  Variable_ID e = lower_bound.get_local(lb);
  freevar.push_back(lb);
  lower_bound.update_coef(e, -1);
  
  GEQ_Handle upper_bound = f_root->add_GEQ();
  upper_bound.update_coef(new_bound.set_var(1), -1);
  Free_Var_Decl *ub = new Free_Var_Decl("ub");
  Variable_ID e1 = upper_bound.get_local(ub);
  freevar.push_back(ub);
  upper_bound.update_coef(e1, 1);
  
  new_bound.setup_names();
  new_bound.simplify();
  
  Relation new_IS = replicate_IS_and_add_at_pos(stmt[stmt_num].IS, loop_level,
                                                new_bound);
  Relation new_xform(stmt[stmt_num].IS.n_set() + 1,
                     stmt[stmt_num].xform.n_out() + 2);
  
  F_And *f_root_ = new_xform.add_and();
  for (int j = 1; j <= new_IS.n_set(); j++) {
    omega::EQ_Handle h = f_root_->add_EQ();
    h.update_coef(new_xform.output_var(2 * j), 1);
    h.update_coef(new_xform.input_var(j), -1);
  }
  for (int j = 1; j <= 2 * new_IS.n_set() + 1; j += 2) {
    omega::EQ_Handle h = f_root_->add_EQ();
    h.update_coef(new_xform.output_var(j), 1);
    h.update_const(-lex[j - 1]);
  }
  
  /*  for (int j = 1; j <= new_IS.n_set(); j++) {
      new_xform.name_input_var(j, stmt[stmt_num].xform.input_var(j)->name());
      new_xform.name_output_var(j, stmt[stmt_num].xform.output_var(2*j)->name());
      new_xform.name_output_var(2*j+1, stmt[stmt_num].xform.output_var(2*j+1)->name());
      }
      
      new_xform.name_input_var(stmt[stmt_num].IS.n_set() + 1, new_IS.set_var(stmt[stmt_num].IS.n_set() + 1)->name());
      new_xform.name_output_var(2*stmt[stmt_num].IS.n_set()+2, "_t"+omega::to_string(tmp_loop_var_name_counter+1));
      new_xform.name_output_var(2*stmt[stmt_num].IS.n_set()+3, "_t"+omega::to_string(tmp_loop_var_name_counter+2));
      stmt[stmt_num].xform = new_xform;ir->Repr2Ref(indir_refs[i]->index(0))->name()
  */
  stmt[stmt_num].xform = new_xform;
  stmt[stmt_num].IS = new_IS;
  this->known = Extend_Set(copy(this->known),
                           stmt[stmt_num].IS.n_set() - this->known.n_set());
  
  if (_DEBUG_) {
    stmt[stmt_num].xform.print();
    stmt[stmt_num].IS.print();
  }
  
  //4. Insert Guard Condition and replace indirect access with new loop index. Setup Code AST.
  CG_outputBuilder *ocg = ir->builder();


  debug_fprintf(stderr, "loop.cc  new_loop_index %s\n", new_loop_index.c_str()); 
  debug_fprintf(stderr, "loop.cc  %s == %s[%s] ??\n", 
          new_loop_index.c_str(),
          inner_arr_name.c_str(), 
          loop_index.c_str()); 

  // the following used a string insteaad of a variable. Seems iffy
  //CG_outputRepr *guard = ocg->CreateEQ(ocg->CreateIdent(new_loop_index),
  //                                     ocg->CreateArrayRefExpression(inner_arr_name,
  //                                                                   ocg->CreateIdent(loop_index)));
  
  CG_outputRepr *array    = ocg->CreateIdent(inner_arr_name); 
  CG_outputRepr *arindex  = ocg->CreateIdent(loop_index);
  CG_outputRepr *arrayref = ocg->CreateArrayRefExpression(array, arindex);

  CG_outputRepr *loopind  = ocg->CreateIdent(loop_index);
  CG_outputRepr *guard = ocg->CreateEQ(loopind, arrayref); // k == col[something]




  
  CG_outputRepr *body = stmt[stmt_num].code->clone();
  //  delete stmt[stmt_num].code;
  
  std::vector<IR_ArrayRef *> indir_refs = ir->FindArrayRef(body);
  for (int i = 0; i < indir_refs.size(); i++)
    if (indir_refs[i]->name() == inner_arr_name
        && ir->Repr2Ref(indir_refs[i]->index(0))->name() == loop_index)
      ir->ReplaceExpression(indir_refs[i],
                            ocg->CreateIdent(new_loop_index));
  
  CG_outputRepr * new_code = ocg->CreateIf(0, guard, body, NULL);
  
  stmt[stmt_num].code = new_code;
  
  LoopLevel dense_loop;
  
  dense_loop.type = LoopLevelOriginal;
  dense_loop.payload = loop_level - 1;
  dense_loop.parallel_level = 0;
  dense_loop.segreducible =
    stmt[stmt_num].loop_level[loop_level - 1].segreducible;
  if (dense_loop.segreducible)
    dense_loop.segment_descriptor =
      stmt[stmt_num].loop_level[loop_level - 1].segment_descriptor;
  
  debug_fprintf(stderr, "\n\n*** loop_level.insert() ***\n\n\n"); // this is the only insert
  stmt[stmt_num].loop_level.insert(
                                   stmt[stmt_num].loop_level.begin() + (loop_level - 1), dense_loop);
  
  stmt[stmt_num].loop_level[loop_level].payload = loop_level;
  
  num_dep_dim += 1;
  //hack for permute:Anand should be removed
  int size_ = dep.vertex.size();
  dep = DependenceGraph(num_dep_dim);
  for (int i = 0; i < size_; i++)
    dep.insert();
}



void Loop::split_with_alignment(int stmt_num, int level, int alignment,
                                int direction) {
  
  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument(
                                "invalid statement number " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("2invalid loop level " + to_string(level));
  int dim = 2 * level - 1;
  std::set<int> subloop = getSubLoopNest(stmt_num, level);
  std::vector<Relation> Rs;
  std::map<int, int> what_stmt_num;
  for (std::set<int>::iterator i = subloop.begin(); i != subloop.end(); i++) {
    Relation r = getNewIS(*i);
    Relation f(r.n_set(), level);
    F_And *f_root = f.add_and();
    for (int j = 1; j <= level; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(f.input_var(2 * j), 1);
      h.update_coef(f.output_var(j), -1);
    }
    //Anand composition will fail due to unintepreted function symbols introduced by flattening
    //r = Composition(f, r);
    r = omega::Range(Restrict_Domain(f, r));
    r.simplify();
    Rs.push_back(r);
  }
  Relation hull = SimpleHull(Rs);
  
  GEQ_Handle bound_eq;
  bool found_bound = false;
  
  for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++)
    if (!(*e).has_wildcards() && (*e).get_coef(hull.set_var(level)) < 0) {
      bound_eq = *e;
      found_bound = true;
      break;
    }
  if (!found_bound)
    for (GEQ_Iterator e(hull.single_conjunct()->GEQs()); e; e++)
      if ((*e).has_wildcards()
          && (*e).get_coef(hull.set_var(level)) < 0) {
        bool is_bound = true;
        for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
          std::pair<bool, GEQ_Handle> result = find_floor_definition(
                                                                     hull, cvi.curr_var());
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
  if (!found_bound)
    throw loop_error(
                     "can't find upper bound for split_with_alignment at level  "
                     + to_string(level));
  
  Relation r(level);
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  //  GEQ_Handle h = f_root->add_GEQ();
  
  Variable_ID ub = f_exists->declare();
  
  coef_t coef = bound_eq.get_coef(hull.set_var(level));
  if (coef == -1) { // e.g. if i <= m+5, then UB = m+5
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(ub, -1);
    for (Constr_Vars_Iter ci(bound_eq); ci; ci++) {
      switch ((*ci).var->kind()) {
      case Input_Var: {
        int pos = (*ci).var->get_position();
        if (pos != level)
          h.update_coef(r.set_var(ci.curr_var()->get_position()),
                        ci.curr_coef());
        break;
      }
      case Wildcard_Var: {
        Variable_ID v = replicate_floor_definition(hull, ci.curr_var(),
                                                   r, f_exists, f_root, exists_mapping);
        h.update_coef(v, ci.curr_coef());
        break;
      }
      case Global_Var: {
        Global_Var_ID g = ci.curr_var()->get_global_var();
        Variable_ID v;
        if (g->arity() == 0)
          v = r.get_local(g);
        else
          v = r.get_local(g, ci.curr_var()->function_of());
        h.update_coef(v, ci.curr_coef());
        break;
      }
      default:
        throw loop_error(
                         "cannot handle complex upper bound in split_with_alignment!");
      }
    }
    h.update_const(bound_eq.get_const());
  } else { // e.g. if 2i <= m+5, then m+5-2 < 2*UB <= m+5
    
    throw loop_error(
                     "cannot handle complex upper bound in split_with_alignment!");
  }
  
  
  
  
  Variable_ID aligned_ub = f_exists->declare();
  
  Variable_ID e = f_exists->declare();
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(aligned_ub, 1);
  h.update_coef(e, -alignment);
  
  GEQ_Handle h1 = f_root->add_GEQ();
  GEQ_Handle h2 = f_root->add_GEQ();
  h1.update_coef(e, alignment);
  h2.update_coef(e, -alignment);
  h1.update_coef(ub, -1);
  h2.update_coef(ub, 1);
  h2.update_const(1);
  h1.update_const(alignment - 2);
  
  GEQ_Handle h3 = f_root->add_GEQ();
  
  h3.update_coef(r.set_var(level), -1);
  h3.update_coef(aligned_ub, 1);
  h3.update_const(-1);
  
  r.simplify();
  //split(stmt_num, level, r);
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim - 1);
  int cur_lex = lex[dim - 1];
  apply_xform(stmt_num);
  
  int n = stmt[stmt_num].IS.n_set();
  Relation part1 = Intersection(copy(stmt[stmt_num].IS),
                                Extend_Set(copy(r), n - level));
  //Relation part2 = Intersection(copy(stmt[stmt_num].IS),
  //      Extend_Set(Complement(copy(r)),
  //          n - level));
  
  Relation part2;
  {
    
    Relation r(level);
    F_Exists *f_exists = r.add_and()->add_exists();
    F_And *f_root = f_exists->add_and();
    std::map<Variable_ID, Variable_ID> exists_mapping;
    //  GEQ_Handle h = f_root->add_GEQ();
    
    Variable_ID ub = f_exists->declare();
    
    coef_t coef = bound_eq.get_coef(hull.set_var(level));
    if (coef == -1) { // e.g. if i <= m+5, then UB = m+5
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(ub, -1);
      for (Constr_Vars_Iter ci(bound_eq); ci; ci++) {
        switch ((*ci).var->kind()) {
        case Input_Var: {
          int pos = (*ci).var->get_position();
          if (pos != level)
            h.update_coef(r.set_var(ci.curr_var()->get_position()),
                          ci.curr_coef());
          break;
        }
        case Wildcard_Var: {
          Variable_ID v = replicate_floor_definition(hull, ci.curr_var(),
                                                     r, f_exists, f_root, exists_mapping);
          h.update_coef(v, ci.curr_coef());
          break;
        }
        case Global_Var: {
          Global_Var_ID g = ci.curr_var()->get_global_var();
          Variable_ID v;
          if (g->arity() == 0)
            v = r.get_local(g);
          else
            v = r.get_local(g, ci.curr_var()->function_of());
          h.update_coef(v, ci.curr_coef());
          break;
        }
        default:
          throw loop_error(
                           "cannot handle complex upper bound in split_with_alignment!");
        }
      }
      h.update_const(bound_eq.get_const());
    } else { // e.g. if 2i <= m+5, then m+5-2 < 2*UB <= m+5
      
      throw loop_error(
                       "cannot handle complex upper bound in split_with_alignment!");
    }
    
    
    
    
    //Working
    
    /*    Variable_ID aligned_lb = f_exists->declare();
          
          Variable_ID e = f_exists->declare();
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(aligned_lb, 1);
          h.update_coef(e, -alignment);
          
          GEQ_Handle h1 = f_root->add_GEQ();
          GEQ_Handle h2 = f_root->add_GEQ();
          h1.update_coef(e, alignment);
          h2.update_coef(e, -alignment);
          h1.update_coef(ub, -1);
          h2.update_coef(ub, 1);
          h2.update_const(1);
          h1.update_const(alignment - 2);
          
          EQ_Handle h3 = f_root->add_EQ();
          
          h3.update_coef(r.set_var(level), -1);
          h3.update_coef(aligned_lb, 1);
          
          
          r.simplify();
          
          part2 = Intersection(copy(stmt[stmt_num].IS),
          Extend_Set(copy(r), n - level));
    */
    
    // e.g. to align at 4, aligned_lb = 4*alpha && LB-4 < 4*alpha <= LB
    
    /*Variable_ID aligned_lb = f_exists->declare();
      Variable_ID e = f_exists->declare();
      
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(aligned_lb, 1);
      h.update_coef(e, -alignment);
      
      GEQ_Handle h1 = f_root->add_GEQ();
      
      h1.update_coef(e, alignment);
      
      h1.update_coef(ub, -1);
      
      h1.update_const(alignment - 1);
      
      GEQ_Handle h2 = f_root->add_GEQ();
      
      h2.update_coef(e, -alignment);
      
      h2.update_coef(ub, 1);
      
      h2.update_const(-1);
      
      GEQ_Handle h3 = f_root->add_GEQ();
      
      h3.update_coef(ub, 1);
      h3.update_coef(r.set_var(level), -1);
      h3.update_coef(aligned_lb, -1);
      h3.update_const(-1);
      
      
      GEQ_Handle h4 = f_root->add_GEQ();
      
      
      h4.update_coef(r.set_var(level), 1);
      h4.update_coef(aligned_lb, -1);
      
      r.simplify();
    */
    
    
    Variable_ID aligned_lb = f_exists->declare();
    
    Variable_ID e = f_exists->declare();
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(aligned_lb, 1);
    h.update_coef(e, -alignment);
    
    GEQ_Handle h1 = f_root->add_GEQ();
    GEQ_Handle h2 = f_root->add_GEQ();
    h1.update_coef(e, alignment);
    h2.update_coef(e, -alignment);
    h1.update_coef(ub, -1);
    h2.update_coef(ub, 1);
    h2.update_const(1);
    h1.update_const(alignment - 2);
    
    GEQ_Handle h3 = f_root->add_GEQ();
    
    h3.update_coef(r.set_var(level), 1);
    h3.update_coef(aligned_lb, -1);
    
    
    r.simplify();
    
    part2 = Intersection(copy(stmt[stmt_num].IS),
                         Extend_Set(copy(r), n - level));
  }
  
  
  
  part1.simplify(2,4);
  part2.simplify(2,4);
  stmt[stmt_num].IS = part1;
  
  
  //if (Intersection(copy(part2),
  //    Extend_Set(copy(this->known), n - this->known.n_set())).is_upper_bound_satisfiable()) {
  Statement new_stmt;
  new_stmt.code = stmt[stmt_num].code->clone();
  new_stmt.IS = part2;
  new_stmt.xform = copy(stmt[stmt_num].xform);
  new_stmt.ir_stmt_node = NULL;
  new_stmt.loop_level = stmt[stmt_num].loop_level;
  new_stmt.has_inspector = stmt[stmt_num].has_inspector;
  new_stmt.reduction = stmt[stmt_num].reduction;
  new_stmt.reductionOp = stmt[stmt_num].reductionOp;
  stmt_nesting_level_.push_back(stmt_nesting_level_[stmt_num]);
  
  assign_const(new_stmt.xform, dim - 1, cur_lex + 1);
  
  
  debug_fprintf(stderr, "loop.cc L7814 adding stmt %d\n", stmt.size()); 
  stmt.push_back(new_stmt);

  uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
  uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
  dep.insert();
  what_stmt_num[stmt_num] = stmt.size() - 1;
  
  
  
  //}
  
  // update dependence graph
  int dep_dim = get_dep_dim_of(stmt_num, level);
  for (int i = 0; i < stmt.size() -1 ; i++) {
    std::vector<std::pair<int, std::vector<DependenceVector> > > D;
    
    for (DependenceGraph::EdgeList::iterator j =
           dep.vertex[i].second.begin();
         j != dep.vertex[i].second.end(); j++) {
      if (same_loop.find(i) != same_loop.end()) {
        if (same_loop.find(j->first) != same_loop.end()) {
          if (what_stmt_num.find(i) != what_stmt_num.end()
              && what_stmt_num.find(j->first)
              != what_stmt_num.end())
            dep.connect(what_stmt_num[i],
                        what_stmt_num[j->first], j->second);
          if (what_stmt_num.find(j->first)
              != what_stmt_num.end()) {
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
              D.push_back(
                          std::make_pair(what_stmt_num[j->first],
                                         dvs));
          } /*else if (!place_after
              && what_stmt_num.find(i)
              != what_stmt_num.end()) {
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
              
              }*/
        } else {
          if (what_stmt_num.find(i) != what_stmt_num.end())
            dep.connect(what_stmt_num[i], j->first, j->second);
        }
      } else if (same_loop.find(j->first) != same_loop.end()) {
        if (what_stmt_num.find(j->first) != what_stmt_num.end())
          D.push_back(
                      std::make_pair(what_stmt_num[j->first],
                                     j->second));
      }
    }
    
    for (int j = 0; j < D.size(); j++)
      dep.connect(i, D[j].first, D[j].second);
  }
  
  
}



extern void stencil( chillAST_node *topstatement ); 


bool Loop::find_stencil_shape( int statement ) { 
  //debug_fprintf(stderr, "Loop::find_stencil_shape( %d )\n", statement);
  omega::CG_chillRepr *CR = (omega::CG_chillRepr *)stmt[ statement].code;
  
  // cheat for now
  //debug_fprintf(stderr, "loop has %d stmt\n", stmt.size()); 
  //CR->chillnodes[0]->print(); printf("\n\n\n"); fflush(stdout);
  
  // try to create a list of statements
  std::vector<chillAST_node*> AST_statements;
  //debug_fprintf(stderr, "chill gathering statements\n");
  
  for (int i=0; i<CR->chillnodes.size(); i++) { 
    CR->chillnodes[i]->gatherStatements( AST_statements );
  }
  
  
  //debug_fprintf(stderr, "%d chill gathered statements\n", statements.size());
  //for (int i=0; i<statements.size(); i++) { 
  //debug_fprintf(stderr, "\nstatement %d\n", i);
  //statements[i]->print(); printf("\n"); fflush(stdout); 
  //} 
  
  
  // problem here is there are multiple AST statements 
  stmt[statement].statementStencil = new stencilInfo( AST_statements[0] ); 
  return true; 
  
}

