/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Useful tools involving Omega manipulation.

 Notes:

 History:
   01/2006 Created by Chun Chen.
   03/2009 Upgrade Omega's interaction with compiler to IR_Code, by Chun Chen.
*****************************************************************************/

#include <codegen.h>
// #include <code_gen/output_repr.h>
#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"

using namespace omega;

namespace {
  struct DependenceLevel {
    Relation r;
    int level;
    int dir; // direction upto current level:
    // -1:negative, 0: undetermined, 1: postive
    std::vector<coef_t> lbounds;
    std::vector<coef_t> ubounds;
    DependenceLevel(const Relation &_r, int _dims):
      r(_r), level(0), dir(0), lbounds(_dims), ubounds(_dims) {}
  };
}




std::string tmp_e() {
  static int counter = 1;
  return std::string("e")+to_string(counter++);
}



//-----------------------------------------------------------------------------
// Convert expression tree to omega relation.  "destroy" means shallow
// deallocation of "repr", not freeing the actual code inside.
// -----------------------------------------------------------------------------
void exp2formula(IR_Code *ir, Relation &r, F_And *f_root, std::vector<Free_Var_Decl*> &freevars,
                 CG_outputRepr *repr, Variable_ID lhs, char side, IR_CONDITION_TYPE rel, bool destroy) {
  
// void exp2formula(IR_Code *ir, Relation &r, F_And *f_root, std::vector<Free_Var_Decl*> &freevars,
//                   CG_outputRepr *repr, Variable_ID lhs, char side, char rel, bool destroy) {
  
  switch (ir->QueryExpOperation(repr)) {
  case IR_OP_CONSTANT:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[0]));
    if (!ref->is_integer())
      throw ir_exp_error("non-integer constant coefficient");
    
    coef_t c = ref->integer();
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_root->add_GEQ();
      h.update_coef(lhs, 1);
      if (rel == IR_COND_GE)
        h.update_const(-c);
      else
        h.update_const(-c-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_root->add_GEQ();
      h.update_coef(lhs, -1);
      if (rel == IR_COND_LE)
        h.update_const(c);
      else
        h.update_const(c-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(lhs, 1);
      h.update_const(-c);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    delete v[0];
    delete ref;
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_VARIABLE:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    IR_ScalarRef *ref = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v[0]));
    
    std::string s = ref->name();
    Variable_ID e = find_index(r, s, side);
    
    if (e == NULL) { // must be free variable
      Free_Var_Decl *t = NULL;
      for (unsigned i = 0; i < freevars.size(); i++) {
        std::string ss = freevars[i]->base_name();
        if (s == ss) {
          t = freevars[i];
          break;
        }
      }
      
      if (t == NULL) {
        t = new Free_Var_Decl(s);
        freevars.insert(freevars.end(), t);
      }
      
      e = r.get_local(t);
    }
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_root->add_GEQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, -1);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_root->add_GEQ();
      h.update_coef(lhs, -1);
      h.update_coef(e, 1);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, -1);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    //  delete v[0];
    delete ref;
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_ASSIGNMENT:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    exp2formula(ir, r, f_root, freevars, v[0], lhs, side, rel, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_PLUS:
  {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, 1);
      h.update_coef(e1, -1);
      h.update_coef(e2, -1);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, -1);
      h.update_coef(e1, 1);
      h.update_coef(e2, 1);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e1, -1);
      h.update_coef(e2, -1);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    exp2formula(ir, r, f_and, freevars, v[0], e1, side, IR_COND_EQ, true);
    exp2formula(ir, r, f_and, freevars, v[1], e2, side, IR_COND_EQ, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_MINUS:
  {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, 1);
      h.update_coef(e1, -1);
      h.update_coef(e2, 1);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, -1);
      h.update_coef(e1, 1);
      h.update_coef(e2, -1);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e1, -1);
      h.update_coef(e2, 1);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    exp2formula(ir, r, f_and, freevars, v[0], e1, side, IR_COND_EQ, true);
    exp2formula(ir, r, f_and, freevars, v[1], e2, side, IR_COND_EQ, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_MULTIPLY:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    coef_t coef;
    CG_outputRepr *term;
    if (ir->QueryExpOperation(v[0]) == IR_OP_CONSTANT) {
      IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[0]));
      coef = ref->integer();
      delete v[0];
      delete ref;
      term = v[1];
    }
    else if (ir->QueryExpOperation(v[1]) == IR_OP_CONSTANT) {
      IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[1]));
      coef = ref->integer();
      delete v[1];
      delete ref;
      term = v[0];
    }
    else
      throw ir_exp_error("not presburger expression");
    
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, -coef);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, -1);
      h.update_coef(e, coef);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, -coef);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    exp2formula(ir, r, f_and, freevars, term, e, side, IR_COND_EQ, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_DIVIDE:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    assert(ir->QueryExpOperation(v[1]) == IR_OP_CONSTANT);
    IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[1]));
    coef_t coef = ref->integer();
    delete v[1];
    delete ref;
    
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, coef);
      h.update_coef(e, -1);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, -coef);
      h.update_coef(e, 1);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(lhs, coef);
      h.update_coef(e, -1);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    exp2formula(ir, r, f_and, freevars, v[0], e, side, IR_COND_EQ, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_POSITIVE:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    exp2formula(ir, r, f_root, freevars, v[0], lhs, side, rel, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_NEGATIVE:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, 1);
      if (rel == IR_COND_GT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      GEQ_Handle h = f_and->add_GEQ();
      h.update_coef(lhs, -1);
      h.update_coef(e, -1);
      if (rel == IR_COND_LT)
        h.update_const(-1);
    }
    else if (rel == IR_COND_EQ) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, 1);
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    exp2formula(ir, r, f_and, freevars, v[0], e, side, IR_COND_EQ, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_OP_MIN:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    F_Exists *f_exists = f_root->add_exists();
    
    if (rel == IR_COND_GE || rel == IR_COND_GT) {
      F_Or *f_or = f_exists->add_and()->add_or();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());
        F_And *f_and = f_or->add_and();
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, true);
      }
    }
    else if (rel == IR_COND_LE || rel == IR_COND_LT) {
      F_And *f_and = f_exists->add_and();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());        
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, 1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, true);
      }
    }
    else if (rel == IR_COND_EQ) {
      F_Or *f_or = f_exists->add_and()->add_or();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());
        F_And *f_and = f_or->add_and();
        
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, false);
        
        for (int j = 0; j < v.size(); j++)
          if (j != i) {
            Variable_ID e2 = f_exists->declare(tmp_e());
            GEQ_Handle h2 = f_and->add_GEQ();
            h2.update_coef(e, -1);
            h2.update_coef(e2, 1);
            
            exp2formula(ir, r, f_and, freevars, v[j], e2, side, IR_COND_EQ, false);
          }
      }
      
      for (int i = 0; i < v.size(); i++)
        delete v[i];
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    if (destroy)
      delete repr;
  }
  case IR_OP_MAX:
  {
    std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
    
    F_Exists *f_exists = f_root->add_exists();
    
    if (rel == IR_COND_LE || rel == IR_COND_LT) {
      F_Or *f_or = f_exists->add_and()->add_or();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());
        F_And *f_and = f_or->add_and();
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, 1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, true);
      }
    }
    else if (rel == IR_COND_GE || rel == IR_COND_GT) {
      F_And *f_and = f_exists->add_and();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());        
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, true);
      }
    }
    else if (rel == IR_COND_EQ) {
      F_Or *f_or = f_exists->add_and()->add_or();
      for (int i = 0; i < v.size(); i++) {
        Variable_ID e = f_exists->declare(tmp_e());
        F_And *f_and = f_or->add_and();
        
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -1);
        
        exp2formula(ir, r, f_and, freevars, v[i], e, side, IR_COND_EQ, false);
        
        for (int j = 0; j < v.size(); j++)
          if (j != i) {
            Variable_ID e2 = f_exists->declare(tmp_e());
            GEQ_Handle h2 = f_and->add_GEQ();
            h2.update_coef(e, 1);
            h2.update_coef(e2, -1);
            
            exp2formula(ir, r, f_and, freevars, v[j], e2, side, IR_COND_EQ, false);
          }
      }
      
      for (int i = 0; i < v.size(); i++)
        delete v[i];
    }
    else
      throw std::invalid_argument("unsupported condition type");
    
    if (destroy)
      delete repr;
  }
  case IR_OP_NULL:
    break;
  default:
    throw ir_exp_error("unsupported operand type");
  }
}


//-----------------------------------------------------------------------------
// Build dependence relation for two array references.
// -----------------------------------------------------------------------------
Relation arrays2relation(IR_Code *ir, std::vector<Free_Var_Decl*> &freevars,
                         const IR_ArrayRef *ref_src, const Relation &IS_w,
                         const IR_ArrayRef *ref_dst, const Relation &IS_r) {
  Relation &IS1 = const_cast<Relation &>(IS_w);
  Relation &IS2 = const_cast<Relation &>(IS_r);
  
  Relation r(IS1.n_set(), IS2.n_set());
  
  for (int i = 1; i <= IS1.n_set(); i++)
    r.name_input_var(i, IS1.set_var(i)->name());
  
  for (int i = 1; i <= IS2.n_set(); i++)
    r.name_output_var(i, IS2.set_var(i)->name()+"'");
  
  IR_Symbol *sym_src = ref_src->symbol();
  IR_Symbol *sym_dst = ref_dst->symbol();
  if (*sym_src != *sym_dst) {
    r.add_or(); // False Relation
    delete sym_src;
    delete sym_dst;
    return r;
  }
  else {
    delete sym_src;
    delete sym_dst;
  }
  
  F_And *f_root = r.add_and();
  
  for (int i = 0; i < ref_src->n_dim(); i++) {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    
    CG_outputRepr *repr_src = ref_src->index(i);
    CG_outputRepr *repr_dst = ref_dst->index(i);
    
    bool has_complex_formula = false;
    try {
      exp2formula(ir, r, f_and, freevars, repr_src, e1, 'w', IR_COND_EQ, false);
      exp2formula(ir, r, f_and, freevars, repr_dst, e2, 'r', IR_COND_EQ, false);
    }
    catch (const ir_exp_error &e) {
      has_complex_formula = true;
    }
    
    if (!has_complex_formula) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(e1, 1);
      h.update_coef(e2, -1);
    }
    
    repr_src->clear();
    repr_dst->clear();
    delete repr_src;
    delete repr_dst;
  }
  
  // add iteration space restriction
  r = Restrict_Domain(r, copy(IS1));
  r = Restrict_Range(r, copy(IS2));
  
  // reset the output variable names lost in restriction
  for (int i = 1; i <= IS2.n_set(); i++)
    r.name_output_var(i, IS2.set_var(i)->name()+"'");
  
  return r;
}


//-----------------------------------------------------------------------------
// Convert array dependence relation into set of dependence vectors, assuming
// ref_w is lexicographically before ref_r in the source code.
// -----------------------------------------------------------------------------
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > relation2dependences (const IR_ArrayRef *ref_src, const IR_ArrayRef *ref_dst, const Relation &r) {
  assert(r.n_inp() == r.n_out());
  
  std::vector<DependenceVector> dependences1, dependences2;  
  std::stack<DependenceLevel> working;
  working.push(DependenceLevel(r, r.n_inp()));
  
  while (!working.empty()) {
    DependenceLevel dep = working.top();
    working.pop();
    
    // No dependence exists, move on.
    if (!dep.r.is_satisfiable())
      continue;
    
    if (dep.level == r.n_inp()) {
      DependenceVector dv;
      
      // for loop independent dependence, use lexical order to
      // determine the correct source and destination
      if (dep.dir == 0) {
        if (*ref_src == *ref_dst)
          continue; // trivial self zero-dependence
        
        if (ref_src->is_write()) {
          if (ref_dst->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_dst->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
        
      }
      else if (dep.dir == 1) {
        if (ref_src->is_write()) {
          if (ref_dst->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_dst->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
      }
      else { // dep.dir == -1
        if (ref_dst->is_write()) {
          if (ref_src->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_src->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
      }
      
      dv.lbounds = dep.lbounds;
      dv.ubounds = dep.ubounds;
      dv.sym = ref_src->symbol();
      
      if (dep.dir == 0 || dep.dir == 1)
        dependences1.push_back(dv);
      else
        dependences2.push_back(dv);
    }
    else {
      // now work on the next dimension level
      int level = ++dep.level;
      
      coef_t lbound, ubound;
      Relation delta = Deltas(copy(dep.r));
      delta.query_variable_bounds(delta.set_var(level), lbound, ubound);
      
      if (dep.dir == 0) {
        if (lbound > 0) {
          dep.dir = 1;
          dep.lbounds[level-1] = lbound;
          dep.ubounds[level-1] = ubound;
          
          working.push(dep);
        }
        else if (ubound < 0) {
          dep.dir = -1;
          dep.lbounds[level-1] = -ubound;
          dep.ubounds[level-1] = -lbound;
          
          working.push(dep);
        }
        else {
          // split the dependence vector into flow- and anti-dependence
          // for the first non-zero distance, also separate zero distance
          // at this level.
          {
            DependenceLevel dep2 = dep;
            
            dep2.lbounds[level-1] =  0;
            dep2.ubounds[level-1] =  0;
            
            F_And *f_root = dep2.r.and_with_and();
            EQ_Handle h = f_root->add_EQ();
            h.update_coef(dep2.r.input_var(level), 1);
            h.update_coef(dep2.r.output_var(level), -1);
            
            working.push(dep2);
          }
          
          if (lbound < 0 && *ref_src != *ref_dst) {
            DependenceLevel dep2 = dep;
            
            F_And *f_root = dep2.r.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(dep2.r.input_var(level), 1);
            h.update_coef(dep2.r.output_var(level), -1);
            h.update_const(-1);
            
            // get tighter bounds under new constraints
            coef_t lbound, ubound;
            delta = Deltas(copy(dep2.r));
            delta.query_variable_bounds(delta.set_var(level),
                                        lbound, ubound);
            
            dep2.dir = -1;            
            dep2.lbounds[level-1] = max(-ubound,static_cast<coef_t>(1)); // use max() to avoid Omega retardness
            dep2.ubounds[level-1] = -lbound;
            
            working.push(dep2);
          }
          
          if (ubound > 0) {
            DependenceLevel dep2 = dep;
            
            F_And *f_root = dep2.r.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(dep2.r.input_var(level), -1);
            h.update_coef(dep2.r.output_var(level), 1);
            h.update_const(-1);
            
            // get tighter bonds under new constraints
            coef_t lbound, ubound;
            delta = Deltas(copy(dep2.r));
            delta.query_variable_bounds(delta.set_var(level),
                                        lbound, ubound);
            dep2.dir = 1;
            dep2.lbounds[level-1] = max(lbound,static_cast<coef_t>(1)); // use max() to avoid Omega retardness
            dep2.ubounds[level-1] = ubound;
            
            working.push(dep2);
          }
        }
      }
      // now deal with dependence vector with known direction
      // determined at previous levels
      else {
        // For messy bounds, further test to see if the dependence distance
        // can be reduced to positive/negative.  This is an omega hack.
        if (lbound == negInfinity && ubound == posInfinity) {
          {
            Relation t = dep.r;
            F_And *f_root = t.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(t.input_var(level), 1);
            h.update_coef(t.output_var(level), -1);
            h.update_const(-1);
            
            if (!t.is_satisfiable()) {
              lbound = 0;
            }
          }
          {
            Relation t = dep.r;
            F_And *f_root = t.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(t.input_var(level), -1);
            h.update_coef(t.output_var(level), 1);
            h.update_const(-1);
            
            if (!t.is_satisfiable()) {
              ubound = 0;
            }
          }
        }
        
        // Same thing as above, test to see if zero dependence
        // distance possible.
        if (lbound == 0 || ubound == 0) {
          Relation t = dep.r;
          F_And *f_root = t.and_with_and();
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(t.input_var(level), 1);
          h.update_coef(t.output_var(level), -1);
          
          if (!t.is_satisfiable()) {
            if (lbound == 0)
              lbound = 1;
            if (ubound == 0)
              ubound = -1;
          }
        }
        
        if (dep.dir == -1) {
          dep.lbounds[level-1] = -ubound;
          dep.ubounds[level-1] = -lbound;
        }
        else { // dep.dir == 1
          dep.lbounds[level-1] = lbound;
          dep.ubounds[level-1] = ubound;
        }
        
        working.push(dep);
      }
    }
  }
  
  return std::make_pair(dependences1, dependences2);
}


//-----------------------------------------------------------------------------
// Convert a boolean expression to omega relation.  "destroy" means shallow
// deallocation of "repr", not freeing the actual code inside.
//-----------------------------------------------------------------------------
void exp2constraint(IR_Code *ir, Relation &r, F_And *f_root,
                    std::vector<Free_Var_Decl *> &freevars,
                    CG_outputRepr *repr, bool destroy) {
  IR_CONDITION_TYPE cond = ir->QueryBooleanExpOperation(repr);
  switch (cond) {
  case IR_COND_LT:
  case IR_COND_LE:
  case IR_COND_EQ:
  case IR_COND_GT:
  case IR_COND_GE: {
    F_Exists *f_exist = f_root->add_exists();
    Variable_ID e = f_exist->declare();
    F_And *f_and = f_exist->add_and();
    std::vector<omega::CG_outputRepr *> op = ir->QueryExpOperand(repr);
    exp2formula(ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ, true);
    exp2formula(ir, r, f_and, freevars, op[1], e, 's', cond, true);
    if (destroy)
      delete repr;
    break;
  }
  case IR_COND_NE: {
    F_Exists *f_exist = f_root->add_exists();
    Variable_ID e = f_exist->declare();
    F_Or *f_or = f_exist->add_or();
    F_And *f_and = f_or->add_and();
    std::vector<omega::CG_outputRepr *> op = ir->QueryExpOperand(repr);
    exp2formula(ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ, false);
    exp2formula(ir, r, f_and, freevars, op[1], e, 's', IR_COND_GT, false);
    
    f_and = f_or->add_and();
    exp2formula(ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ, true);
    exp2formula(ir, r, f_and, freevars, op[1], e, 's', IR_COND_LT, true);
    
    if (destroy)
      delete repr;
    break;
  }    
  default:
    throw ir_exp_error("unrecognized conditional expression");
  }
}





// inline void exp2formula(IR_Code *ir, Relation &r, F_And *f_root,
//                         std::vector<Free_Var_Decl*> &freevars,
//                         const CG_outputRepr *repr, Variable_ID lhs, char side, char rel) {
//   exp2formula(ir, r, f_root, freevars, const_cast<CG_outputRepr *>(repr), lhs, side, rel, false);
// }







//-----------------------------------------------------------------------------
// Convert suif expression tree to omega relation.
//-----------------------------------------------------------------------------

// void suif2formula(Relation &r, F_And *f_root,
//                   std::vector<Free_Var_Decl*> &freevars,
//                   operand op, Variable_ID lhs,
//                   char side, char rel) {
//   if (op.is_immed()) {
//     immed im = op.immediate();

//     if (im.is_integer()) {
//       int c = im.integer();

//       if (rel == '>') {
//         GEQ_Handle h = f_root->add_GEQ();
//         h.update_coef(lhs, 1);
//         h.update_const(-1*c);
//       }
//       else if (rel == '<') {
//         GEQ_Handle h = f_root->add_GEQ();
//         h.update_coef(lhs, -1);
//         h.update_const(c);
//       }
//       else { // '='
//         EQ_Handle h = f_root->add_EQ();
//         h.update_coef(lhs, 1);
//         h.update_const(-1*c);
//       }
//     }
//     else {
//       return;  //add Function in the future
//     }
//   }
//   else if (op.is_symbol()) {
//     String s = op.symbol()->name();
//     Variable_ID e = find_index(r, s, side);

//     if (e == NULL) { // must be free variable
//       Free_Var_Decl *t = NULL;
//       for (unsigned i = 0; i < freevars.size(); i++) {
//         String ss = freevars[i]->base_name();
//         if (s == ss) {
//           t = freevars[i];
//           break;
//         }
//       }

//       if (t == NULL) {
//         t = new Free_Var_Decl(s);
//         freevars.insert(freevars.end(), t);
//       }

//       e = r.get_local(t);
//     }

//     if (rel == '>') {
//       GEQ_Handle h = f_root->add_GEQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e, -1);
//     }
//     else if (rel == '<') {
//       GEQ_Handle h = f_root->add_GEQ();
//       h.update_coef(lhs, -1);
//       h.update_coef(e, 1);
//     }
//     else { // '='
//       EQ_Handle h = f_root->add_EQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e, -1);
//     }
//   }
//   else if (op.is_instr())
//     suif2formula(r, f_root, freevars, op.instr(), lhs, side, rel);
// }


// void suif2formula(Relation &r, F_And *f_root,
//                   std::vector<Free_Var_Decl*> &freevars,
//                   instruction *ins, Variable_ID lhs,
//                   char side, char rel) {
//   if (ins->opcode() == io_cpy) {
//     suif2formula(r, f_root, freevars, ins->src_op(0), lhs, side, rel);
//   }
//   else if (ins->opcode() == io_add || ins->opcode() == io_sub) {
//     F_Exists *f_exists = f_root->add_exists();
//     Variable_ID e1 = f_exists->declare(tmp_e());
//     Variable_ID e2 = f_exists->declare(tmp_e());
//     F_And *f_and = f_exists->add_and();

//     int add_or_sub = ins->opcode() == io_add ? 1 : -1;
//     if (rel == '>') {
//       GEQ_Handle h = f_and->add_GEQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e1, -1);
//       h.update_coef(e2, -1 * add_or_sub);
//     }
//     else if (rel == '<') {
//       GEQ_Handle h = f_and->add_GEQ();
//       h.update_coef(lhs, -1);
//       h.update_coef(e1, 1);
//       h.update_coef(e2, 1 * add_or_sub);
//     }
//     else { // '='
//       EQ_Handle h = f_and->add_EQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e1, -1);
//       h.update_coef(e2, -1 * add_or_sub);
//     }

//     suif2formula(r, f_and, freevars, ins->src_op(0), e1, side, '=');
//     suif2formula(r, f_and, freevars, ins->src_op(1), e2, side, '=');
//   }
//   else if (ins->opcode() == io_mul) {
//     operand op1 = ins->src_op(0);
//     operand op2 = ins->src_op(1);

//     if (!op1.is_immed() && !op2.is_immed())
//       return;  // add Function in the future
//     else {
//       operand op;
//       immed im;
//       if (op1.is_immed()) {
//         im = op1.immediate();
//         op = op2;
//       }
//       else {
//         im = op2.immediate();
//         op = op1;
//       }

//       if (!im.is_integer())
//         return; //add Function in the future
//       else {
//         int c = im.integer();

//         F_Exists *f_exists = f_root->add_exists();
//         Variable_ID e = f_exists->declare(tmp_e());
//         F_And *f_and = f_exists->add_and();

//         if (rel == '>') {
//           GEQ_Handle h = f_and->add_GEQ();
//           h.update_coef(lhs, 1);
//           h.update_coef(e, -c);
//         }
//         else if (rel == '<') {
//           GEQ_Handle h = f_and->add_GEQ();
//           h.update_coef(lhs, -1);
//           h.update_coef(e, c);
//         }
//         else {
//           EQ_Handle h = f_and->add_EQ();
//           h.update_coef(lhs, 1);
//           h.update_coef(e, -c);
//         }

//         suif2formula(r, f_and, freevars, op, e, side, '=');
//       }
//     }
//   }
//   else if (ins->opcode() == io_div) {
//     operand op1 = ins->src_op(0);
//     operand op2 = ins->src_op(1);

//     if (!op2.is_immed())
//       return;  //add Function in the future
//     else {
//       immed im = op2.immediate();

//       if (!im.is_integer())
//         return;  //add Function in the future
//       else {
//         int c = im.integer();

//         F_Exists *f_exists = f_root->add_exists();
//         Variable_ID e = f_exists->declare(tmp_e());
//         F_And *f_and = f_exists->add_and();

//         if (rel == '>') {
//           GEQ_Handle h = f_and->add_GEQ();
//           h.update_coef(lhs, c);
//           h.update_coef(e, -1);
//         }
//         else if (rel == '<') {
//           GEQ_Handle h = f_and->add_GEQ();
//           h.update_coef(lhs, -c);
//           h.update_coef(e, 1);
//         }
//         else {
//           EQ_Handle h = f_and->add_EQ();
//           h.update_coef(lhs, c);
//           h.update_coef(e, -1);
//         }

//         suif2formula(r, f_and, freevars, op1, e, side, '=');
//       }
//     }
//   }       
//   else if (ins->opcode() == io_neg) {    
//     F_Exists *f_exists = f_root->add_exists();
//     Variable_ID e = f_exists->declare(tmp_e());
//     F_And *f_and = f_exists->add_and();

//     if (rel == '>') {
//       GEQ_Handle h = f_and->add_GEQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e, 1);
//     }
//     else if (rel == '<') {
//       GEQ_Handle h = f_and->add_GEQ();
//       h.update_coef(lhs, -1);
//       h.update_coef(e, -1);
//     }
//     else {
//       EQ_Handle h = f_and->add_EQ();
//       h.update_coef(lhs, 1);
//       h.update_coef(e, 1);
//     }

//     suif2formula(r, f_and, freevars, ins->src_op(0), e, side, '=');
//   }
//   else if (ins->opcode() == io_min) {
//     operand op1 = ins->src_op(0);
//     operand op2 = ins->src_op(1);

//     F_Exists *f_exists = f_root->add_exists();
//     Variable_ID e1 = f_exists->declare(tmp_e());
//     Variable_ID e2 = f_exists->declare(tmp_e());
//     F_And *f_and = f_exists->add_and();

//     if (rel == '>') {
//       F_Or *f_or = f_and->add_or();
//       F_And *f_and1 = f_or->add_and();
//       GEQ_Handle h1 = f_and1->add_GEQ();
//       h1.update_coef(lhs, 1);
//       h1.update_coef(e1, -1);
//       F_And *f_and2 = f_or->add_and();
//       GEQ_Handle h2 = f_and2->add_GEQ();
//       h2.update_coef(lhs, 1);
//       h2.update_coef(e2, -1);
//     }
//     else if (rel == '<') {
//       GEQ_Handle h1 = f_and->add_GEQ();
//       h1.update_coef(lhs, -1);
//       h1.update_coef(e1, 1);
//       GEQ_Handle h2 = f_and->add_GEQ();
//       h2.update_coef(lhs, -1);
//       h2.update_coef(e2, 1);
//     }
//     else {
//       F_Or *f_or = f_and->add_or();
//       F_And *f_and1 = f_or->add_and();
//       EQ_Handle h1 = f_and1->add_EQ();
//       h1.update_coef(lhs, 1);
//       h1.update_coef(e1, -1);
//       GEQ_Handle h2 = f_and1->add_GEQ();
//       h2.update_coef(e1, -1);
//       h2.update_coef(e2, 1);
//       F_And *f_and2 = f_or->add_and();
//       EQ_Handle h3 = f_and2->add_EQ();
//       h3.update_coef(lhs, 1);
//       h3.update_coef(e2, -1);
//       GEQ_Handle h4 = f_and2->add_GEQ();
//       h4.update_coef(e1, 1);
//       h4.update_coef(e2, -1);
//     }

//     suif2formula(r, f_and, freevars, op1, e1, side, '=');
//     suif2formula(r, f_and, freevars, op2, e2, side, '=');
//   }
//   else if (ins->opcode() == io_max) {
//     operand op1 = ins->src_op(0);
//     operand op2 = ins->src_op(1);

//     F_Exists *f_exists = f_root->add_exists();
//     Variable_ID e1 = f_exists->declare(tmp_e());
//     Variable_ID e2 = f_exists->declare(tmp_e());
//     F_And *f_and = f_exists->add_and();

//     if (rel == '>') {
//       GEQ_Handle h1 = f_and->add_GEQ();
//       h1.update_coef(lhs, 1);
//       h1.update_coef(e1, -1);
//       GEQ_Handle h2 = f_and->add_GEQ();
//       h2.update_coef(lhs, 1);
//       h2.update_coef(e2, -1);
//     }
//     else if (rel == '<') {
//       F_Or *f_or = f_and->add_or();
//       F_And *f_and1 = f_or->add_and();
//       GEQ_Handle h1 = f_and1->add_GEQ();
//       h1.update_coef(lhs, -1);
//       h1.update_coef(e1, 1);
//       F_And *f_and2 = f_or->add_and();
//       GEQ_Handle h2 = f_and2->add_GEQ();
//       h2.update_coef(lhs, -1);
//       h2.update_coef(e2, 1);
//     }
//     else {
//       F_Or *f_or = f_and->add_or();
//       F_And *f_and1 = f_or->add_and();
//       EQ_Handle h1 = f_and1->add_EQ();
//       h1.update_coef(lhs, 1);
//       h1.update_coef(e1, -1);
//       GEQ_Handle h2 = f_and1->add_GEQ();
//       h2.update_coef(e1, 1);
//       h2.update_coef(e2, -1);
//       F_And *f_and2 = f_or->add_and();
//       EQ_Handle h3 = f_and2->add_EQ();
//       h3.update_coef(lhs, 1);
//       h3.update_coef(e2, -1);
//       GEQ_Handle h4 = f_and2->add_GEQ();
//       h4.update_coef(e1, -1);
//       h4.update_coef(e2, 1);
//     }

//     suif2formula(r, f_and, freevars, op1, e1, side, '=');
//     suif2formula(r, f_and, freevars, op2, e2, side, '=');
//   }      
// }

//-----------------------------------------------------------------------------
// Generate iteration space constraints
//-----------------------------------------------------------------------------

// void add_loop_stride_constraints(Relation &r, F_And *f_root,
//                                  std::vector<Free_Var_Decl*> &freevars,
//                                  tree_for *tnf, char side) {

//   std::string name(tnf->index()->name());
//   int dim = 0;
//   for (;dim < r.n_set(); dim++)
//     if (r.set_var(dim+1)->name() == name)
//       break;

//   Relation bound = get_loop_bound(r, dim);

//   operand op = tnf->step_op();
//   if (!op.is_null()) {
//     if (op.is_immed()) {
//       immed im = op.immediate();
//       if (im.is_integer()) {
//         int c = im.integer();

//         if (c != 1 && c != -1)
//           add_loop_stride(r, bound, dim, c);
//       }
//       else
//         assert(0); // messy stride
//     }
//     else
//       assert(0);  // messy stride
//   }
// }

// void add_loop_bound_constraints(IR_Code *ir, Relation &r, F_And *f_root,
//                                 std::vector<Free_Var_Decl*> &freevars,
//                                 tree_for *tnf,
//                                 char upper_or_lower, char side, IR_CONDITION_TYPE rel) {
//   Variable_ID v = find_index(r, tnf->index()->name(), side);

//   tree_node_list *tnl;

//   if (upper_or_lower == 'u')
//     tnl = tnf->ub_list();
//   else
//     tnl = tnf->lb_list();

//   tree_node_list_iter iter(tnl);
//   while (!iter.is_empty()) {
//     tree_node *tn = iter.step();
//     if (tn->kind() != TREE_INSTR)
//       break; // messy bounds

//     instruction *ins = static_cast<tree_instr *>(tn)->instr();


//     if (upper_or_lower == 'u' && (tnf->test() == FOR_SLT || tnf->test() == FOR_ULT)) {
//       operand op1(ins->clone());
//       operand op2(new in_ldc(type_s32, operand(), immed(1)));
//       instruction *t = new in_rrr(io_sub, op1.type(), operand(), op1, op2);

//       CG_suifRepr *repr = new CG_suifRepr(operand(t));
//       exp2formula(ir, r, f_root, freevars, repr, v, side, rel, true);
//       delete t;
//     }
//     else if (tnf->test() == FOR_SLT || tnf->test() == FOR_SLTE || tnf->test() == FOR_ULT || tnf->test() == FOR_ULTE) {
//       CG_suifRepr *repr = new CG_suifRepr(operand(ins));
//       exp2formula(ir, r, f_root, freevars, repr, v, side, rel, true);
//     }
//     else
//       assert(0);
//   }
// } 


// Relation loop_iteration_space(std::vector<Free_Var_Decl*> &freevars,
//                               tree_node *tn, std::vector<tree_for*> &loops) {
//   Relation r(loops.size());
//   for (unsigned i = 0; i < loops.size(); i++) {
//     String s = loops[i]->index()->name();
//     r.name_set_var(i+1, s);
//   }

//   F_And *f_root = r.add_and();

//   std::vector<tree_for *> outer = find_outer_loops(tn);
//   std::vector<LexicalOrderType> loops_lex(loops.size(), LEX_UNKNOWN);

//   for (unsigned i = 0; i < outer.size(); i++) {
//     unsigned j;

//     for (j = 0; j < loops.size(); j++) {
//       if (outer[i] == loops[j]) {
//         loops_lex[j] = LEX_MATCH;
//         break;
//       } else if (outer[i]->index() == loops[j]->index()) {
//         loops_lex[j] = lexical_order(outer[i],loops[j]);
//         break;
//       }
//     }

//     if (j != loops.size()) {
//       add_loop_bound_constraints(r, f_root, freevars, outer[i], 'l', 's', '>');
//       add_loop_bound_constraints(r, f_root, freevars, outer[i], 'u', 's', '<');
//       add_loop_stride_constraints(r,f_root, freevars, outer[i], 's');
//     }
//   }

//   // Add degenerated constraints for non-enclosing loops for this
//   // statement. We treat low-dim space as part of whole
//   // iteration space.
//   LexicalOrderType lex = LEX_MATCH;
//   for (unsigned i = 0; i < loops.size(); i++) {
//     if (loops_lex[i] != 0) {
//       if (lex == LEX_MATCH)
//         lex = loops_lex[i];
//       continue;
//     }

//     if (lex == LEX_MATCH) {
//       for (unsigned j = i+1; j < loops.size(); j++) {
//         if (loops_lex[j] == LEX_BEFORE || loops_lex[j] == LEX_AFTER) {
//           lex = loops_lex[j];
//           break;
//         }
//       }
//     }

//     if (lex == LEX_MATCH)
//       lex = lexical_order(tn, loops[i]);

//     if (lex == LEX_BEFORE)
//       add_loop_bound_constraints(r, f_root, freevars, loops[i], 'l', 's', '=');
//     else
//       add_loop_bound_constraints(r, f_root, freevars, loops[i], 'u', 's', '=');
//   }

//   return r;
// }

// Relation arrays2relation(std::vector<Free_Var_Decl*> &freevars,
//                          in_array *ia_w, const Relation &IS1_,
//                          in_array *ia_r, const Relation &IS2_) {
//   Relation &IS1 = const_cast<Relation &>(IS1_);
//   Relation &IS2 = const_cast<Relation &>(IS2_);

//   Relation r(IS1.n_set(), IS2.n_set());

//   for (int i = 1; i <= IS1.n_set(); i++)
//     r.name_input_var(i, IS1.set_var(i)->name());

//   for (int i = 1; i <= IS2.n_set(); i++)
//     r.name_output_var(i, IS2.set_var(i)->name()+"'");

//   if (get_sym_of_array(ia_w) != get_sym_of_array(ia_r)) {
//     r.add_or(); // False Relation
//     return r;
//   }

//   F_And *f_root = r.add_and();

//   for (unsigned i = 0; i < ia_w->dims(); i++) {
//     F_Exists *f_exists = f_root->add_exists();
//     Variable_ID e = f_exists->declare(tmp_e());
//     F_And *f_and = f_exists->add_and();

//     suif2formula(r, f_and, freevars, ia_w->index(i), e, 'w', '=');
//     suif2formula(r, f_and, freevars, ia_r->index(i), e, 'r', '=');
//   }

//   // add iteration space restriction
//   r = Restrict_Domain(r, copy(IS1));
//   r = Restrict_Range(r, copy(IS2));

//   // reset the output variable names lost in restriction
//   for (int i = 1; i <= IS2.n_set(); i++)
//     r.name_output_var(i, IS2.set_var(i)->name()+"'");

//   return r;
// }


// std::vector<DependenceVector> relation2dependences (IR_Code *ir, in_array *ia_w, in_array *ia_r, const Relation &r) {
//   assert(r.n_inp() == r.n_out());

//   std::vector<DependenceVector> dependences;

//   std::stack<DependenceLevel> working;
//   working.push(DependenceLevel(r, r.n_inp()));

//   while (!working.empty()) {
//     DependenceLevel dep = working.top();
//     working.pop();

//     // No dependence exists, move on.
//     if (!dep.r.is_satisfiable())
//       continue;

//     if (dep.level == r.n_inp()) {
//       DependenceVector dv;

//       // for loop independent dependence, use lexical order to
//       // determine the correct source and destination
//       if (dep.dir == 0) {
//         LexicalOrderType order = lexical_order(ia_w->parent(), ia_r->parent());

//         if (order == LEX_MATCH)
//           continue; //trivial self zero-dependence
//         else if (order == LEX_AFTER) {
//           dv.src = new IR_suifArrayRef(ir, ia_r);
//           dv.dst = new IR_suifArrayRef(ir, ia_w);
//         }
//         else {
//           dv.src = new IR_suifArrayRef(ir, ia_w);
//           dv.dst = new IR_suifArrayRef(ir,ia_r);
//         }
//       }
//       else if (dep.dir == 1) {
//         dv.src = new IR_suifArrayRef(ir, ia_w);
//         dv.dst = new IR_suifArrayRef(ir, ia_r);
//       }
//       else { // dep.dir == -1
//         dv.src = new IR_suifArrayRef(ir, ia_r);
//         dv.dst = new IR_suifArrayRef(ir, ia_w);
//       }

//       dv.lbounds = dep.lbounds;
//       dv.ubounds = dep.ubounds;

//  //      // set the dependence type
// //       if (is_lhs(dv.source) && is_lhs(dv.dest))
// //         dv.type = 'o';
// //       else if (!is_lhs(dv.source) && ! is_lhs(dv.dest))
// //         dv.type = 'i';
// //       else if (is_lhs(dv.source))
// //         dv.type = 'f';
// //       else
// //         dv.type = 'a';

//       dependences.push_back(dv);
//     }
//     else {
//       // now work on the next dimension level
//       int level = ++dep.level;

//       coef_t lbound, ubound;
//       Relation delta = Deltas(copy(dep.r));
//       delta.query_variable_bounds(delta.set_var(level), lbound, ubound);

//       if (dep.dir == 0) {
//         if (lbound > 0) {
//           dep.dir = 1;
//           dep.lbounds[level-1] = lbound;
//           dep.ubounds[level-1] = ubound;

//           working.push(dep);
//         }
//         else if (ubound < 0) {
//           dep.dir = -1;
//           dep.lbounds[level-1] = -ubound;
//           dep.ubounds[level-1] = -lbound;

//           working.push(dep);
//         }
//         else {
//           // split the dependence vector into flow- and anti-dependence
//           // for the first non-zero distance, also separate zero distance
//           // at this level.
//           {
//             DependenceLevel dep2 = dep;

//             dep2.lbounds[level-1] =  0;
//             dep2.ubounds[level-1] =  0;

//             F_And *f_root = dep2.r.and_with_and();
//             EQ_Handle h = f_root->add_EQ();
//             h.update_coef(dep2.r.input_var(level), 1);
//             h.update_coef(dep2.r.output_var(level), -1);

//             working.push(dep2);
//           }

//           if (lbound < 0 && ia_w != ia_r) {
//             DependenceLevel dep2 = dep;

//             F_And *f_root = dep2.r.and_with_and();
//             GEQ_Handle h = f_root->add_GEQ();
//             h.update_coef(dep2.r.input_var(level), 1);
//             h.update_coef(dep2.r.output_var(level), -1);
//             h.update_const(-1);

//             // get tighter bounds under new constraints
//             coef_t lbound, ubound;
//             delta = Deltas(copy(dep2.r));
//             delta.query_variable_bounds(delta.set_var(level),
//                                         lbound, ubound);

//             dep2.dir = -1;            
//             dep2.lbounds[level-1] = max(-ubound,static_cast<coef_t>(1)); // use max() to avoid Omega retardness
//             dep2.ubounds[level-1] = -lbound;

//             working.push(dep2);
//           }

//           if (ubound > 0) {
//             DependenceLevel dep2 = dep;

//             F_And *f_root = dep2.r.and_with_and();
//             GEQ_Handle h = f_root->add_GEQ();
//             h.update_coef(dep2.r.input_var(level), -1);
//             h.update_coef(dep2.r.output_var(level), 1);
//             h.update_const(-1);

//             // get tighter bonds under new constraints
//             coef_t lbound, ubound;
//             delta = Deltas(copy(dep2.r));
//             delta.query_variable_bounds(delta.set_var(level),
//                                         lbound, ubound);
//             dep2.dir = 1;
//             dep2.lbounds[level-1] = max(lbound,static_cast<coef_t>(1)); // use max() to avoid Omega retardness
//             dep2.ubounds[level-1] = ubound;

//             working.push(dep2);
//           }
//         }
//       }
//       // now deal with dependence vector with known direction
//       // determined at previous levels
//       else {
//         // For messy bounds, further test to see if the dependence distance
//         // can be reduced to positive/negative.  This is an omega hack.
//         if (lbound == negInfinity && ubound == posInfinity) {
//           {
//             Relation t = dep.r;
//             F_And *f_root = t.and_with_and();
//             GEQ_Handle h = f_root->add_GEQ();
//             h.update_coef(t.input_var(level), 1);
//             h.update_coef(t.output_var(level), -1);
//             h.update_const(-1);

//             if (!t.is_satisfiable()) {
//               lbound = 0;
//             }
//           }
//           {
//             Relation t = dep.r;
//             F_And *f_root = t.and_with_and();
//             GEQ_Handle h = f_root->add_GEQ();
//             h.update_coef(t.input_var(level), -1);
//             h.update_coef(t.output_var(level), 1);
//             h.update_const(-1);

//             if (!t.is_satisfiable()) {
//               ubound = 0;
//             }
//           }
//         }

//         // Same thing as above, test to see if zero dependence
//         // distance possible.
//         if (lbound == 0 || ubound == 0) {
//           Relation t = dep.r;
//           F_And *f_root = t.and_with_and();
//           EQ_Handle h = f_root->add_EQ();
//           h.update_coef(t.input_var(level), 1);
//           h.update_coef(t.output_var(level), -1);

//           if (!t.is_satisfiable()) {
//             if (lbound == 0)
//               lbound = 1;
//             if (ubound == 0)
//               ubound = -1;
//           }
//         }

//         if (dep.dir == -1) {
//           dep.lbounds[level-1] = -ubound;
//           dep.ubounds[level-1] = -lbound;
//         }
//         else { // dep.dir == 1
//           dep.lbounds[level-1] = lbound;
//           dep.ubounds[level-1] = ubound;
//         }

//         working.push(dep);
//       }
//     }
//   }

//   return dependences;
// }

//-----------------------------------------------------------------------------
// Determine whether the loop (starting from 0) in the iteration space
// has only one iteration.
//-----------------------------------------------------------------------------
bool is_single_loop_iteration(const Relation &r, int level, const Relation &known) {
  int n = r.n_set();
  Relation r1 = Intersection(copy(r), Extend_Set(copy(known), n-known.n_set()));
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= level; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  r1 = Range(Restrict_Domain(mapping, r1));
  r1.simplify();
  
  Variable_ID v = r1.set_var(level);
  for (DNF_Iterator di(r1.query_DNF()); di; di++) {
    bool is_single = false;
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++)
      if ((*ei).get_coef(v) != 0 && !(*ei).has_wildcards()) {
        is_single = true;
        break;
      }
    
    if (!is_single)
      return false;
  }
  
  return true;
}




bool is_single_iteration(const Relation &r, int dim) {
  assert(r.is_set());
  const int n = r.n_set();
  
  if (dim >= n)
    return true;
  
  Relation bound = get_loop_bound(r, dim);
  
//   if (!bound.has_single_conjunct())
//     return false;
  
//   Conjunct *c = bound.query_DNF()->single_conjunct();
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    bool is_single = false;
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++)
      if (!(*ei).has_wildcards()) {
        is_single = true;
        break;
      }
    
    if (!is_single)
      return false;
  }
  
  return true;
  
  
  
  
//   Relation r = copy(r_);
//   const int n = r.n_set();
  
//   if (dim >= n)
//     return true;
  
//   Relation bound = get_loop_bound(r, dim);
//   bound = Approximate(bound);
//   Conjunct *c = bound.query_DNF()->single_conjunct();
  
//   return c->n_GEQs() == 0;
  
  
  
  
  
//   Relation r = copy(r_);
//   r.simplify();
//   const int n = r.n_set();
  
//   if (dim >= n)
//     return true;
  
//   for (DNF_Iterator i(r.query_DNF()); i; i++) {
//     std::vector<bool> is_single(n);
//     for (int j = 0; j < dim; j++)
//       is_single[j] = true;
//     for (int j = dim; j < n; j++)
//       is_single[j] = false;
  
//     bool found_new_single = true;
//     while (found_new_single) {
//       found_new_single = false;
  
//       for (EQ_Iterator j = (*i)->EQs(); j; j++) {
//         int saved_pos = -1;
//         for (Constr_Vars_Iter k(*j); k; k++)
//           if ((*k).var->kind() == Set_Var || (*k).var->kind() == Input_Var) {
//             int pos = (*k).var->get_position() - 1;
//             if (!is_single[pos])
//               if (saved_pos == -1)
//                 saved_pos = pos;
//               else {
//                 saved_pos = -1;
//                 break;
//               }
//           }
  
//         if (saved_pos != -1) {
//           is_single[saved_pos] = true;
//           found_new_single = true;
//         }
//       }
  
//       if (is_single[dim])
//         break;
//     }
  
//     if (!is_single[dim])
//       return false;
//   }
  
//   return true;
}

//-----------------------------------------------------------------------------
// Set/get the value of a variable which is know to be constant.
//-----------------------------------------------------------------------------
void assign_const(Relation &r, int dim, int val) {
  const int n = r.n_out();
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  
  for (int i = 1; i <= n; i++) {
    if (i != dim+1) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_coef(mapping.input_var(i), -1);
    }
    else {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_const(-val);
    }
  }
  
  r = Composition(mapping, r);
}


int get_const(const Relation &r, int dim, Var_Kind type) {
//  Relation rr = copy(r);
  Relation &rr = const_cast<Relation &>(r);
  
  Variable_ID v;
  switch (type) {
    // case Set_Var:
    //   v = rr.set_var(dim+1);
    //   break;
  case Input_Var:
    v = rr.input_var(dim+1);
    break;
  case Output_Var:
    v = rr.output_var(dim+1);
    break;
  default:
    throw std::invalid_argument("unsupported variable type");
  }
  
  for (DNF_Iterator di(rr.query_DNF()); di; di++)
    for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
      if ((*ei).is_const(v))
        return (*ei).get_const();
  
  throw std::runtime_error("cannot get variable's constant value");
}






//---------------------------------------------------------------------------
// Get the bound for a specific loop.
//---------------------------------------------------------------------------
Relation get_loop_bound(const Relation &r, int dim) {
  assert(r.is_set());
  const int n = r.n_set();
  
//  Relation r1 = project_onto_levels(copy(r), dim+1, true);
  Relation mapping(n,n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= dim+1; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  Relation r1 = Range(Restrict_Domain(mapping, copy(r)));
  for (int i = 1; i <= n; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  Relation r2 = Project(copy(r1), dim+1, Set_Var);
  
  return Gist(r1, r2, 1);
}

Relation get_loop_bound(const Relation &r, int level, const Relation &known) {
  int n = r.n_set();
  Relation r1 = Intersection(copy(r), Extend_Set(copy(known), n-known.n_set()));
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= level; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  r1 = Range(Restrict_Domain(mapping, r1));
  Relation r2 = Project(copy(r1), level, Set_Var);
  r1 = Gist(r1, r2, 1);
  
  for (int i = 1; i <= n; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  
  return r1;
}



Relation get_max_loop_bound(const std::vector<Relation> &r, int dim) {
  if (r.size() == 0)
    return Relation::Null();
  
  const int n = r[0].n_set();
  Relation res(Relation::False(n));
  for (int i = 0; i < r.size(); i++) {
    Relation &t = const_cast<Relation &>(r[i]);
    if (t.is_satisfiable())
      res = Union(get_loop_bound(t, dim), res);
  }
  
  res.simplify();
  
  return res;
}

Relation get_min_loop_bound(const std::vector<Relation> &r, int dim) {
  if (r.size() == 0)
    return Relation::Null();
  
  const int n = r[0].n_set();
  Relation res(Relation::True(n));
  for (int i = 0; i < r.size(); i++) {
    Relation &t = const_cast<Relation &>(r[i]);
    if (t.is_satisfiable())
      res = Intersection(get_loop_bound(t, dim), res);
  }
  
  res.simplify();
  
  return res;
}

//-----------------------------------------------------------------------------
// Add strident to a loop.
// Issues:
// - Don't work with relations with multiple disjuncts.
// - Omega's dealing with max lower bound is awkward.
//-----------------------------------------------------------------------------
void add_loop_stride(Relation &r, const Relation &bound_, int dim, int stride) {
  F_And *f_root = r.and_with_and();
  Relation &bound = const_cast<Relation &>(bound_);
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    EQ_Handle stride_eq = f_and->add_EQ();
    stride_eq.update_coef(e1, 1);
    stride_eq.update_coef(e2, stride);
    if (!r.is_set())
      stride_eq.update_coef(r.output_var(dim+1), -1);
    else
      stride_eq.update_coef(r.set_var(dim+1), -1);
    F_Or *f_or = f_and->add_or();
    
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
      if ((*gi).get_coef(bound.set_var(dim+1)) > 0) {
        // copy the lower bound constraint
        EQ_Handle h1 = f_or->add_and()->add_EQ();
        GEQ_Handle h2 = f_and->add_GEQ();
        for (Constr_Vars_Iter ci(*gi); ci; ci++) {
          switch ((*ci).var->kind()) {
            // case Set_Var:
          case Input_Var: {
            int pos = (*ci).var->get_position();
            if (pos == dim + 1) {
              h1.update_coef(e1, (*ci).coef);
              h2.update_coef(e1, (*ci).coef);
            }
            else {
              if (!r.is_set()) {
                h1.update_coef(r.output_var(pos), (*ci).coef);
                h2.update_coef(r.output_var(pos), (*ci).coef);
              }
              else {
                h1.update_coef(r.set_var(pos), (*ci).coef);
                h2.update_coef(r.set_var(pos), (*ci).coef);
              }                
            }
            break;
          }
          case Global_Var: {
            Global_Var_ID g = (*ci).var->get_global_var();
            h1.update_coef(r.get_local(g, (*ci).var->function_of()), (*ci).coef);
            h2.update_coef(r.get_local(g, (*ci).var->function_of()), (*ci).coef);
            break;
          }
          default:
            break;
          }
        }
        h1.update_const((*gi).get_const());
        h2.update_const((*gi).get_const());
      }
    }
  }
}


bool is_inner_loop_depend_on_level(const Relation &r, int level, const Relation &known) {
  Relation r1 = Intersection(copy(r), Extend_Set(copy(known), r.n_set()-known.n_set()));
  Relation r2 = copy(r1);
  for (int i = level+1; i <= r2.n_set(); i++)
    r2 = Project(r2, r2.set_var(i));
  r2.simplify(2, 4);
  Relation r3 = Gist(r1, r2);
  
  Variable_ID v = r3.set_var(level);
  for (DNF_Iterator di(r3.query_DNF()); di; di++) {
    for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
      if ((*ei).get_coef(v) != 0)
        return true;
    
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++)
      if ((*gi).get_coef(v) != 0)
        return true;
  }
  
  return false;
}


//-----------------------------------------------------------------------------
// Suppose loop dim is i. Replace i with i+adjustment in loop bounds.
// e.g. do i = 1, n
//        do j = i, n
// after call with dim = 0 and adjustment = 1:
//      do i = 1, n
//        do j = i+1, n
// -----------------------------------------------------------------------------
Relation adjust_loop_bound(const Relation &r, int level, int adjustment) {
  if (adjustment == 0)
    return copy(r);
  
  const int n = r.n_set();
  Relation r1 = copy(r);
  for (int i = level+1; i <= r1.n_set(); i++)
    r1 = Project(r1, r1.set_var(i));
  r1.simplify(2, 4);
  Relation r2 = Gist(copy(r), copy(r1));
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= n; i++)
    if (i == level) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.input_var(level), -1);
      h.update_coef(mapping.output_var(level), 1);
      h.update_const(static_cast<coef_t>(adjustment));
    }
    else {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.input_var(i), -1);
      h.update_coef(mapping.output_var(i), 1);
    }
  
  r2 = Range(Restrict_Domain(mapping, r2));
  r1 = Intersection(r1, r2);
  r1.simplify();
  
  for (int i = 1; i <= n; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  return r1;
}


// commented out on 07/14/2010
// void adjust_loop_bound(Relation &r, int dim, int adjustment, std::vector<Free_Var_Decl *> globals) {
//   assert(r.is_set());

//   if (adjustment == 0)
//     return;

//   const int n = r.n_set();
//   Tuple<std::string> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(copy(r), copy(r1));

//   // remove old bogus global variable conditions since we are going to
//   // update the value.
//   if (globals.size() > 0)
//     r1 = Gist(r1, project_onto_levels(copy(r), 0, true));

//   Relation r4 = Relation::True(n);

//     for (DNF_Iterator di(r2.query_DNF()); di; di++) {
//       for (EQ_Iterator ei = (*di)->EQs(); ei; ei++) {
//         EQ_Handle h = r4.and_with_EQ(*ei);

//         Variable_ID v = r2.set_var(dim+1);
//         coef_t c = (*ei).get_coef(v);
//         if (c != 0)
//           h.update_const(c*adjustment);

//         for (int i = 0; i < globals.size(); i++) {  
//           Variable_ID v = r2.get_local(globals[i]);
//           coef_t c = (*ei).get_coef(v);
//           if (c != 0)
//             h.update_const(c*adjustment);
//         }
//       }

//       for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
//         GEQ_Handle h = r4.and_with_GEQ(*gi);

//         Variable_ID v = r2.set_var(dim+1);
//         coef_t c = (*gi).get_coef(v);
//         if (c != 0)
//           h.update_const(c*adjustment);

//         for (int i = 0; i < globals.size(); i++) {  
//           Variable_ID v = r2.get_local(globals[i]);
//           coef_t c = (*gi).get_coef(v);
//           if (c != 0)
//             h.update_const(c*adjustment);
//         }
//       }
//     }
//     r = Intersection(r1, r4);
// //   }
// //   else
// //     r = Intersection(r1, r2);

//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }


// void adjust_loop_bound(Relation &r, int dim, int adjustment) {
//   assert(r.is_set());
//   const int n = r.n_set();
//   Tuple<String> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(r, copy(r1));

//   Relation r3(n, n);
//   F_And *f_root = r3.add_and();
//   for (int i = 0; i < n; i++) {
//     EQ_Handle h = f_root->add_EQ();
//     h.update_coef(r3.output_var(i+1), 1);
//     h.update_coef(r3.input_var(i+1), -1);
//     if (i == dim)
//       h.update_const(adjustment);
//   }

//   r2 = Range(Restrict_Domain(r3, r2));
//   r = Intersection(r1, r2);

//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }  

// void adjust_loop_bound(Relation &r, int dim, Free_Var_Decl *global_var, int adjustment) {
//   assert(r.is_set());
//   const int n = r.n_set();
//   Tuple<String> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(r, copy(r1));

//   Relation r3(n);
//   Variable_ID v = r2.get_local(global_var);

//   for (DNF_Iterator di(r2.query_DNF()); di; di++) {
//     for (EQ_Iterator ei = (*di)->EQs(); ei; ei++) {
//       coef_t c = (*ei).get_coef(v);
//       EQ_Handle h = r3.and_with_EQ(*ei);
//       if (c != 0)
//         h.update_const(c*adjustment);
//     }
//     for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
//       coef_t c = (*gi).get_coef(v);
//       GEQ_Handle h = r3.and_with_GEQ(*gi);
//       if (c != 0)
//         h.update_const(c*adjustment);
//     }
//   }

//   r = Intersection(r1, r3);
//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }



//------------------------------------------------------------------------------
// If the dimension has value posInfinity, the statement should be privatized
// at this dimension.
//------------------------------------------------------------------------------
// boolean is_private_statement(const Relation &r, int dim) {
//   int n;
//   if (r.is_set())
//     n = r.n_set();
//   else
//     n = r.n_out();

//   if (dim >= n)
//     return false;

//   try {
//     coef_t c;
//     if (r.is_set())
//       c = get_const(r, dim, Set_Var);
//     else
//       c = get_const(r, dim, Output_Var);
//     if (c == posInfinity)
//       return true;
//     else
//       return false;
//   }
//   catch (loop_error e){
//   }

//   return false;
// }



// // ----------------------------------------------------------------------------
// // Calculate v mod dividend based on equations inside relation r.
// // Return posInfinity if it is not a constant.
// // ----------------------------------------------------------------------------
// static coef_t mod_(const Relation &r_, Variable_ID v, int dividend, std::set<Variable_ID> &working_on) {
//   assert(dividend > 0);
//   if (v->kind() == Forall_Var || v->kind() == Exists_Var || v->kind() == Wildcard_Var)
//     return posInfinity;

//   working_on.insert(v);

//   Relation &r = const_cast<Relation &>(r_);
//   Conjunct *c = r.query_DNF()->single_conjunct();

//   for (EQ_Iterator ei(c->EQs()); ei; ei++) {
//     int coef = mod((*ei).get_coef(v), dividend);
//     if (coef != 1 && coef != dividend - 1 )
//       continue;

//     coef_t result = 0;
//     for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
//       if ((*cvi).var != v) {
//         int p = mod((*cvi).coef, dividend);

//         if (p == 0)
//           continue;

//         if (working_on.find((*cvi).var) != working_on.end()) {
//           result = posInfinity;
//           break;
//         }

//         coef_t q = mod_(r, (*cvi).var, dividend, working_on);
//         if (q == posInfinity) {
//           result = posInfinity;
//           break;
//         }
//         result += p * q;
//       }

//     if (result != posInfinity) {
//       result += (*ei).get_const();
//       if (coef == 1)
//         result = -result;
//       working_on.erase(v);

//       return mod(result, dividend);
//     }
//   }

//   working_on.erase(v);
//   return posInfinity;
// }


// coef_t mod(const Relation &r, Variable_ID v, int dividend) {
//   std::set<Variable_ID> working_on = std::set<Variable_ID>();

//   return mod_(r, v, dividend, working_on);
// }



//-----------------------------------------------------------------------------
// Generate mapping relation for permuation.
//-----------------------------------------------------------------------------
Relation permute_relation(const std::vector<int> &pi) {
  const int n = pi.size();
  
  Relation r(n, n);
  F_And *f_root = r.add_and();
  
  for (int i = 0; i < n; i++) {    
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(r.output_var(i+1), 1);
    h.update_coef(r.input_var(pi[i]+1), -1);
  }
  
  return r;
}



//---------------------------------------------------------------------------
// Find the position index variable in a Relation by name.
//---------------------------------------------------------------------------
Variable_ID find_index(Relation &r, const std::string &s, char side) {
  // Omega quirks: assure the names are propagated inside the relation
  r.setup_names();
  
  if (r.is_set()) { // side == 's'
    for (int i = 1; i <= r.n_set(); i++) {
      std::string ss = r.set_var(i)->name();
      if (s == ss) {
        return r.set_var(i);
      }
    }
  }
  else if (side == 'w') {
    for (int i = 1; i <= r.n_inp(); i++) {
      std::string ss = r.input_var(i)->name();
      if (s == ss) {
        return r.input_var(i);
      }
    }
  }
  else { // side == 'r'
    for (int i = 1; i <= r.n_out(); i++) {
      std::string ss = r.output_var(i)->name();
      if (s+"'" == ss) {
        return r.output_var(i);
      }
    }
  }
  
  return NULL;
}

// EQ_Handle get_eq(const Relation &r, int dim, Var_Kind type) {
//   Variable_ID v;
//   switch (type) {
//   case Set_Var:
//     v = r.set_var(dim+1);
//     break;
//   case Input_Var:
//     v = r.input_var(dim+1);
//     break;
//   case Output_Var:
//     v = r.output_var(dim+1);
//     break;
//   default:
//     return NULL;
//   }
//   for (DNF_iterator di(r.query_DNF()); di; di++)
//     for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
//       if ((*ei).get_coef(v) != 0)
//         return (*ei);

//   return NULL;
// }


// std::Pair<Relation, Relation> split_loop(const Relation &r, const Relation &cond) {
//   Relation r1 = Intersection(copy(r), copy(cond));
//   Relation r2 = Intersection(copy(r), Complement(copy(cond)));

//   return std::Pair<Relation, Relation>(r1, r2);
// }
