/*****************************************************************************
 Copyright (C) 2010 University of Utah
 All Rights Reserved.

 Purpose:
   Useful tools to analyze code in compiler IR format.

 Notes:

 History:
   06/2010 Created by Chun Chen.
*****************************************************************************/

#include <iostream>
#include <code_gen/CG_outputBuilder.h>
#include "irtools.hh"
#include "omegatools.hh"
#include "chill_error.hh"

using namespace omega;

// Build IR tree from the source code.  Block type node can only be
// leaf, i.e., there is no further structures inside a block allowed.
std::vector<ir_tree_node *> build_ir_tree(IR_Control *control, ir_tree_node *parent) {
  std::vector<ir_tree_node *> result;
  
  switch (control->type()) {
  case IR_CONTROL_BLOCK: {
    std::vector<IR_Control *> controls = control->ir_->FindOneLevelControlStructure(static_cast<IR_Block *>(control));
    if (controls.size() == 0) {
      ir_tree_node *node = new ir_tree_node;
      node->content = control;
      node->parent = parent;
      node->payload = -1;
      result.push_back(node);
    }
    else {
      delete control;
      for (int i = 0; i < controls.size(); i++)
        switch (controls[i]->type()) {
        case IR_CONTROL_BLOCK: {
          std::vector<ir_tree_node *> t = build_ir_tree(controls[i], parent);
          result.insert(result.end(), t.begin(), t.end());
          break;
        }
        case IR_CONTROL_LOOP: {
          ir_tree_node *node = new ir_tree_node;
          node->content = controls[i];
          node->parent = parent;
          node->children = build_ir_tree(static_cast<IR_Loop *>(controls[i])->body(), node);
          node->payload = -1;
          result.push_back(node);
          break;
        }
        case IR_CONTROL_IF: {
          static int unique_if_identifier = 0;
          
          IR_Block *block = static_cast<IR_If *>(controls[i])->then_body();
          if (block != NULL) {
            ir_tree_node *node = new ir_tree_node;
            node->content = controls[i];
            node->parent = parent;
            node->children = build_ir_tree(block, node);
            node->payload = unique_if_identifier+1;
            result.push_back(node);
          }
          
          
          block = static_cast<IR_If *>(controls[i])->else_body();
          if ( block != NULL) {
            ir_tree_node *node = new ir_tree_node;
            node->content = controls[i]->clone();
            node->parent = parent;
            node->children = build_ir_tree(block, node);
            node->payload = unique_if_identifier;
            result.push_back(node);
          }
          
          unique_if_identifier += 2;
          break;
        }
        default:
          ir_tree_node *node = new ir_tree_node;
          node->content = controls[i];
          node->parent = parent;
          node->payload = -1;
          result.push_back(node);
          break;
        }
    }
    break;
  }
  case IR_CONTROL_LOOP: {
    ir_tree_node *node = new ir_tree_node;
    node->content = control;
    node->parent = parent;
    node->children = build_ir_tree(static_cast<const IR_Loop *>(control)->body(), node);
    node->payload = -1;
    result.push_back(node);
    break;
  }
  default:
    ir_tree_node *node = new ir_tree_node;
    node->content = control;
    node->parent = parent;
    node->payload = -1;
    result.push_back(node);
    break;
  }
  
  return result;
}


// Extract statements from IR tree. Statements returned are ordered in
// lexical order in the source code.
std::vector<ir_tree_node *> extract_ir_stmts(const std::vector<ir_tree_node *> &ir_tree) {
  std::vector<ir_tree_node *> result;
  for (int i = 0; i < ir_tree.size(); i++)
    switch (ir_tree[i]->content->type()) {
    case IR_CONTROL_BLOCK:
      result.push_back(ir_tree[i]);
      break;
    case IR_CONTROL_LOOP: {
      // clear loop payload from previous unsuccessful initialization process
      ir_tree[i]->payload = -1;
      
      std::vector<ir_tree_node *> t = extract_ir_stmts(ir_tree[i]->children);
      result.insert(result.end(), t.begin(), t.end());
      break;
    }      
    case IR_CONTROL_IF: {
      std::vector<ir_tree_node *> t = extract_ir_stmts(ir_tree[i]->children);
      result.insert(result.end(), t.begin(), t.end());
      break;
    }
    default:
      throw std::invalid_argument("invalid ir tree");
    }
  
  return result;
}


bool is_dependence_valid(ir_tree_node *src_node, ir_tree_node *dst_node,
                         const DependenceVector &dv, bool before) {
  std::set<ir_tree_node *> loop_nodes;
  ir_tree_node *itn = src_node;
  
  if (!dv.is_scalar_dependence) {
    while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP)
        loop_nodes.insert(itn);
    }
    
    int last_dim = -1;
    itn = dst_node;
    while (itn->parent != NULL) {
      itn = itn->parent;
      if (itn->content->type() == IR_CONTROL_LOOP
          && loop_nodes.find(itn) != loop_nodes.end()
          && itn->payload > last_dim)
        last_dim = itn->payload;
    }
    
    if (last_dim == -1)
      return true;
    
    for (int i = 0; i <= last_dim; i++) {
      if (dv.lbounds[i] > 0)
        return true;
      else if (dv.lbounds[i] < 0)
        return false;
    }
    
    if (before)
      return true;
    else
      return false;
  }
  
  return true;
  
}



// Test data dependences between two statements. The first statement
// in parameter must be lexically before the second statement in
// parameter.  Returned dependences are all lexicographically
// positive. The first vector in returned pair is dependences from the
// first statement to the second statement and the second vector in
// returned pair is in reverse order.
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > test_data_dependences(
  IR_Code *ir, const CG_outputRepr *repr1, const Relation &IS1,
  const CG_outputRepr *repr2, const Relation &IS2,
  std::vector<Free_Var_Decl*> &freevar, std::vector<std::string> index,
  int i, int j) {
  std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > result;
  
  if (repr1 == repr2) {
    std::vector<IR_ArrayRef *> access = ir->FindArrayRef(repr1);
    
    for (int i = 0; i < access.size(); i++) {
      IR_ArrayRef *a = access[i];
      IR_ArraySymbol *sym_a = a->symbol();
      for (int j = i; j < access.size(); j++) {
        IR_ArrayRef *b = access[j];
        IR_ArraySymbol *sym_b = b->symbol();
        
        if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
          Relation r = arrays2relation(ir, freevar, a, IS1, b, IS2);
          std::pair<std::vector<DependenceVector>,
            std::vector<DependenceVector> > dv =
            relation2dependences(a, b, r);
          result.first.insert(result.first.end(), dv.first.begin(),
                              dv.first.end());
          result.second.insert(result.second.end(), dv.second.begin(),
                               dv.second.end());
        }
        delete sym_b;
      }
      delete sym_a;
      
    }
    
    for (int i = 0; i < access.size(); i++)
      delete access[i];
  } else {
    std::vector<IR_ArrayRef *> access1 = ir->FindArrayRef(repr1);
    std::vector<IR_ArrayRef *> access2 = ir->FindArrayRef(repr2);
    
    for (int i = 0; i < access1.size(); i++) {
      IR_ArrayRef *a = access1[i];
      IR_ArraySymbol *sym_a = a->symbol();
      
      for (int j = 0; j < access2.size(); j++) {
        IR_ArrayRef *b = access2[j];
        IR_ArraySymbol *sym_b = b->symbol();
        if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
          Relation r = arrays2relation(ir, freevar, a, IS1, b, IS2);
          std::pair<std::vector<DependenceVector>,
            std::vector<DependenceVector> > dv =
            relation2dependences(a, b, r);
          
          result.first.insert(result.first.end(), dv.first.begin(),
                              dv.first.end());
          result.second.insert(result.second.end(), dv.second.begin(),
                               dv.second.end());
        }
        delete sym_b;
      }
      delete sym_a;
    }
    
    for (int i = 0; i < access1.size(); i++)
      delete access1[i];
    for (int i = 0; i < access2.size(); i++)
      delete access2[i];
  }
  /*std::pair<std::vector<DependenceVector>,
    std::vector<DependenceVector> > dv =
    ir->FindScalarDeps(repr1, repr2, index, i, j);
    
    
    result.first.insert(result.first.end(), dv.first.begin(),
    dv.first.end());
    result.second.insert(result.second.end(), dv.second.begin(),
    dv.second.end());*/
  /*result.first.insert(result.first.end(), dv.first.begin(),
    dv.first.end());
    result.second.insert(result.second.end(), dv.second.begin(),
    dv.second.end());
  */
  
  return result;
}

