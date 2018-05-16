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

// Build Chill IR tree from the source code (from the front end compiler's AST). 
// Block type node can only be a leaf, i.e., there are no further structures 
// inside a block allowed.
std::vector<ir_tree_node *> build_ir_tree(IR_Control *control, 
                                          ir_tree_node *parent) {
  std::vector<ir_tree_node *> result;
  
  debug_fprintf(stderr, "irtools.cc, build_ir_tree( control, parent)   building a CHILL IR tree \n");

  switch (control->type()) {
  case IR_CONTROL_BLOCK: {
    debug_fprintf(stderr, "irtools.cc L31   case IR_CONTROL_BLOCK\n"); 
    IR_Block *IRCB = static_cast<IR_Block *>(control); 
    std::vector<IR_Control *> controls = control->ir_->FindOneLevelControlStructure(IRCB);

    debug_fprintf(stderr, "irtools.cc BACK FROM FindOneLevelControlStructure()  %d controls\n", controls.size()); 

    if (controls.size() == 0) {
      debug_fprintf(stderr, "controls.size() == 0\n"); 

      ir_tree_node *node = new ir_tree_node;
      node->content = control; 
      node->parent  = parent;
      node->payload = -1;
      result.push_back(node);
    }
    else {
      debug_fprintf(stderr, "controls.size() == %d  (NONZERO)\n", controls.size()); 
      delete control;

      for (int i = 0; i < controls.size(); i++)
        switch (controls[i]->type()) {
        case IR_CONTROL_BLOCK: {
          debug_fprintf(stderr, "controls[%d] is IR_CONTROL_BLOCK\n", i); 
          std::vector<ir_tree_node *> t = build_ir_tree(controls[i], parent);
          result.insert(result.end(), t.begin(), t.end());
          break;
        }
        case IR_CONTROL_LOOP: {
          debug_fprintf(stderr, "controls[%d] is IR_CONTROL_LOOP\n", i); 
          ir_tree_node *node = new ir_tree_node;
          node->content = controls[i];
          node->parent = parent;
          node->children = build_ir_tree(static_cast<IR_Loop *>(controls[i])->body(), node); // recurse
          node->payload = -1;
          result.push_back(node);
          break;
        }
        case IR_CONTROL_IF: {
          debug_fprintf(stderr, "controls[%d] is IR_CONTROL_IF\n", i); 
          static int unique_if_identifier = 0;
          
          IR_If* theif = static_cast<IR_If *>(controls[i]); 
          IR_Block *block = theif->then_body();
          if (block != NULL) {
            ir_tree_node *node = new ir_tree_node;
            node->content = controls[i];
            node->parent = parent;
            node->children = build_ir_tree(block, node); // recurse 
            node->payload = unique_if_identifier+1;
            result.push_back(node);
          }
          
          
          block = theif->else_body();
          if (block != NULL) { 
            debug_fprintf(stderr, "IF_CONTROL has an else\n"); 
            ir_tree_node *node = new ir_tree_node;
            node->content = controls[i]->clone();
            node->parent = parent;
            node->children = build_ir_tree(block, node); // recurse
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
    debug_fprintf(stderr, "case IR_CONTROL_LOOP\n"); 
    ir_tree_node *node = new ir_tree_node;
    node->content = control;
    node->parent  = parent;
    debug_fprintf(stderr, "recursing. build_ir_tree() of CONTROL_LOOP creating children  L122\n");
    node->children = build_ir_tree(
      static_cast<const IR_Loop *>(control)->body(), node);
    node->payload = -1;
    result.push_back(node);
    debug_fprintf(stderr, "recursing. build_ir_tree() of CONTROL_LOOP creating children DONE\n");
    break;
  }
  default:
    ir_tree_node *node = new ir_tree_node;
    node->content = control;
    node->parent  = parent;
    node->payload = -1;
    result.push_back(node);
    break;
  }
  
  debug_fprintf(stderr, "build_ir_tree()  vector result has %ld parts\n", result.size());  
  return result;
}


// Extract statements from IR tree. Statements returned are ordered in
// lexical order in the source code.
std::vector<ir_tree_node *> extract_ir_stmts(const std::vector<ir_tree_node *> &ir_tree) {

  debug_fprintf(stderr, "extract_ir_stmts()   ir_tree.size() %d\n", ir_tree.size()); 
  std::vector<ir_tree_node *> result;
  for (int i = 0; i < ir_tree.size(); i++)
    switch (ir_tree[i]->content->type()) {

    case IR_CONTROL_BLOCK:
      debug_fprintf(stderr, "IR_CONTROL_BLOCK\n"); 
      result.push_back(ir_tree[i]);
      break;

    case IR_CONTROL_LOOP: {
      debug_fprintf(stderr, "IR_CONTROL_LOOP( recursing )\n"); 
      // clear loop payload from previous unsuccessful initialization process
      ir_tree[i]->payload = -1;
      
      std::vector<ir_tree_node *> t = extract_ir_stmts(ir_tree[i]->children);
      
      result.insert(result.end(), t.begin(), t.end());
      break;
    }      
    case IR_CONTROL_IF: {
      debug_fprintf(stderr, "IR_CONTROL_IF( recursing )\n"); 
      std::vector<ir_tree_node *> t = extract_ir_stmts(ir_tree[i]->children);
      result.insert(result.end(), t.begin(), t.end());
      break;
    }
    default:
      throw std::invalid_argument("invalid ir tree");
    }
  
  return result;
}

std::string chill_ir_control_type_string( IR_CONTROL_TYPE type ) {
  switch(type) {
  case IR_CONTROL_BLOCK:  return std::string( "IR_CONTROL_BLOCK");
  case IR_CONTROL_LOOP:   return std::string( "IR_CONTROL_LOOP" );
  case IR_CONTROL_IF:     return std::string( "IR_CONTROL_IF" );
  case IR_CONTROL_WHILE:  return std::string( "IR_CONTROL_WHLIE"); break;
  default: return std::string( "UNKNOWN_IR_NODE_TYPE" ); 
  }
}

std::string chill_ir_node_type_string( ir_tree_node *node ) {
  return chill_ir_control_type_string( node->content->type() );
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
    
    for (int i = 0; i <= last_dim; i++) {
      if (dv.lbounds[i] > 0)
        return true;
      else if (dv.lbounds[i] < 0)
        return false;
    }

    return before;
  }
  
  return true;
  
}


//Anand: Adding function to collect the loop inductive and possibly if conditions
//enclosing a statement

std::vector<omega::CG_outputRepr *> collect_loop_inductive_and_conditionals(
  ir_tree_node * stmt_node) {
  
  std::vector<omega::CG_outputRepr *> to_return;
  ir_tree_node *itn = stmt_node;
  
  while (itn->parent != NULL) {
    itn = itn->parent;
    
    switch (itn->content->type()) {
    case IR_CONTROL_LOOP: {
      IR_Loop *lp = static_cast<IR_Loop *>(itn->content);
      to_return.push_back(lp->lower_bound());
      to_return.push_back(lp->upper_bound());
      
      break;
    }
    case IR_CONTROL_IF: {
      CG_outputRepr *cond =
        static_cast<IR_If *>(itn->content)->condition();
      
      to_return.push_back(cond);
      break;
    }
    default:
      throw std::invalid_argument("invalid ir tree");
    }
  }
  return to_return;
}


// Test data dependences between two statements. The first statement
// in parameter must be lexically before the second statement in
// parameter.  Returned dependences are all lexicographically
// positive. The first vector in returned pair is dependences from the
// first statement to the second statement and the second vector in
// returned pair is in reverse order.
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > test_data_dependences(Loop *loop,
                                                                                               IR_Code *ir, const CG_outputRepr *repr1, const Relation &IS1,
                                                                                               const CG_outputRepr *repr2, const Relation &IS2,
                                                                                               std::vector<Free_Var_Decl*> &freevar, std::vector<std::string> index,
                                                                                               int nestLeveli, int nestLevelj, std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols,
                                                                                               std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols_stringrepr,
                                                                                               std::map<std::string, std::vector<omega::Relation > > &unin_rel,
                                                                                               std::vector<omega::Relation> &dep_relation) {
  std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > result;

  if (repr1 == repr2) {
    std::vector<IR_ArrayRef *> access = ir->FindArrayRef(repr1);

    // Manu:: variables/structures added to identify dependence vectors related to reduction operation
    tempResultMap trMap;
    tempResultMap::iterator ittrMap;
    int ref2Stmt[access.size()]; // mapping of reference to statement
    std::set<int> nrStmts; // stores statements that can't be reduced
    std::set<int> tnrStmts;
    int stmtId = 1;
    int tempStmtId = 1;
    std::map<int,std::set<int> > rMap; // This maps statement number to a set of dependences
    std::map<int, std::set<int> >::iterator itMap;
    for (int i = 0; i < access.size(); i++) {
      ref2Stmt[i] = -1;
    }

    // Manu -- changes for identifying possible reduction operation
    // The below loop nest is used to classify array references into different statements
    mapRefstoStatements(ir,access,ref2Stmt,rMap,tnrStmts,nrStmts);
    //-------------------------------------------------------------
    omega::coef_t lbound[3], ubound[3]; 	// for each  kind of dependence. We can potentially have reduction only if all
    // lbounds match and all ubounds match. At present, we only check the last loop level.
    lbound[0] = lbound[1] = lbound[2] = LLONG_MAX;
    ubound[0] = ubound[1] = ubound[2] = LLONG_MIN;
    //-------------------------------------------------------------

    for (int i = 0; i < access.size(); i++) {

      IR_ArrayRef *a = access[i];
      IR_ArraySymbol *sym_a = a->symbol();

      //Anand changing j= i into j=i+1
      for (int j = i; j < access.size(); j++) {
        IR_ArrayRef *b = access[j];
        IR_ArraySymbol *sym_b = b->symbol();

        if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
          Relation r = arrays2relation(loop, ir, freevar, a, IS1, b, IS2,uninterpreted_symbols,uninterpreted_symbols_stringrepr, unin_rel);
          r.simplify();
          dep_relation.push_back(copy(r));

          std::pair<std::vector<DependenceVector>,
              std::vector<DependenceVector> > dv =
              relation2dependences(a, b, r);

          // Manu:: check if the array references belong to the same statement
          // If yes, set the flag in the dependence vector
          //----------------------------------------------
          if(DEP_DEBUG){
            std::cout << "Size of the dependence vector '" << a->name().c_str() << "'  -- " << dv.first.size() << "\n";
            std::cout << "------------ Printing dependence vector START ---------------\n";

            for (std::vector<DependenceVector>::iterator itd = dv.first.begin(); itd != dv.first.end(); itd++){
              if (itd->type == DEP_R2W)
                std::cout<<"WAR\n";
              else if (itd->type == DEP_W2R)
                std::cout<<"RAW\n";
              else if (itd->type == DEP_W2W)
                std::cout<<"WAW\n";

              std::vector<omega::coef_t>::iterator itu = itd->ubounds.begin();
              for (std::vector<omega::coef_t>::iterator itl = itd->lbounds.begin(); itl != itd->lbounds.end(); itl++){
                std::cout << "(" << *itl << ", " << *itu << ")\n";
                itu++;
              }
            }
            std::cout << "--------\n";
            for (std::vector<DependenceVector>::iterator itd = dv.second.begin(); itd != dv.second.end(); itd++){
              if (itd->type == DEP_R2W)
                std::cout<<"WAR\n";
              else if (itd->type == DEP_W2R)
                std::cout<<"RAW\n";
              else if (itd->type == DEP_W2W)
                std::cout<<"WAW\n";

              std::vector<omega::coef_t>::iterator itu = itd->ubounds.begin();
              for (std::vector<omega::coef_t>::iterator itl = itd->lbounds.begin(); itl != itd->lbounds.end(); itl++){
                std::cout << "(" << *itl << ", " << *itu << ")\n";
                itu++;
              }
            }
            std::cout << "------------ Printing dependence vector END---------------\n";
          }
          checkReductionDependence(i,j,nestLeveli,lbound,ubound,ref2Stmt,rMap,dv,trMap,nrStmts);
          //----------------------------------------------

//					 // Manu:: original code without the condition
          if (((rMap.find(ref2Stmt[i])->second).size() != 3) || (lbound[0] != lbound[1]) || (lbound[1] != lbound[2]) ||
              (lbound[0] != lbound[2]) || (ubound[0] != ubound[1]) || (ubound[1] != ubound[2]) || (ubound[0] != ubound[2])) { // Manu:: original code without the condition
            result.first.insert(result.first.end(),
                                dv.first.begin(), dv.first.end());
            result.second.insert(result.second.end(),
                                 dv.second.begin(), dv.second.end());
          }
        }
        delete sym_b;
      }
      delete sym_a;
    }

    // Manu
    for (ittrMap = trMap.begin(); ittrMap != trMap.end(); ittrMap++) {
      DVPair tdv = ittrMap->second;
      result.first.insert(result.first.end(), tdv.first.begin(),
                          tdv.first.end());
      result.second.insert(result.second.end(), tdv.second.begin(),
                           tdv.second.end());
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
          Relation r = arrays2relation(loop, ir, freevar, a, IS1, b, IS2,uninterpreted_symbols,uninterpreted_symbols_stringrepr, unin_rel);
          dep_relation.push_back(copy(r));
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

  return result;
}


//Manu:: This function tests if two references are from the same statement
//CG_outputRepr * from_same_statement(IR_Code *ir, IR_ArrayRef *a, IR_ArrayRef *b) {
bool from_same_statement(IR_Code *ir, IR_ArrayRef *a, IR_ArrayRef *b) {
  return ir->FromSameStmt(a,b);
}

// Manu
int stmtType(IR_Code *ir, const CG_outputRepr *repr) {
  auto rep = static_cast<const CG_chillRepr*>(repr);
  return (rep->getChillCode().size() == 0 && rep->getChillCode()[0]->isBinaryOperator()) ? 1 : 0;
}

// Manu:: set the reduction operation
IR_OPERATION_TYPE getReductionOperator(IR_Code *ir, const CG_outputRepr *repr) {
  return (ir->getReductionOp(repr));
}

// Manu:: map references to its corresponding statements
void mapRefstoStatements(IR_Code *ir, std::vector<IR_ArrayRef *> access, int ref2Stmt[], std::map<int,std::set<int> >& rMap, std::set<int>& tnrStmts, std::set<int>& nrStmts) {
  
  int stmtId = 1;
  for (int i = 0; i < access.size(); i++) {
    IR_ArrayRef *a = access[i];
    IR_ArraySymbol *sym_a = a->symbol();
    for (int j = i; j < access.size(); j++) {
      IR_ArrayRef *b = access[j];
      IR_ArraySymbol *sym_b = b->symbol();
      bool inSameStmt;
      if (from_same_statement(ir,access[i],access[j])) {
        inSameStmt = true;
//            std::cout << "Manu:: inSameStmt " << a->name().c_str() << ", " << b->name().c_str() << "\n";
      } else {
        inSameStmt = false;
//            std::cout << "Manu:: NOT inSameStmt " << a->name().c_str() << ", " << b->name().c_str() << "\n";
      }
      if (inSameStmt) {
        if (ref2Stmt[i] == -1)
          ref2Stmt[i] = stmtId++;
        ref2Stmt[j] = ref2Stmt[i];
        rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],std::set<int>()));
      } else {
        if (ref2Stmt[i] == -1)
          ref2Stmt[i] = stmtId++;
        if (ref2Stmt[j] == -1)
          ref2Stmt[j] = stmtId++;
        if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
          tnrStmts.insert(i);
          tnrStmts.insert(j);
        }
      }
      
    }
  }
  std::set<int>::iterator itS;
  for (itS = tnrStmts.begin(); itS != tnrStmts.end(); itS++) {
    nrStmts.insert(ref2Stmt[*itS]);
  }
  
}

// Manu:: This function tests reduction dependence and updates corresponding data structures
void checkReductionDependence(int i, int j, int nestLeveli, omega::coef_t lbound[], omega::coef_t ubound[], int ref2Stmt[], std::map<int,std::set<int> >& rMap, DVPair& dv, tempResultMap& trMap, std::set<int> nrStmts ) {
  
  std::map<int, std::set<int> >::iterator itMap;
  tempResultMap::iterator ittrMap;
  bool raw,war,waw, flg;
  raw = war = waw = flg = false;
  if ((ref2Stmt[i] == ref2Stmt[j]) && (nrStmts.find(ref2Stmt[i])== nrStmts.end())) {
    for (int k = 0; k < dv.first.size(); k++) {
      if ((dv.first[k].lbounds[nestLeveli-1] == 0) && (dv.first[k].ubounds[nestLeveli-1] == 0))
        continue;
      itMap = rMap.find(ref2Stmt[i]);
      if (dv.first[k].type == DEP_R2W) {
        war = true;
        std::set<int> s = itMap->second;
        s.insert(1); // war == 1
        rMap.erase(itMap);
        rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
        if (lbound[0] > dv.first[k].lbounds[nestLeveli-1])
          lbound[0] = dv.first[k].lbounds[nestLeveli-1];
        if(ubound[0] < dv.first[k].ubounds[nestLeveli-1])
          ubound[0] = dv.first[k].ubounds[nestLeveli-1];
      } else if (dv.first[k].type == DEP_W2R) {
        //    for (int k1 = 0; k1 < dv.first[k].lbounds.size(); k1++) {
        //      omega::coef_t lbound = dv.first[k].lbounds[k1];
        omega::coef_t lbound1 = dv.first[k].lbounds[nestLeveli-1];
        if (lbound1 > 0) {
          flg = true;
          //        break;
        }
        //    }
        raw = true;
        if (raw) {
          std::set<int> s = itMap->second;
          s.insert(2); // raw == 2
          rMap.erase(itMap);
          rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
          if (lbound[1] > dv.first[k].lbounds[nestLeveli-1])
            lbound[1] = dv.first[k].lbounds[nestLeveli-1];
          if(ubound[1] < dv.first[k].ubounds[nestLeveli-1])
            ubound[1] = dv.first[k].ubounds[nestLeveli-1];
        }
      } else if (dv.first[k].type == DEP_W2W) {
        waw = true;
        std::set<int> s = itMap->second;
        s.insert(3); // waw == 3
        rMap.erase(itMap);
        rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
        if (lbound[2] > dv.first[k].lbounds[nestLeveli-1])
          lbound[2] = dv.first[k].lbounds[nestLeveli-1];
        if(ubound[2] < dv.first[k].ubounds[nestLeveli-1])
          ubound[2] = dv.first[k].ubounds[nestLeveli-1];
      }
//              std::cout<< "Manu:: Flags:: " << "raw " << raw << ", war " << war << ", waw " << waw << "\n";
    }
    flg = false;
    for (int k = 0; k < dv.second.size(); k++) {
      if ((dv.second[k].lbounds[nestLeveli-1] == 0) && (dv.second[k].ubounds[nestLeveli-1] == 0))
        continue;
      itMap = rMap.find(ref2Stmt[i]);
      if (dv.second[k].type == DEP_R2W) {
        war = true;
        std::set<int> s = itMap->second;
        s.insert(1); // war == 1
        rMap.erase(itMap);
        rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
        if (lbound[0] > dv.second[k].lbounds[nestLeveli-1])
          lbound[0] = dv.second[k].lbounds[nestLeveli-1];
        if (ubound[0] < dv.second[k].ubounds[nestLeveli-1])
          ubound[0] = dv.second[k].ubounds[nestLeveli-1];
        
      } else if (dv.second[k].type == DEP_W2R) {
        //    for (int k1 = 0; k1 < dv.second[k].lbounds.size(); k1++) {
        //omega::coef_t lbound = dv.second[k].lbounds[k1];
        omega::coef_t lbound1 = dv.second[k].lbounds[nestLeveli-1];
        if (lbound1 > 0) {
          flg = true;
          //        break;
        }
        //    }
        raw = true;
        if (raw) {
          std::set<int> s = itMap->second;
          s.insert(2); // raw == 2
          rMap.erase(itMap);
          rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
          if (lbound[1] > dv.second[k].lbounds[nestLeveli-1])
            lbound[1] = dv.second[k].lbounds[nestLeveli-1];
          if (ubound[1] < dv.second[k].ubounds[nestLeveli-1])
            ubound[1] = dv.second[k].ubounds[nestLeveli-1];
          
        }
        
      } else if (dv.second[k].type == DEP_W2W) {
        waw = true;
        std::set<int> s = itMap->second;
        s.insert(3); // waw == 3
        rMap.erase(itMap);
        rMap.insert(std::pair<int,std::set<int> >(ref2Stmt[i],s));
        if (lbound[2] > dv.second[k].lbounds[nestLeveli-1])
          lbound[2] = dv.second[k].lbounds[nestLeveli-1];
        if (ubound[2] < dv.second[k].ubounds[nestLeveli-1])
          ubound[2] = dv.second[k].ubounds[nestLeveli-1];
        
      }
//              std::cout<< "Manu:: Flags:: " << "raw " << raw << ", war " << war << ", waw " << waw << "\n";
    }
    
//            if ((rMap.find(ref2Stmt[i])->second).size() == 3) {
    if(DEP_DEBUG){
      std::cout << "lbounds: " << lbound[0] << ", " << lbound[1] << ", " <<lbound[2] << "\n";
      std::cout << "ubounds: " << ubound[0] << ", " << ubound[1] << ", " <<ubound[2] << "\n";
    }
    if (((rMap.find(ref2Stmt[i])->second).size() == 3) && (lbound[0] == lbound[1]) && (lbound[1] == lbound[2])
        && (ubound[0] == ubound[1]) && (ubound[1] == ubound[2])) {
//              std::cout << "Manu:: All dependences present 1 \n";
      for (int k = 0; k < dv.second.size(); k++)
        dv.second[k].is_reduction_cand = true;
      for (int k = 0; k < dv.first.size(); k++)
        dv.first[k].is_reduction_cand = true;
      trMap.insert(std::pair<int,DVPair>(ref2Stmt[i],DVPair(dv.first,dv.second)));
    }
  } else {
    //  tempArrayRefId[i] = tempArrayRefId[j] = 0;
    for (int k = 0; k < dv.second.size(); k++)
      dv.second[k].is_reduction_cand = false;
    for (int k = 0; k < dv.first.size(); k++)
      dv.first[k].is_reduction_cand = false;
//            reductionCand = false;
    ittrMap = trMap.find(ref2Stmt[i]);
    if (ittrMap != trMap.end()) {
      DVPair tdv = ittrMap->second;
      for (int k = 0; k < (tdv.first).size(); k++)
        tdv.first[k].is_reduction_cand = false;
      for (int k = 0; k < (tdv.second).size(); k++)
        tdv.second[k].is_reduction_cand = false;
      trMap.erase(ittrMap);
      trMap.insert(std::pair<int,DVPair>(ref2Stmt[i],DVPair(tdv.first,tdv.second)));
    }
  }
  
}



void print_control(  IR_Control *con ) {
  IR_CONTROL_TYPE type = con->type();
  debug_fprintf(stderr, "this is IR_Control of type %s\n", chill_ir_control_type_string( type ).c_str());

  switch (type) {
  case IR_CONTROL_BLOCK:  
  case IR_CONTROL_LOOP:   
  case IR_CONTROL_IF:     
  case IR_CONTROL_WHILE:  
  default:  return; 
  }

}
