/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   SUIF interface utilities.

 Notes:

 Update history:
   01/2006 created by Chun Chen
*****************************************************************************/

//#include <suif1.h>
//#include <useful.h>
//#include <vector>
//#include <algorithm>
//#include <code_gen/CG_suifRepr.h>
#include "ir_rose_utils.hh"



std::vector<SgForStatement *> find_loops(SgNode *tnl) {
  std::vector<SgForStatement *> result;
  
  //tree_node_list_iter iter(tnl);
  
  /*while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    if (tn->kind() == TREE_FOR)
    result.push_back(static_cast<tree_for *>(tn));
    }
  */
  
  SgStatementPtrList& blockStatements = isSgBasicBlock(tnl)->get_statements();
  for(SgStatementPtrList::const_iterator j = blockStatements.begin(); j != blockStatements.end(); j++)
    if(isSgForStatement(*j))
      result.push_back(isSgForStatement(*j));
  
  return result;
}

std::vector<SgForStatement *> find_deepest_loops(SgStatementPtrList& tnl) {
  
  std::vector<SgForStatement *> loops;
  
  
  
  for(SgStatementPtrList::const_iterator j = tnl.begin(); j != tnl.end(); j++)
  {
    std::vector<SgForStatement *> t = find_deepest_loops(isSgNode(*j));
    if (t.size() > loops.size())
      loops = t;
  }       
  
  
  
  return loops;
  
}








std::vector<SgForStatement *> find_deepest_loops(SgNode *tn) {
  if (isSgForStatement(tn)) {
    std::vector<SgForStatement *> loops;
    
    SgForStatement *tnf = static_cast<SgForStatement*>(tn);
    loops.insert(loops.end(), tnf);
    std::vector<SgForStatement*> t = find_deepest_loops(isSgNode(tnf->get_loop_body()));
    std::copy(t.begin(), t.end(), std::back_inserter(loops));
    
    return loops;
  }
  else if (isSgBasicBlock(tn)) {
    SgBasicBlock *tnb = static_cast<SgBasicBlock*>(tn);
    return find_deepest_loops(tnb->get_statements());
  }
  else 
    return std::vector<SgForStatement *>();               
}

