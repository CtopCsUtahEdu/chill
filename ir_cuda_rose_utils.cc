/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   ROASE interface utilities.   TODO remove all Sg references 

 Notes:

 Update history:
   01/2006 created by Chun Chen
*****************************************************************************/


#ifdef FRONTEND_ROSE 
#include "ir_rose_utils.hh"


/**
 * Returns the body of the for loop found by finding the first loop in
 * code, and if level > 1 recursively calling on the body of the found
 * loop and (level-1)
 */
SgNode* loop_body_at_level(SgNode* tnl, int level) {
  SgNode *inner_nl = 0;
  //Now strip out the tnl on the inner level of the for loop
  //tree_node_list_iter tnli(tnl);
  
  if (isSgBasicBlock(tnl)) {
    
    SgStatementPtrList& tnli = isSgBasicBlock(tnl)->get_statements();
    
    for (SgStatementPtrList::iterator it = tnli.begin(); it != tnli.end();
         it++) {
      if (isSgForStatement(*it)) {
        inner_nl = loop_body_at_level(isSgForStatement(*it), level);
        break;
      }
      
    }
    
  }
  
  return inner_nl;
}

SgNode* loop_body_at_level(SgForStatement* loop, int level) {
  if (level > 1)
    return loop_body_at_level(loop->get_loop_body(), level - 1);
  return loop->get_loop_body();
}

void swap_node_for_node_list(SgNode* tn, SgNode* new_tnl) {
  SgStatement *s = isSgStatement(tn);
  
  SgStatement* p;
  if (s != 0) {
    p = isSgStatement(tn->get_parent());
    
    if (p != 0) {
      
      if (isSgBasicBlock(new_tnl)) {
        
        /*SgStatementPtrList & list_ =
          isSgBasicBlock(new_tnl)->get_statements();
          
          if (isSgForStatement(p)) {
          if (!isSgBasicBlock(isSgForStatement(p)->get_loop_body()))
          p->replace_statement(s, isSgStatement(new_tnl));
          else {
          p->insert_statement(s, list_, true);
          p->remove(s);
          }
          } else {
          p->insert_statement(s, list_, true);
          p->remove(s);
          }
        */
        if (isSgForStatement(p)) {
          if (!isSgBasicBlock(isSgForStatement(p)->get_loop_body()))
            p->replace_statement(s, isSgStatement(new_tnl));
          else {
            
            SgStatementPtrList& list_ =
              isSgBasicBlock(new_tnl)->get_statements();
            
            //std::vector<SgStatement*> list;
            
            SgStatementPtrList::iterator it = list_.begin();
            SgStatement* begin = *it;
            begin->set_parent(p);
            
            p->replace_statement(s, begin);
            it++;
            //SgStatement* stmt = first;
            SgStatement* temp = begin;
            for (; it != list_.end(); it++) {
              (*it)->set_parent(p);
              p->insert_statement(temp, *it, false);
              temp = *it;
            }
            
          }
          
        } else {
          
          
          SgStatementPtrList& list_ =
            isSgBasicBlock(new_tnl)->get_statements();
          
          //std::vector<SgStatement*> list;
          
          SgStatementPtrList::iterator it = list_.begin();
          SgStatement* begin = *it;
          begin->set_parent(p);
          
          p->replace_statement(s, begin);
          it++;
          //SgStatement* stmt = first;
          SgStatement* temp = begin;
          for (; it != list_.end(); it++) {
            (*it)->set_parent(p);
            p->insert_statement(temp, *it, false);
            temp = *it;
          }
          
        }
        
        /*  SgStatement* temp = s;
            
            SgStatementPtrList::iterator it = list_.begin();
            p->insert_statement(temp, *it, true);
            temp = *it;
            p->remove_statement(s);
            it++;
            for (; it != list_.end(); it++) {
            p->insert_statement(temp, *it, false);
            temp = *it;
            }
            
            // new_tnl->set_parent(p);
            //new_tnl->get_statements();
            SgStatementPtrList& list =
            isSgBasicBlock(new_tnl)->get_statements();
            
            //std::vector<SgStatement*> list;
            
            SgStatementPtrList::iterator it = list.begin();
            SgStatement* begin = *it;
            begin->set_parent(p);
            
            p->replace_statement(s, begin);
            it++;
            //SgStatement* stmt = first;
            SgStatement* temp = begin;
            for (; it != list.end(); it++) {
            (*it)->set_parent(p);
            p->insert_statement(temp, *it, false);
            temp = *it;
            }
        */
        /*              SgStatementPtrList& stmt_list = isSgBasicBlock(new_tnl)->get_statements();
                        SgStatement* target =   s;
                        
                        for(SgStatementPtrList::iterator it = stmt_list.begin() ; it != stmt_list.end(); it++)
                        {
                        isSgNode(*it)->set_parent(p);
                        p->insert_statement(isSgStateme, *it, false);
                        target = *it;
                        }
                        
                        p->remove_statement(s);
                        
        */
      }else if(isSgIfStmt(p)) {
        
        if(isSgIfStmt(p)->get_true_body() == s)
          isSgIfStmt(p)->set_true_body(isSgStatement(new_tnl));
        else if(isSgIfStmt(p)->get_false_body() == s)
          isSgIfStmt(p)->set_false_body(isSgStatement(new_tnl));
        new_tnl->set_parent(p);
      } 
      else {
        p->replace_statement(s, isSgStatement(new_tnl));
        new_tnl->set_parent(p);
      }
    }
    
  }
  //    return isSgNode(p);
}

#endif
