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

#include <suif1.h>
#include "ir_suif_utils.hh"


/**
 * Returns the body of the for loop found by finding the first loop in
 * code, and if level > 1 recursively calling on the body of the found
 * loop and (level-1)
 */
tree_node_list* loop_body_at_level(tree_node_list* tnl, int level)
{
  tree_node_list *inner_nl = 0;
  //Now strip out the tnl on the inner level of the for loop
  tree_node_list_iter tnli(tnl);
  while (!tnli.is_empty()) {
    tree_node *node = tnli.step();
    if(node->kind() == TREE_FOR)
    {
      //Found the first tree_for, call sibling function
      inner_nl = loop_body_at_level((tree_for*)node, level);
      break;
    }
  }
  return inner_nl;
}

tree_node_list* loop_body_at_level(tree_for* loop, int level)
{
  if(level > 1)
    return loop_body_at_level(loop->body(), level-1);
  return loop->body();
}

tree_node_list*  swap_node_for_node_list(tree_node* tn, tree_node_list* new_tnl)
{
  tree_node_list* tnl  = tn->parent();
  tnl->insert_after(new_tnl, tn->list_e());
  delete tnl->remove(tn->list_e());
  return tnl;
}
