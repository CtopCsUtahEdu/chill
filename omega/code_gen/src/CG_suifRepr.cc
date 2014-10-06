/*****************************************************************************
 Copyright (C) 2008 University of Southern California. 
 All Rights Reserved.

 Purpose:
   omega holder for suif implementaion

 Notes:

 History:
   02/01/06 - Chun Chen - created
*****************************************************************************/

#include <code_gen/CG_suifRepr.h>
#include <stdio.h>

namespace omega {

CG_suifRepr::CG_suifRepr(): tnl_(NULL), op_() {
}

CG_suifRepr::CG_suifRepr(tree_node_list *tnl): tnl_(tnl),op_() {
}

CG_suifRepr::CG_suifRepr(operand op): tnl_(NULL), op_(op) {
}
  
CG_suifRepr::~CG_suifRepr() {
  // delete nothing here. operand or tree_node_list should already be
  // grafted to other expression tree or statement list
}

CG_outputRepr* CG_suifRepr::clone() {
  if (!op_.is_null() ) {
    operand op = op_.clone();
    return new CG_suifRepr(op);
  }
  else if (tnl_ != NULL) {
    tree_node_list *tnl = tnl_->clone();
    return new CG_suifRepr(tnl);
  }
  else
    return new CG_suifRepr();
}

void CG_suifRepr::clear() {
  if (!op_.is_null()) {
    if (op_.is_instr())
      delete op_.instr();
    op_.set_null();
  }
  else if (tnl_ != NULL) {
    delete tnl_;
    tnl_ = NULL;
  }
}

tree_node_list* CG_suifRepr::GetCode() const {
  return tnl_;
}

operand CG_suifRepr::GetExpression() const {
  return op_;
} 

void CG_suifRepr::Dump() const {
  if (tnl_ != NULL)
    tnl_->print();
  else if (!op_.is_null())
    op_.print();
}

void CG_suifRepr::DumpToFile(FILE *fp) const {
  if (tnl_ != NULL)
    tnl_->print(fp);
  else if (!op_.is_null())
    op_.print(fp);
}


} // namespace
