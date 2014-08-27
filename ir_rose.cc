/*****************************************************************************
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
 CHiLL's rose interface.

 Notes:
 Array supports mixed pointer and array type in a single declaration.

 History:
 02/23/2009 Created by Chun Chen.
*****************************************************************************/
#include <string>
#include "ir_rose.hh"
#include "ir_rose_utils.hh"
#include <code_gen/rose_attributes.h>
#include <code_gen/CG_roseRepr.h>
#include <code_gen/CG_roseBuilder.h>

using namespace SageBuilder;
using namespace SageInterface;
using namespace omega;
// ----------------------------------------------------------------------------
// Class: IR_roseScalarSymbol
// ----------------------------------------------------------------------------

std::string IR_roseScalarSymbol::name() const {
  return vs_->get_name().getString();
}

int IR_roseScalarSymbol::size() const {
  return (vs_->get_type()->memoryUsage()) / (vs_->get_type()->numberOfNodes());
}

bool IR_roseScalarSymbol::operator==(const IR_Symbol &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseScalarSymbol *l_that =
    static_cast<const IR_roseScalarSymbol *>(&that);
  return this->vs_ == l_that->vs_;
}

IR_Symbol *IR_roseScalarSymbol::clone() const {
  return NULL;
}

// ----------------------------------------------------------------------------
// Class: IR_roseArraySymbol
// ----------------------------------------------------------------------------

std::string IR_roseArraySymbol::name() const {
  return (vs_->get_declaration()->get_name().getString());
}

int IR_roseArraySymbol::elem_size() const {
  
  SgType *tn = vs_->get_type();
  SgType* arrType;
  
  int elemsize;
  
  if (arrType = isSgArrayType(tn)) {
    while (isSgArrayType(arrType)) {
      arrType = arrType->findBaseType();
    }
  } else if (arrType = isSgPointerType(tn)) {
    while (isSgPointerType(arrType)) {
      arrType = arrType->findBaseType();
    }
  }
  
  elemsize = (int) arrType->memoryUsage() / arrType->numberOfNodes();
  return elemsize;
}

int IR_roseArraySymbol::n_dim() const {
  int dim = 0;
  SgType* arrType = isSgArrayType(vs_->get_type());
  SgType* ptrType = isSgPointerType(vs_->get_type());
  if (arrType != NULL) {
    while (isSgArrayType(arrType)) {
      arrType = isSgArrayType(arrType)->get_base_type();
      dim++;
    }
  } else if (ptrType != NULL) {
    while (isSgPointerType(ptrType)) {
      ptrType = isSgPointerType(ptrType)->get_base_type();
      dim++;
    }
  }

  // Manu:: fortran support
  if (static_cast<const IR_roseCode *>(ir_)->is_fortran_) {

	  if (arrType != NULL) {
		  dim = 0;
		  SgExprListExp * dimList = isSgArrayType(vs_->get_type())->get_dim_info();
		  SgExpressionPtrList::iterator it = dimList->get_expressions().begin();
		  for(;it != dimList->get_expressions().end(); it++) {
		    dim++;
		  }
	  } else if (ptrType != NULL) {
		  //std::cout << "pntrType \n";
		  ; // not sure if this case will happen
	  }
  }

  return dim;
}

omega::CG_outputRepr *IR_roseArraySymbol::size(int dim) const {
  
  SgArrayType* arrType = isSgArrayType(vs_->get_type());
  // SgExprListExp* dimList = arrType->get_dim_info();
  int count = 0;
  SgExpression* expr;
  SgType* pntrType = isSgPointerType(vs_->get_type());
  
  if (arrType != NULL) {
    SgExprListExp* dimList = arrType->get_dim_info();
    if (!static_cast<const IR_roseCode *>(ir_)->is_fortran_) {
      SgExpressionPtrList::iterator it =
        dimList->get_expressions().begin();
      
      while ((it != dimList->get_expressions().end()) && (count < dim)) {
        it++;
        count++;
      }
      
      expr = *it;
    } else {
      SgExpressionPtrList::reverse_iterator i =
        dimList->get_expressions().rbegin();
      for (; (i != dimList->get_expressions().rend()) && (count < dim);
           i++) {
        
        count++;
      }
      
      expr = *i;
    }
  } else if (pntrType != NULL) {
    
    while (count < dim) {
      pntrType = (isSgPointerType(pntrType))->get_base_type();
      count++;
    }
    if (isSgPointerType(pntrType))
      expr = new SgExpression;
  }
  
  if (!expr)
    throw ir_error("Index variable is NULL!!");
  
  // Manu :: debug
  std::cout << "---------- size :: " << isSgNode(expr)->unparseToString().c_str() << "\n";

  return new omega::CG_roseRepr(expr);
  
}

IR_ARRAY_LAYOUT_TYPE IR_roseArraySymbol::layout_type() const {
  if (static_cast<const IR_roseCode *>(ir_)->is_fortran_)
    return IR_ARRAY_LAYOUT_COLUMN_MAJOR;
  else
    return IR_ARRAY_LAYOUT_ROW_MAJOR;
  
}

bool IR_roseArraySymbol::operator==(const IR_Symbol &that) const {
  
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseArraySymbol *l_that =
    static_cast<const IR_roseArraySymbol *>(&that);
  return this->vs_ == l_that->vs_;
  
}

IR_Symbol *IR_roseArraySymbol::clone() const {
  return new IR_roseArraySymbol(ir_, vs_);
}

// ----------------------------------------------------------------------------
// Class: IR_roseConstantRef
// ----------------------------------------------------------------------------

bool IR_roseConstantRef::operator==(const IR_Ref &that) const {
  
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseConstantRef *l_that =
    static_cast<const IR_roseConstantRef *>(&that);
  
  if (this->type_ != l_that->type_)
    return false;
  
  if (this->type_ == IR_CONSTANT_INT)
    return this->i_ == l_that->i_;
  else
    return this->f_ == l_that->f_;
  
}

omega::CG_outputRepr *IR_roseConstantRef::convert() {
  if (type_ == IR_CONSTANT_INT) {
    omega::CG_roseRepr *result = new omega::CG_roseRepr(
      isSgExpression(buildIntVal(static_cast<int>(i_))));
    delete this;
    return result;
  } else
    throw ir_error("constant type not supported");
  
}

IR_Ref *IR_roseConstantRef::clone() const {
  if (type_ == IR_CONSTANT_INT)
    return new IR_roseConstantRef(ir_, i_);
  else if (type_ == IR_CONSTANT_FLOAT)
    return new IR_roseConstantRef(ir_, f_);
  else
    throw ir_error("constant type not supported");
  
}

// ----------------------------------------------------------------------------
// Class: IR_roseScalarRef
// ----------------------------------------------------------------------------

bool IR_roseScalarRef::is_write() const {
  /*    if (ins_pos_ != NULL && op_pos_ == -1)
        return true;
        else
        return false;
  */
  
  if (is_write_ == 1)
    return true;
  
  return false;
}

IR_ScalarSymbol *IR_roseScalarRef::symbol() const {
  return new IR_roseScalarSymbol(ir_, vs_->get_symbol());
}

bool IR_roseScalarRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseScalarRef *l_that =
    static_cast<const IR_roseScalarRef *>(&that);
  
  if (this->ins_pos_ == NULL)
    return this->vs_ == l_that->vs_;
  else
    return this->ins_pos_ == l_that->ins_pos_
      && this->op_pos_ == l_that->op_pos_;
}

omega::CG_outputRepr *IR_roseScalarRef::convert() {
  omega::CG_roseRepr *result = new omega::CG_roseRepr(isSgExpression(vs_));
  delete this;
  return result;
  
}

IR_Ref * IR_roseScalarRef::clone() const {
  //if (ins_pos_ == NULL)
  return new IR_roseScalarRef(ir_, vs_, this->is_write_);
  //else
  //        return new IR_roseScalarRef(ir_, , op_pos_);
  
}

// ----------------------------------------------------------------------------
// Class: IR_roseArrayRef
// ----------------------------------------------------------------------------

bool IR_roseArrayRef::is_write() const {
  SgAssignOp* assignment;
  
  if (is_write_ == 1 || is_write_ == 0)
    return is_write_;
  if (assignment = isSgAssignOp(ia_->get_parent())) {
    if (assignment->get_lhs_operand() == ia_)
      return true;
  } else if (SgExprStatement* expr_stmt = isSgExprStatement(
               ia_->get_parent())) {
    SgExpression* exp = expr_stmt->get_expression();
    
    if (exp) {
      if (assignment = isSgAssignOp(exp)) {
        if (assignment->get_lhs_operand() == ia_)
          return true;
        
      }
    }
    
  }
  return false;
}

omega::CG_outputRepr *IR_roseArrayRef::index(int dim) const {
  
  SgExpression *current = isSgExpression(ia_);
  SgExpression* expr;
  int count = 0;
  
  while (isSgPntrArrRefExp(current)) {
    current = isSgPntrArrRefExp(current)->get_lhs_operand();
    count++;
  }
  
  current = ia_;
  
  while (count > dim) {
    expr = isSgPntrArrRefExp(current)->get_rhs_operand();
    current = isSgPntrArrRefExp(current)->get_lhs_operand();
    count--;
  }

  // Manu:: fortran support
  if (static_cast<const IR_roseCode *>(ir_)->is_fortran_) {
	  expr = isSgPntrArrRefExp(ia_)->get_rhs_operand();
	  count = 0;
	  if (isSgExprListExp(expr)) {
		  SgExpressionPtrList::iterator indexList = isSgExprListExp(expr)->get_expressions().begin();
		  while (count < dim) {
			  indexList++;
			  count++;
		  }
		  expr = isSgExpression(*indexList);
	  }
  }

  if (!expr)
    throw ir_error("Index variable is NULL!!");


  omega::CG_roseRepr* ind = new omega::CG_roseRepr(expr);
  
  return ind->clone();
  
}

IR_ArraySymbol *IR_roseArrayRef::symbol() const {
  
  SgExpression *current = isSgExpression(ia_);
  
  SgVarRefExp* base;
  SgVariableSymbol *arrSymbol;
  while (isSgPntrArrRefExp(current) || isSgUnaryOp(current)) {
    if (isSgPntrArrRefExp(current))
      current = isSgPntrArrRefExp(current)->get_lhs_operand();
    else if (isSgUnaryOp(current))
      /* To handle support for addressof operator and pointer dereference
       * both of which are unary ops
       */
      current = isSgUnaryOp(current)->get_operand();
  }
  if (base = isSgVarRefExp(current)) {
    arrSymbol = (SgVariableSymbol*) (base->get_symbol());
    std::string x = arrSymbol->get_name().getString();
  } else
    throw ir_error("Array Symbol is not a variable?!");
  
  return new IR_roseArraySymbol(ir_, arrSymbol);
  
}

bool IR_roseArrayRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseArrayRef *l_that = static_cast<const IR_roseArrayRef *>(&that);
  
  return this->ia_ == l_that->ia_;
}

omega::CG_outputRepr *IR_roseArrayRef::convert() {
  omega::CG_roseRepr *temp = new omega::CG_roseRepr(
    isSgExpression(this->ia_));
  omega::CG_outputRepr *result = temp->clone();
//  delete this;   // Commented by Manu
  return result;
}

IR_Ref *IR_roseArrayRef::clone() const {
  return new IR_roseArrayRef(ir_, ia_, is_write_);
}

// ----------------------------------------------------------------------------
// Class: IR_roseLoop
// ----------------------------------------------------------------------------

IR_ScalarSymbol *IR_roseLoop::index() const {
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  SgVariableSymbol* vs = NULL;
  if (tf) {
    SgForInitStatement* list = tf->get_for_init_stmt();
    SgStatementPtrList& initStatements = list->get_init_stmt();
    SgStatementPtrList::const_iterator j = initStatements.begin();
    
    if (SgExprStatement *expr = isSgExprStatement(*j))
      if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
        if (SgVarRefExp* var_ref = isSgVarRefExp(op->get_lhs_operand()))
          vs = var_ref->get_symbol();
  } else if (tfortran) {
    SgExpression* init = tfortran->get_initialization();
    
    if (SgAssignOp* op = isSgAssignOp(init))
      if (SgVarRefExp* var_ref = isSgVarRefExp(op->get_lhs_operand()))
        vs = var_ref->get_symbol();
    
  }
  
  if (vs == NULL)
    throw ir_error("Index variable is NULL!!");
  
  return new IR_roseScalarSymbol(ir_, vs);
}

omega::CG_outputRepr *IR_roseLoop::lower_bound() const {
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  
  SgExpression* lowerBound = NULL;
  
  if (tf) {
    SgForInitStatement* list = tf->get_for_init_stmt();
    SgStatementPtrList& initStatements = list->get_init_stmt();
    SgStatementPtrList::const_iterator j = initStatements.begin();
    
    if (SgExprStatement *expr = isSgExprStatement(*j))
      if (SgAssignOp* op = isSgAssignOp(expr->get_expression())) {
        lowerBound = op->get_rhs_operand();
        //Rose sometimes introduces an unnecessary cast which is a unary op
        if (isSgUnaryOp(lowerBound))
          lowerBound = isSgUnaryOp(lowerBound)->get_operand();
        
      }
  } else if (tfortran) {
    SgExpression* init = tfortran->get_initialization();
    
    if (SgAssignOp* op = isSgAssignOp(init))
      lowerBound = op->get_rhs_operand();
  }
  
  if (lowerBound == NULL)
    throw ir_error("Lower Bound is NULL!!");
  
  return new omega::CG_roseRepr(lowerBound);
}

omega::CG_outputRepr *IR_roseLoop::upper_bound() const {
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  SgExpression* upperBound = NULL;
  if (tf) {
    SgBinaryOp* test_expr = isSgBinaryOp(tf->get_test_expr());
    if (test_expr == NULL)
      throw ir_error("Test Expression is NULL!!");
    
    upperBound = test_expr->get_rhs_operand();
    //Rose sometimes introduces an unnecessary cast which is a unary op
    if (isSgUnaryOp(upperBound))
      upperBound = isSgUnaryOp(upperBound)->get_operand();
    if (upperBound == NULL)
      throw ir_error("Upper Bound is NULL!!");
  } else if (tfortran) {
    
    upperBound = tfortran->get_bound();
    
  }
  
  return new omega::CG_roseRepr(upperBound);
  
}

IR_CONDITION_TYPE IR_roseLoop::stop_cond() const {
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  
  if (tf) {
    SgExpression* stopCond = NULL;
    SgExpression* test_expr = tf->get_test_expr();
    
    if (isSgLessThanOp(test_expr))
      return IR_COND_LT;
    else if (isSgLessOrEqualOp(test_expr))
      return IR_COND_LE;
    else if (isSgGreaterThanOp(test_expr))
      return IR_COND_GT;
    else if (isSgGreaterOrEqualOp(test_expr))
      return IR_COND_GE;
    
    else
      throw ir_error("loop stop condition unsupported");
  } else if (tfortran) {
    SgExpression* increment = tfortran->get_increment();
    if (!isSgNullExpression(increment)) {
      if (isSgMinusOp(increment)
          && !isSgBinaryOp(isSgMinusOp(increment)->get_operand()))
        return IR_COND_GE;
      else
        return IR_COND_LE;
    } else {
    	return IR_COND_LE; // Manu:: if increment is not present, assume it to be 1. Just a workaround, not sure if it will be correct for all cases.
      SgExpression* lowerBound = NULL;
      SgExpression* upperBound = NULL;
      SgExpression* init = tfortran->get_initialization();
      SgIntVal* ub;
      SgIntVal* lb;
      if (SgAssignOp* op = isSgAssignOp(init))
        lowerBound = op->get_rhs_operand();
      
      upperBound = tfortran->get_bound();
      
      if ((upperBound != NULL) && (lowerBound != NULL)) {
        
        if ((ub = isSgIntVal(isSgValueExp(upperBound))) && (lb =
                                                            isSgIntVal(isSgValueExp(lowerBound)))) {
          if (ub->get_value() > lb->get_value())
            return IR_COND_LE;
          else
            return IR_COND_GE;
        } else
          throw ir_error("loop stop condition unsupported");
        
      } else
        throw ir_error("malformed fortran loop bounds!!");
      
    }
  }
  
}

IR_Block *IR_roseLoop::body() const {
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  SgNode* loop_body = NULL;
  SgStatement* body_statements = NULL;
  
  if (tf) {
    body_statements = tf->get_loop_body();
  } else if (tfortran) {
    body_statements = isSgStatement(tfortran->get_body());
    
  }
  
  loop_body = isSgNode(body_statements);
  
  SgStatementPtrList list;
  if (isSgBasicBlock(loop_body)) {
    list = isSgBasicBlock(loop_body)->get_statements();
    
    if (list.size() == 1)
      loop_body = isSgNode(*(list.begin()));
  }
  
  if (loop_body == NULL)
    throw ir_error("for loop body is NULL!!");
  
  return new IR_roseBlock(ir_, loop_body);
}

int IR_roseLoop::step_size() const {
  
  SgForStatement *tf = isSgForStatement(tf_);
  SgFortranDo *tfortran = isSgFortranDo(tf_);
  
  if (tf) {
    SgExpression *increment = tf->get_increment();
    
    if (isSgPlusPlusOp(increment))
      return 1;
    if (isSgMinusMinusOp(increment))
      return -1;
    else if (SgAssignOp* assignment = isSgAssignOp(increment)) {
      SgBinaryOp* stepsize = isSgBinaryOp(assignment->get_lhs_operand());
      if (stepsize == NULL)
        throw ir_error("Step size expression is NULL!!");
      SgIntVal* step = isSgIntVal(stepsize->get_lhs_operand());
      return step->get_value();
    } else if (SgBinaryOp* inc = isSgPlusAssignOp(increment)) {
      SgIntVal* step = isSgIntVal(inc->get_rhs_operand());
      return (step->get_value());
    } else if (SgBinaryOp * inc = isSgMinusAssignOp(increment)) {
      SgIntVal* step = isSgIntVal(inc->get_rhs_operand());
      return -(step->get_value());
    } else if (SgBinaryOp * inc = isSgCompoundAssignOp(increment)) {
      SgIntVal* step = isSgIntVal(inc->get_rhs_operand());
      return (step->get_value());
    }
    
  } else if (tfortran) {
    
    SgExpression* increment = tfortran->get_increment();
    
    if (!isSgNullExpression(increment)) {
      if (isSgMinusOp(increment)) {
        if (SgValueExp *inc = isSgValueExp(
              isSgMinusOp(increment)->get_operand()))
          if (isSgIntVal(inc))
            return -(isSgIntVal(inc)->get_value());
      } else {
        if (SgValueExp* inc = isSgValueExp(increment))
          if (isSgIntVal(inc))
            return isSgIntVal(inc)->get_value();
      }
    } else {
    	return 1; // Manu:: if increment is not present, assume it to be 1. Just a workaround, not sure if it will be correct for all cases.
      SgExpression* lowerBound = NULL;
      SgExpression* upperBound = NULL;
      SgExpression* init = tfortran->get_initialization();
      SgIntVal* ub;
      SgIntVal* lb;
      if (SgAssignOp* op = isSgAssignOp(init))
        lowerBound = op->get_rhs_operand();
      
      upperBound = tfortran->get_bound();
      
      if ((upperBound != NULL) && (lowerBound != NULL)) {
        
        if ((ub = isSgIntVal(isSgValueExp(upperBound))) && (lb =
                                                            isSgIntVal(isSgValueExp(lowerBound)))) {
          if (ub->get_value() > lb->get_value())
            return 1;
          else
            return -1;
        } else
          throw ir_error("loop stop condition unsupported");
        
      } else
        throw ir_error("loop stop condition unsupported");
      
    }
    
  }
  
}

IR_Block *IR_roseLoop::convert() {
  const IR_Code *ir = ir_;
  SgNode *tnl = isSgNode(tf_);
  delete this;
  return new IR_roseBlock(ir, tnl);
}

IR_Control *IR_roseLoop::clone() const {
  
  return new IR_roseLoop(ir_, tf_);
  
}

// ----------------------------------------------------------------------------
// Class: IR_roseBlock
// ----------------------------------------------------------------------------

omega::CG_outputRepr *IR_roseBlock::original() const {
  
  omega::CG_outputRepr * tnl;
  
  if (isSgBasicBlock(tnl_)) {
    
    SgStatementPtrList *bb = new SgStatementPtrList();
    SgStatementPtrList::iterator it;
    for (it = (isSgBasicBlock(tnl_)->get_statements()).begin();
         it != (isSgBasicBlock(tnl_)->get_statements()).end()
           && (*it != start_); it++)
      ;
    
    if (it != (isSgBasicBlock(tnl_)->get_statements()).end()) {
      for (; it != (isSgBasicBlock(tnl_)->get_statements()).end(); it++) {
        bb->push_back(*it);
        if ((*it) == end_)
          break;
      }
    }
    tnl = new omega::CG_roseRepr(bb);
    //block = tnl->clone();
    
  } else {
    tnl = new omega::CG_roseRepr(tnl_);
    
    //block = tnl->clone();
  }
  
  return tnl;
  
}
omega::CG_outputRepr *IR_roseBlock::extract() const {
  
  std::string x = tnl_->unparseToString();
  
  omega::CG_roseRepr * tnl;
  
  omega::CG_outputRepr* block;
  
  if (isSgBasicBlock(tnl_)) {
    
    SgStatementPtrList *bb = new SgStatementPtrList();
    SgStatementPtrList::iterator it;
    for (it = (isSgBasicBlock(tnl_)->get_statements()).begin();
         it != (isSgBasicBlock(tnl_)->get_statements()).end()
           && (*it != start_); it++)
      ;
    
    if (it != (isSgBasicBlock(tnl_)->get_statements()).end()) {
      for (; it != (isSgBasicBlock(tnl_)->get_statements()).end(); it++) {
        bb->push_back(*it);
        if ((*it) == end_)
          break;
      }
    }
    tnl = new omega::CG_roseRepr(bb);
    block = tnl->clone();
    
  } else {
    tnl = new omega::CG_roseRepr(tnl_);
    
    block = tnl->clone();
  }
  
  delete tnl;
  return block;
}

IR_Control *IR_roseBlock::clone() const {
  return new IR_roseBlock(ir_, tnl_, start_, end_);
  
}
// ----------------------------------------------------------------------------
// Class: IR_roseIf
// ----------------------------------------------------------------------------
omega::CG_outputRepr *IR_roseIf::condition() const {
  SgNode *tnl = isSgNode(isSgIfStmt(ti_)->get_conditional());
  SgExpression* exp = NULL;
  if (SgExprStatement* stmt = isSgExprStatement(tnl))
    exp = stmt->get_expression();
  /*
    SgExpression *op = iter(tnl);
    if (iter.is_empty())
    throw ir_error("unrecognized if structure");
    tree_node *tn = iter.step();
    if (!iter.is_empty())
    throw ir_error("unrecognized if structure");
    if (!tn->is_instr())
    throw ir_error("unrecognized if structure");
    instruction *ins = static_cast<tree_instr *>(tn)->instr();
    if (!ins->opcode() == io_bfalse)
    throw ir_error("unrecognized if structure");
    operand op = ins->src_op(0);*/
  if (exp == NULL)
    return new omega::CG_roseRepr(tnl);
  else
    return new omega::CG_roseRepr(exp);
}

IR_Block *IR_roseIf::then_body() const {
  SgNode *tnl = isSgNode(isSgIfStmt(ti_)->get_true_body());
  
  //tree_node_list *tnl = ti_->then_part();
  if (tnl == NULL)
    return NULL;
  /*
    tree_node_list_iter iter(tnl);
    if (iter.is_empty())
    return NULL; */
  
  return new IR_roseBlock(ir_, tnl);
}

IR_Block *IR_roseIf::else_body() const {
  SgNode *tnl = isSgNode(isSgIfStmt(ti_)->get_false_body());
  
  //tree_node_list *tnl = ti_->else_part();
  
  if (tnl == NULL)
    return NULL;
  /*
    tree_node_list_iter iter(tnl);
    if (iter.is_empty())
    return NULL;*/
  
  return new IR_roseBlock(ir_, tnl);
}

IR_Block *IR_roseIf::convert() {
  const IR_Code *ir = ir_;
  /* SgNode *tnl = ti_->get_parent();
     SgNode *start, *end;
     start = end = ti_;
     
     //tree_node_list *tnl = ti_->parent();
     //tree_node_list_e *start, *end;
     //start = end = ti_->list_e();
     */
  delete this;
  return new IR_roseBlock(ir, ti_);
}

IR_Control *IR_roseIf::clone() const {
  return new IR_roseIf(ir_, ti_);
}

// -----------------------------------------------------------y-----------------
// Class: IR_roseCode_Global_Init
// ----------------------------------------------------------------------------

IR_roseCode_Global_Init *IR_roseCode_Global_Init::pinstance = 0;

IR_roseCode_Global_Init * IR_roseCode_Global_Init::Instance(char** argv) {
  if (pinstance == 0) {
    pinstance = new IR_roseCode_Global_Init;
    pinstance->project = frontend(2, argv);
    
  }
  return pinstance;
}

// ----------------------------------------------------------------------------
// Class: IR_roseCode
// ----------------------------------------------------------------------------

IR_roseCode::IR_roseCode(const char *filename, const char* proc_name) :
  IR_Code() {
  
  SgProject* project;
  
  char* argv[2];
  int counter = 0;
  argv[0] = (char*) malloc(5 * sizeof(char));
  argv[1] = (char*) malloc((strlen(filename) + 1) * sizeof(char));
  strcpy(argv[0], "rose");
  strcpy(argv[1], filename);
  
  project = (IR_roseCode_Global_Init::Instance(argv))->project;
  //main_ssa = new ssa_unfiltered_cfg::SSA_UnfilteredCfg(project);
  //main_ssa->run();
  firstScope = getFirstGlobalScope(project);
  SgFilePtrList& file_list = project->get_fileList();
  
  for (SgFilePtrList::iterator it = file_list.begin(); it != file_list.end();
       it++) {
    file = isSgSourceFile(*it);
    if (file->get_outputLanguage() == SgFile::e_Fortran_output_language)
      is_fortran_ = true;
    else
      is_fortran_ = false;

    // Manu:: debug
    // if (is_fortran_)
    //   std::cout << "Input is a fortran file\n";
    // else
    //     std::cout << "Input is a C file\n";
    
    root = file->get_globalScope();

    if (!is_fortran_) { // Manu:: this macro should not be created if the input code is in fortran
    	buildCpreprocessorDefineDeclaration(root,
                                        "#define __rose_lt(x,y) ((x)<(y)?(x):(y))",
                                        PreprocessingInfo::before);
    	buildCpreprocessorDefineDeclaration(root,
                                        "#define __rose_gt(x,y) ((x)>(y)?(x):(y))",
                                        PreprocessingInfo::before);
    }
    
    symtab_ = isSgScopeStatement(root)->get_symbol_table();
    SgDeclarationStatementPtrList& declList = root->get_declarations();
    
    p = declList.begin();

    while (p != declList.end()) {
      func = isSgFunctionDeclaration(*p);
      if (func) {
        if (!strcmp((func->get_name().getString()).c_str(), proc_name))
          break;
        
      }
      p++;
      counter++;
    }
    if (p != declList.end())
      break;
    
  }
  
  symtab2_ = func->get_definition()->get_symbol_table();
  symtab3_ = func->get_definition()->get_body()->get_symbol_table();
  // ocg_ = new omega::CG_roseBuilder(func->get_definition()->get_body()->get_symbol_table() , isSgNode(func->get_definition()->get_body())); 
  // Manu:: added is_fortran_ parameter
  ocg_ = new omega::CG_roseBuilder(is_fortran_, root, firstScope,
                                   func->get_definition()->get_symbol_table(),
                                   func->get_definition()->get_body()->get_symbol_table(),
                                   isSgNode(func->get_definition()->get_body()));
  
  i_ = 0; /*i_ handling may need revision */
  
  free(argv[1]);
  free(argv[0]);
  
}

IR_roseCode::~IR_roseCode() {
}

void IR_roseCode::finalizeRose() {
  // Moved this out of the deconstructor
  // ????
  SgProject* project = (IR_roseCode_Global_Init::Instance(NULL))->project;
  // -- Causes coredump. commented out for now -- //
  // processes attributes left in Rose Ast
  //postProcessRoseCodeInsertion(project);
  project->unparse();
  //backend((IR_roseCode_Global_Init::Instance(NULL))->project);
}

IR_ScalarSymbol *IR_roseCode::CreateScalarSymbol(const IR_Symbol *sym, int) {
  char str1[14];
  if (typeid(*sym) == typeid(IR_roseScalarSymbol)) {
    SgType *tn =
      static_cast<const IR_roseScalarSymbol *>(sym)->vs_->get_type();
    sprintf(str1, "newVariable%i\0", i_);
    SgVariableDeclaration* defn = buildVariableDeclaration(str1, tn);
    i_++;
    
    SgInitializedNamePtrList& variables = defn->get_variables();
    SgInitializedNamePtrList::const_iterator i = variables.begin();
    SgInitializedName* initializedName = *i;
    SgVariableSymbol* vs = new SgVariableSymbol(initializedName);
    
    prependStatement(defn,
                     isSgScopeStatement(func->get_definition()->get_body()));
    vs->set_parent(symtab_);
    symtab_->insert(str1, vs);
    
    if (vs == NULL)
      throw ir_error("in CreateScalarSymbol: vs is NULL!!");
    
    return new IR_roseScalarSymbol(this, vs);
  } else if (typeid(*sym) == typeid(IR_roseArraySymbol)) {
    SgType *tn1 =
      static_cast<const IR_roseArraySymbol *>(sym)->vs_->get_type();
    while (isSgArrayType(tn1) || isSgPointerType(tn1)) {
      if (isSgArrayType(tn1))
        tn1 = isSgArrayType(tn1)->get_base_type();
      else if (isSgPointerType(tn1))
        tn1 = isSgPointerType(tn1)->get_base_type();
      else
        throw ir_error(
          "in CreateScalarSymbol: symbol not an array nor a pointer!");
    }
    
    sprintf(str1, "newVariable%i\0", i_);
    i_++;
    
    SgVariableDeclaration* defn1 = buildVariableDeclaration(str1, tn1);
    SgInitializedNamePtrList& variables1 = defn1->get_variables();
    
    SgInitializedNamePtrList::const_iterator i1 = variables1.begin();
    SgInitializedName* initializedName1 = *i1;
    
    SgVariableSymbol *vs1 = new SgVariableSymbol(initializedName1);
    prependStatement(defn1,
                     isSgScopeStatement(func->get_definition()->get_body()));
    
    vs1->set_parent(symtab_);
    symtab_->insert(str1, vs1);
    
    if (vs1 == NULL)
      throw ir_error("in CreateScalarSymbol: vs1 is NULL!!");
    
    return new IR_roseScalarSymbol(this, vs1);
  } else
    throw std::bad_typeid();
  
}

IR_ArraySymbol *IR_roseCode::CreateArraySymbol(const IR_Symbol *sym,
                                               std::vector<omega::CG_outputRepr *> &size, int) {
  SgType *tn;
  char str1[14];
  
  if (typeid(*sym) == typeid(IR_roseScalarSymbol)) {
    tn = static_cast<const IR_roseScalarSymbol *>(sym)->vs_->get_type();
  } else if (typeid(*sym) == typeid(IR_roseArraySymbol)) {
    tn = static_cast<const IR_roseArraySymbol *>(sym)->vs_->get_type();
    while (isSgArrayType(tn) || isSgPointerType(tn)) {
      if (isSgArrayType(tn))
        tn = isSgArrayType(tn)->get_base_type();
      else if (isSgPointerType(tn))
        tn = isSgPointerType(tn)->get_base_type();
      else
        throw ir_error(
          "in CreateScalarSymbol: symbol not an array nor a pointer!");
    }
  } else
    throw std::bad_typeid();

  
  // Manu:: Fortran support
  std::vector<SgExpression *>exprs;
  SgExprListExp *exprLstExp;
  SgExpression* sizeExpression = new SgNullExpression();
  SgArrayType* arrayType = new SgArrayType(tn,sizeExpression);
  sizeExpression->set_parent(arrayType);

  if (!is_fortran_) {
	  for (int i = size.size() - 1; i >= 0; i--) {
		tn = buildArrayType(tn,static_cast<omega::CG_roseRepr *>(size[i])->GetExpression());
	  }
  } else { // Manu:: required for fortran support
	  for (int i = size.size() - 1; i >= 0; i--) {
		exprs.push_back(static_cast<omega::CG_roseRepr *>(size[i])->GetExpression());
	  }
  }

  if (is_fortran_) {
	  exprLstExp = buildExprListExp(exprs);
	  arrayType->set_dim_info(exprLstExp);
 	  exprLstExp->set_parent(arrayType);
 	  arrayType->set_rank(exprLstExp->get_expressions().size());
  }

  static int rose_array_counter = 1;
  SgVariableDeclaration* defn2;
  std::string s;
  if (!is_fortran_) {
	  s = std::string("_P") + omega::to_string(rose_array_counter++);
	  defn2 = buildVariableDeclaration(const_cast<char *>(s.c_str()), tn);
  } else {// Manu:: fortran support
	  s = std::string("f_P") + omega::to_string(rose_array_counter++);
	  defn2 = buildVariableDeclaration(const_cast<char *>(s.c_str()), arrayType);
  }


  SgInitializedNamePtrList& variables2 = defn2->get_variables();
  
  SgInitializedNamePtrList::const_iterator i2 = variables2.begin();
  SgInitializedName* initializedName2 = *i2;
  SgVariableSymbol *vs = new SgVariableSymbol(initializedName2);
  
  prependStatement(defn2,
                   isSgScopeStatement(func->get_definition()->get_body()));
  
  vs->set_parent(symtab_);
  symtab_->insert(SgName(s.c_str()), vs);
  
  return new IR_roseArraySymbol(this, vs);
}

IR_ScalarRef *IR_roseCode::CreateScalarRef(const IR_ScalarSymbol *sym) {
  return new IR_roseScalarRef(this,
                              buildVarRefExp(static_cast<const IR_roseScalarSymbol *>(sym)->vs_));
  
}

IR_ArrayRef *IR_roseCode::CreateArrayRef(const IR_ArraySymbol *sym,
                                         std::vector<omega::CG_outputRepr *> &index) {
  
  int t;
  
  if (sym->n_dim() != index.size())
    throw std::invalid_argument("incorrect array symbol dimensionality");
  
  const IR_roseArraySymbol *l_sym =
    static_cast<const IR_roseArraySymbol *>(sym);
  
  SgVariableSymbol *vs = l_sym->vs_;
  SgExpression* ia1 = buildVarRefExp(vs);
  


  if (is_fortran_) { // Manu:: fortran support
	  std::vector<SgExpression *>exprs;
	  for (int i = 0 ; i < index.size(); i++) {
		exprs.push_back(static_cast<omega::CG_roseRepr *>(index[i])->GetExpression());
	  }
	  SgExprListExp *exprLstExp;
	  exprLstExp = buildExprListExp(exprs);
	  ia1 = buildPntrArrRefExp(ia1,exprLstExp);
  } else {
     for (int i = 0; i < index.size(); i++) {
/*
	  if (is_fortran_)
       t = index.size() - i - 1;
 	  else
       t = i;
*/

     //  std::string y =
    //          isSgNode(
    //                  static_cast<omega::CG_roseRepr *>(index[i])->GetExpression())->unparseToString();
        ia1 = buildPntrArrRefExp(ia1,
                             static_cast<omega::CG_roseRepr *>(index[i])->GetExpression());
    
     }
  }
  
  SgPntrArrRefExp *ia = isSgPntrArrRefExp(ia1);
  //std::string z = isSgNode(ia)->unparseToString();
  
  return new IR_roseArrayRef(this, ia, -1);
  
}

std::vector<IR_ScalarRef *> IR_roseCode::FindScalarRef(
  const omega::CG_outputRepr *repr) const {
  std::vector<IR_ScalarRef *> scalars;
  SgNode *tnl = static_cast<const omega::CG_roseRepr *>(repr)->GetCode();
  SgStatementPtrList *list =
    static_cast<const omega::CG_roseRepr *>(repr)->GetList();
  SgStatement* stmt;
  SgExpression * exp;
  
  if (list != NULL) {
    for (SgStatementPtrList::iterator it = (*list).begin();
         it != (*list).end(); it++) {
      omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgNode(*it));
      std::vector<IR_ScalarRef *> a = FindScalarRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(scalars));
    }
  }
  
  else if (tnl != NULL) {
    if (stmt = isSgStatement(tnl)) {
      if (isSgBasicBlock(stmt)) {
        SgStatementPtrList& stmts =
          isSgBasicBlock(stmt)->get_statements();
        for (int i = 0; i < stmts.size(); i++) {
          omega::CG_roseRepr *r = new omega::CG_roseRepr(
            isSgNode(stmts[i]));
          std::vector<IR_ScalarRef *> a = FindScalarRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(scalars));
        }
        
      } else if (isSgForStatement(stmt)) {
        
        SgForStatement *tnf = isSgForStatement(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgStatement(tnf->get_loop_body()));
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
      } else if (isSgFortranDo(stmt)) {
        SgFortranDo *tfortran = isSgFortranDo(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgStatement(tfortran->get_body()));
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
      } else if (isSgIfStmt(stmt)) {
        SgIfStmt* tni = isSgIfStmt(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgNode(tni->get_conditional()));
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        r = new omega::CG_roseRepr(isSgNode(tni->get_true_body()));
        a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        r = new omega::CG_roseRepr(isSgNode(tni->get_false_body()));
        a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
      } else if (isSgExprStatement(stmt)) {
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgExpression(
            isSgExprStatement(stmt)->get_expression()));
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        
      }
    }
  } else {
    SgExpression* op =
      static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
    if (isSgVarRefExp(op)
        && (!isSgArrayType(isSgVarRefExp(op)->get_type()))) {
      /*    if ((isSgAssignOp(isSgNode(op)->get_parent()))
            && ((isSgAssignOp(isSgNode(op)->get_parent())->get_lhs_operand())
            == op))
            scalars.push_back(
            new IR_roseScalarRef(this,
            isSgAssignOp(isSgNode(op)->get_parent()), -1));
            else
      */
      if (SgBinaryOp* op_ = isSgBinaryOp(
            isSgVarRefExp(op)->get_parent())) {
        if (SgCompoundAssignOp *op__ = isSgCompoundAssignOp(op_)) {
          if (isSgCompoundAssignOp(op_)->get_lhs_operand()
              == isSgVarRefExp(op)) {
            scalars.push_back(
              new IR_roseScalarRef(this, isSgVarRefExp(op),
                                   1));
            scalars.push_back(
              new IR_roseScalarRef(this, isSgVarRefExp(op),
                                   0));
          }
        }
      } else if (SgAssignOp* assmt = isSgAssignOp(
                   isSgVarRefExp(op)->get_parent())) {
        
        if (assmt->get_lhs_operand() == isSgVarRefExp(op))
          scalars.push_back(
            new IR_roseScalarRef(this, isSgVarRefExp(op), 1));
      } else if (SgAssignOp * assmt = isSgAssignOp(
                   isSgVarRefExp(op)->get_parent())) {
        
        if (assmt->get_rhs_operand() == isSgVarRefExp(op))
          scalars.push_back(
            new IR_roseScalarRef(this, isSgVarRefExp(op), 0));
      } else
        scalars.push_back(
          new IR_roseScalarRef(this, isSgVarRefExp(op), 0));
    } else if (isSgAssignOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgAssignOp(op)->get_lhs_operand());
      std::vector<IR_ScalarRef *> a1 = FindScalarRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(scalars));
      omega::CG_roseRepr *r2 = new omega::CG_roseRepr(
        isSgAssignOp(op)->get_rhs_operand());
      std::vector<IR_ScalarRef *> a2 = FindScalarRef(r2);
      delete r2;
      std::copy(a2.begin(), a2.end(), back_inserter(scalars));
      
    } else if (isSgBinaryOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgBinaryOp(op)->get_lhs_operand());
      std::vector<IR_ScalarRef *> a1 = FindScalarRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(scalars));
      omega::CG_roseRepr *r2 = new omega::CG_roseRepr(
        isSgBinaryOp(op)->get_rhs_operand());
      std::vector<IR_ScalarRef *> a2 = FindScalarRef(r2);
      delete r2;
      std::copy(a2.begin(), a2.end(), back_inserter(scalars));
    } else if (isSgUnaryOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgUnaryOp(op)->get_operand());
      std::vector<IR_ScalarRef *> a1 = FindScalarRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(scalars));
    }
    
  }
  return scalars;
  
}

std::vector<IR_ArrayRef *> IR_roseCode::FindArrayRef(
  const omega::CG_outputRepr *repr) const {
  std::vector<IR_ArrayRef *> arrays;
  SgNode *tnl = static_cast<const omega::CG_roseRepr *>(repr)->GetCode();
  SgStatementPtrList* list =
    static_cast<const omega::CG_roseRepr *>(repr)->GetList();
  SgStatement* stmt;
  SgExpression * exp;
  
  if (list != NULL) {
    for (SgStatementPtrList::iterator it = (*list).begin();
         it != (*list).end(); it++) {
      omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgNode(*it));
      std::vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
    }
  } else if (tnl != NULL) {
    if (stmt = isSgStatement(tnl)) {
      if (isSgBasicBlock(stmt)) {
        SgStatementPtrList& stmts =
          isSgBasicBlock(stmt)->get_statements();
        for (int i = 0; i < stmts.size(); i++) {
          omega::CG_roseRepr *r = new omega::CG_roseRepr(
            isSgNode(stmts[i]));
          std::vector<IR_ArrayRef *> a = FindArrayRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(arrays));
        }
        
      } else if (isSgForStatement(stmt)) {
        
        SgForStatement *tnf = isSgForStatement(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgStatement(tnf->get_loop_body()));
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
      } else if (isSgFortranDo(stmt)) {
        SgFortranDo *tfortran = isSgFortranDo(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgStatement(tfortran->get_body()));
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
      } else if (isSgIfStmt(stmt)) {
        SgIfStmt* tni = isSgIfStmt(stmt);
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgNode(tni->get_conditional()));
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        r = new omega::CG_roseRepr(isSgNode(tni->get_true_body()));
        a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        r = new omega::CG_roseRepr(isSgNode(tni->get_false_body()));
        a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
      } else if (isSgExprStatement(stmt)) {
        omega::CG_roseRepr *r = new omega::CG_roseRepr(
          isSgExpression(
            isSgExprStatement(stmt)->get_expression()));
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        
      }
    }
  } else {
    SgExpression* op =
      static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
    if (isSgPntrArrRefExp(op)) {
      
      SgVarRefExp* base;
      SgExpression* op2;
      if (isSgCompoundAssignOp(isSgPntrArrRefExp(op)->get_parent())) {
        IR_roseArrayRef *ref1 = new IR_roseArrayRef(this,
                                                    isSgPntrArrRefExp(op), 0);
        arrays.push_back(ref1);
        IR_roseArrayRef *ref2 = new IR_roseArrayRef(this,
                                                    isSgPntrArrRefExp(op), 1);
        arrays.push_back(ref2);
      } else {
        IR_roseArrayRef *ref3 = new IR_roseArrayRef(this,
                                                    isSgPntrArrRefExp(op), -1);
        arrays.push_back(ref3);
        
        while (isSgPntrArrRefExp(op)) {
          op2 = isSgPntrArrRefExp(op)->get_rhs_operand();
          op = isSgPntrArrRefExp(op)->get_lhs_operand();
          omega::CG_roseRepr *r = new omega::CG_roseRepr(op2);
          std::vector<IR_ArrayRef *> a = FindArrayRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(arrays));
          
        }
      }
      /* base = isSgVarRefExp(op);
         SgVariableSymbol *arrSymbol = (SgVariableSymbol*)(base->get_symbol());
         SgArrayType *arrType = isSgArrayType(arrSymbol->get_type()); 
         
         SgExprListExp* dimList = arrType->get_dim_info();
         
         if(dimList != NULL){  
         SgExpressionPtrList::iterator it = dimList->get_expressions().begin();             
         SgExpression *expr;
         
         
         for (int i = 0; it != dimList->get_expressions().end(); it++, i++)
         {
         expr = *it;         
         
         omega::CG_roseRepr *r = new omega::CG_roseRepr(expr);
         std::vector<IR_ArrayRef *> a = FindArrayRef(r);
         delete r;
         std::copy(a.begin(), a.end(), back_inserter(arrays));
         }
         
         }
         arrays.push_back(ref);
      */
    } else if (isSgAssignOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgAssignOp(op)->get_lhs_operand());
      std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(arrays));
      omega::CG_roseRepr *r2 = new omega::CG_roseRepr(
        isSgAssignOp(op)->get_rhs_operand());
      std::vector<IR_ArrayRef *> a2 = FindArrayRef(r2);
      delete r2;
      std::copy(a2.begin(), a2.end(), back_inserter(arrays));
      
    } else if (isSgBinaryOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgBinaryOp(op)->get_lhs_operand());
      std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(arrays));
      omega::CG_roseRepr *r2 = new omega::CG_roseRepr(
        isSgBinaryOp(op)->get_rhs_operand());
      std::vector<IR_ArrayRef *> a2 = FindArrayRef(r2);
      delete r2;
      std::copy(a2.begin(), a2.end(), back_inserter(arrays));
    } else if (isSgUnaryOp(op)) {
      omega::CG_roseRepr *r1 = new omega::CG_roseRepr(
        isSgUnaryOp(op)->get_operand());
      std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;
      std::copy(a1.begin(), a1.end(), back_inserter(arrays));
    }
    
  }
  return arrays;
  
  /* std::string x;
     SgStatement* stmt = isSgStatement(tnl);
     SGExprStatement* expr_statement = isSgExprStatement(stmt);  
     SgExpression* exp= NULL;
     if(expr_statement == NULL){
     if(! (SgExpression* exp = isSgExpression(tnl))
     throw ir_error("FindArrayRef: Not a stmt nor an expression!!");
     
     if( expr_statement != NULL){  
     for(int i=0; i < tnl->get_numberOfTraversalSuccessors(); i++){   
     
     SgNode* tn = isSgStatement(tnl);
     SgStatement* stmt = isSgStatement(tn);
     if(stmt != NULL){
     SgExprStatement* expr_statement = isSgExprStatement(tn);           
     if(expr_statement != NULL)
     x = isSgNode(expr_statement)->unparseToString();   
     exp = expr_statement->get_expression();
     
     }     
     else{
     
     exp = isSgExpression(tn);
     } 
     if(exp != NULL){
     x = isSgNode(exp)->unparseToString();
     
     if(SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(exp) ){
     if(arrRef == NULL)
     throw ir_error("something wrong");   
     IR_roseArrayRef *ref = new IR_roseArrayRef(this, arrRef);
     arrays.push_back(ref);
     }
     
     omega::CG_outputRepr *r = new omega::CG_roseRepr(isSgNode(exp->get_rhs_operand()));
     std::vector<IR_ArrayRef *> a = FindArrayRef(r);
     delete r;
     std::copy(a.begin(), a.end(), back_inserter(arrays));
     
     omega::CG_outputRepr *r1 = new omega::CG_roseRepr(isSgNode(exp->get_lhs_operand()));
     std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
     delete r1;
     std::copy(a1.begin(), a1.end(), back_inserter(arrays));
     
     }
     }*/
  
}

std::vector<IR_Control *> IR_roseCode::FindOneLevelControlStructure(
  const IR_Block *block) const {

  std::vector<IR_Control *> controls;
  int i;
  int j;
  int begin;
  int end;
  SgNode* tnl_ =
    ((static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_);
  
  if (isSgForStatement(tnl_))
    controls.push_back(new IR_roseLoop(this, tnl_));
  else if (isSgFortranDo(tnl_))
	  controls.push_back(new IR_roseLoop(this, tnl_));
  else if (isSgIfStmt(tnl_))
    controls.push_back(new IR_roseIf(this, tnl_));
  
  else if (isSgBasicBlock(tnl_)) {
    
    SgStatementPtrList& stmts = isSgBasicBlock(tnl_)->get_statements();
    
    for (i = 0; i < stmts.size(); i++) {
      if (isSgNode(stmts[i])
          == ((static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->start_))
        begin = i;
      if (isSgNode(stmts[i])
          == ((static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->end_))
        end = i;
    }
    
    SgNode* start = NULL;
    SgNode* prev = NULL;
    for (i = begin; i <= end; i++) {
      if (isSgForStatement(stmts[i]) || isSgFortranDo(stmts[i])) {
        if (start != NULL) {
          controls.push_back(
            new IR_roseBlock(this,
                             (static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_,
                             start, prev));
          start = NULL;
        }
        controls.push_back(new IR_roseLoop(this, isSgNode(stmts[i])));
      } else if (isSgIfStmt(stmts[i])) {
        if (start != NULL) {
          controls.push_back(
            new IR_roseBlock(this,
                             (static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_,
                             start, prev));
          start = NULL;
        }
        controls.push_back(new IR_roseIf(this, isSgNode(stmts[i])));
        
      } else if (start == NULL)
        start = isSgNode(stmts[i]);
      
      prev = isSgNode(stmts[i]);
    }
    
    if ((start != NULL) && (start != isSgNode(stmts[begin])))
      controls.push_back(
        new IR_roseBlock(this,
                         (static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_,
                         start, prev));
  }
  
  return controls;
  
}

/*std::vector<IR_Control *> IR_roseCode::FindOneLevelControlStructure(const IR_Block *block) const {
  
  std::vector<IR_Control *> controls;
  int i;
  int j;
  SgNode* tnl_ = ((static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_);   
  
  
  if(isSgForStatement(tnl_))
  controls.push_back(new IR_roseLoop(this,tnl_));
  
  else if(isSgBasicBlock(tnl_)){
  
  SgStatementPtrList& stmts = isSgBasicBlock(tnl_)->get_statements();
  
  for(i =0; i < stmts.size(); i++){
  if(isSgNode(stmts[i]) == ((static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->start_))          
  break; 
  }
  
  
  SgNode* start= NULL;
  SgNode* prev= NULL;  
  for(; i < stmts.size(); i++){
  if ( isSgForStatement(stmts[i]) || isSgFortranDo(stmts[i])){
  if(start != NULL){   
  controls.push_back(new IR_roseBlock(this, (static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_ , start, prev)); 
  start = NULL;
  }   
  controls.push_back(new IR_roseLoop(this, isSgNode(stmts[i])));
  } 
  else if( start == NULL )   
  start = isSgNode(stmts[i]);
  
  prev = isSgNode(stmts[i]);
  }   
  
  if((start != NULL) && (start != isSgNode(stmts[0])))
  controls.push_back(new IR_roseBlock(this, (static_cast<IR_roseBlock *>(const_cast<IR_Block *>(block)))->tnl_, start, prev));
  }   
  
  return controls;
  
  }
  
*/
IR_Block *IR_roseCode::MergeNeighboringControlStructures(
  const std::vector<IR_Control *> &controls) const {
  if (controls.size() == 0)
    return NULL;
  
  SgNode *tnl = NULL;
  SgNode *start, *end;
  for (int i = 0; i < controls.size(); i++) {
    switch (controls[i]->type()) {
    case IR_CONTROL_LOOP: {
      SgNode *tf = static_cast<IR_roseLoop *>(controls[i])->tf_;
      if (tnl == NULL) {
        tnl = tf->get_parent();
        start = end = tf;
      } else {
        if (tnl != tf->get_parent())
          throw ir_error("controls to merge not at the same level");
        end = tf;
      }
      break;
    }
    case IR_CONTROL_BLOCK: {
      if (tnl == NULL) {
        tnl = static_cast<IR_roseBlock *>(controls[0])->tnl_;
        start = static_cast<IR_roseBlock *>(controls[0])->start_;
        end = static_cast<IR_roseBlock *>(controls[0])->end_;
      } else {
        if (tnl != static_cast<IR_roseBlock *>(controls[0])->tnl_)
          throw ir_error("controls to merge not at the same level");
        end = static_cast<IR_roseBlock *>(controls[0])->end_;
      }
      break;
    }
    default:
      throw ir_error("unrecognized control to merge");
    }
  }
  
  return new IR_roseBlock(controls[0]->ir_, tnl, start, end);
}

IR_Block *IR_roseCode::GetCode() const {
  SgFunctionDefinition* def = NULL;
  SgBasicBlock* block = NULL;
  if (func != 0) {
    if (def = func->get_definition()) {
      if (block = def->get_body())
        return new IR_roseBlock(this,
                                func->get_definition()->get_body());
    }
  }
  
  return NULL;
  
}

void IR_roseCode::ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr) {
  /*    SgStatementPtrList *tnl =
        static_cast<omega::CG_roseRepr *>(repr)->GetList();
        SgNode *tf_old;
  */
  SgStatementPtrList *tnl =
    static_cast<omega::CG_roseRepr *>(repr)->GetList();
  SgNode* node_ = static_cast<omega::CG_roseRepr *>(repr)->GetCode();
  SgNode * tf_old;
  
  /* May need future revision it tnl has more than one statement */
  
  switch (old->type()) {
    
  case IR_CONTROL_LOOP:
    tf_old = static_cast<IR_roseLoop *>(old)->tf_;
    break;
  case IR_CONTROL_BLOCK:
    tf_old = static_cast<IR_roseBlock *>(old)->start_;
    break;
    
  default:
    throw ir_error("control structure to be replaced not supported");
    break;
  }
  
  std::string y = tf_old->unparseToString();
  SgStatement *s = isSgStatement(tf_old);
  if (s != 0) {
    SgStatement *p = isSgStatement(tf_old->get_parent());
    
    if (p != 0) {
      SgStatement* temp = s;
      if (tnl != NULL) {
        SgStatementPtrList::iterator it = (*tnl).begin();
        p->insert_statement(temp, *it, true);
        temp = *it;
        p->remove_statement(s);
        it++;
        for (; it != (*tnl).end(); it++) {
          p->insert_statement(temp, *it, false);
          temp = *it;
        }
      } else if (node_ != NULL) {
        if (!isSgStatement(node_))
          throw ir_error("Replacing Code not a statement!");
        else {
          SgStatement* replace_ = isSgStatement(node_);
          p->insert_statement(s, replace_, true);
          p->remove_statement(s);
          
        }
      } else {
        throw ir_error("Replacing Code not a statement!");
      }
    } else
      throw ir_error("Replacing Code not a statement!");
  } else
    throw ir_error("Replacing Code not a statement!");
  
  delete old;
  delete repr;
  /* May need future revision it tnl has more than one statement */
  /*
    switch (old->type()) {
    
    case IR_CONTROL_LOOP:
    tf_old = static_cast<IR_roseLoop *>(old)->tf_;
    break;
    case IR_CONTROL_BLOCK:
    tf_old = static_cast<IR_roseBlock *>(old)->start_;
    break;
    
    default:
    throw ir_error("control structure to be replaced not supported");
    break;
    }
    
    // std::string y = tf_old->unparseToString();
    SgStatement *s = isSgStatement(tf_old);
    if (s != 0) {
    SgStatement *p = isSgStatement(tf_old->get_parent());
    
    if (p != 0) {
    //      SgStatement* it2 = isSgStatement(tnl);
    
    //   if(it2 != NULL){
    p->replace_statement(s, *tnl);
    //   }
    //   else {
    //          throw ir_error("Replacing Code not a statement!");
    //      }
    } else
    throw ir_error("Replacing Code not a statement!");
    } else
    throw ir_error("Replacing Code not a statement!");
    //  y = tnl->unparseToString();
    delete old;
    delete repr;
  */
}

void IR_roseCode::ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr) {
  
  SgExpression* op = static_cast<omega::CG_roseRepr *>(repr)->GetExpression();
  
  if (typeid(*old) == typeid(IR_roseArrayRef)) {
    SgPntrArrRefExp* ia_orig = static_cast<IR_roseArrayRef *>(old)->ia_;
    SgExpression* parent = isSgExpression(isSgNode(ia_orig)->get_parent());
    std::string x = isSgNode(op)->unparseToString();
    std::string y = isSgNode(ia_orig)->unparseToString();
    if (parent != NULL) {
      std::string z = isSgNode(parent)->unparseToString();
      parent->replace_expression(ia_orig, op);
      isSgNode(op)->set_parent(isSgNode(parent));
      
      /* if(isSgBinaryOp(parent))
         {
         if(isSgBinaryOp(parent)->get_lhs_operand() == ia_orig){
         isSgBinaryOp(parent)->set_lhs_operand(op);   
         }else if(isSgBinaryOp(parent)->get_rhs_operand() == ia_orig){
         isSgBinaryOp(parent)->set_rhs_operand(op); 
         
         
         } 
         else
         parent->replace_expression(ia_orig, op);
      */
    } else {
      SgStatement* parent_stmt = isSgStatement(
        isSgNode(ia_orig)->get_parent());
      if (parent_stmt != NULL)
        parent_stmt->replace_expression(ia_orig, op);
      else
        throw ir_error(
          "ReplaceExpression: parent neither expression nor statement");
    }
  } else
    throw ir_error("replacing a scalar variable not implemented");
  
  delete old;
}

/*std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > IR_roseCode::FindScalarDeps(
  const omega::CG_outputRepr *repr1, const omega::CG_outputRepr *repr2,
  std::vector<std::string> index, int i, int j) {
  
  std::vector<DependenceVector> dvs1;
  std::vector<DependenceVector> dvs2;
  SgNode *tnl_1 = static_cast<const omega::CG_roseRepr *>(repr1)->GetCode();
  SgNode *tnl_2 = static_cast<const omega::CG_roseRepr *>(repr2)->GetCode();
  SgStatementPtrList* list_1 =
  static_cast<const omega::CG_roseRepr *>(repr1)->GetList();
  SgStatementPtrList output_list_1;
  
  std::map<SgVarRefExp*, IR_ScalarRef*> read_scalars_1;
  std::map<SgVarRefExp*, IR_ScalarRef*> write_scalars_1;
  std::set<std::string> indices;
  //std::set<VirtualCFG::CFGNode> reaching_defs_1;
  std::set<std::string> def_vars_1;
  
  populateLists(tnl_1, list_1, output_list_1);
  populateScalars(repr1, read_scalars_1, write_scalars_1, indices, index);
  //def_vars_1);
  //findDefinitions(output_list_1, reaching_defs_1, write_scalars_1);
  //def_vars_1);
  if (repr1 == repr2)
  checkSelfDependency(output_list_1, dvs1, read_scalars_1,
  write_scalars_1, index, i, j);
  else {
  SgStatementPtrList* list_2 =
  static_cast<const omega::CG_roseRepr *>(repr2)->GetList();
  SgStatementPtrList output_list_2;
  
  std::map<SgVarRefExp*, IR_ScalarRef*> read_scalars_2;
  std::map<SgVarRefExp*, IR_ScalarRef*> write_scalars_2;
  //std::set<VirtualCFG::CFGNode> reaching_defs_2;
  std::set<std::string> def_vars_2;
  
  populateLists(tnl_2, list_2, output_list_2);
  populateScalars(repr2, read_scalars_2, write_scalars_2, indices, index);
  //def_vars_2);
  
  checkDependency(output_list_2, dvs1, read_scalars_2, write_scalars_1,
  index, i, j);
  checkDependency(output_list_1, dvs1, read_scalars_1, write_scalars_2,
  index, i, j);
  checkWriteDependency(output_list_2, dvs1, write_scalars_2,
  write_scalars_1, index, i, j);
  checkWriteDependency(output_list_1, dvs1, write_scalars_1,
  write_scalars_2, index, i, j);
  }
  
  return std::make_pair(dvs1, dvs2);
  //populateLists(tnl_2, list_2, list2);
  
  }
*/
IR_OPERATION_TYPE IR_roseCode::QueryExpOperation(
  const omega::CG_outputRepr *repr) const {
  SgExpression* op =
    static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
  
  if (isSgValueExp(op))
    return IR_OP_CONSTANT;
  else if (isSgVarRefExp(op) || isSgPntrArrRefExp(op))
    return IR_OP_VARIABLE;
  else if (isSgAssignOp(op) || isSgCompoundAssignOp(op))
    return IR_OP_ASSIGNMENT;
  else if (isSgAddOp(op))
    return IR_OP_PLUS;
  else if (isSgSubtractOp(op))
    return IR_OP_MINUS;
  else if (isSgMultiplyOp(op))
    return IR_OP_MULTIPLY;
  else if (isSgDivideOp(op))
    return IR_OP_DIVIDE;
  else if (isSgMinusOp(op))
    return IR_OP_NEGATIVE;
  else if (isSgConditionalExp(op)) {
    SgExpression* cond = isSgConditionalExp(op)->get_conditional_exp();
    if (isSgGreaterThanOp(cond))
      return IR_OP_MAX;
    else if (isSgLessThanOp(cond))
      return IR_OP_MIN;
  } else if (isSgUnaryAddOp(op))
    return IR_OP_POSITIVE;
  else if (isSgNullExpression(op))
    return IR_OP_NULL;
  else
    return IR_OP_UNKNOWN;
}
/*void IR_roseCode::populateLists(SgNode* tnl_1, SgStatementPtrList* list_1,
  SgStatementPtrList& output_list_1) {
  if ((tnl_1 == NULL) && (list_1 != NULL)) {
  output_list_1 = *list_1;
  } else if (tnl_1 != NULL) {
  
  if (isSgForStatement(tnl_1)) {
  SgStatement* check = isSgForStatement(tnl_1)->get_loop_body();
  if (isSgBasicBlock(check)) {
  output_list_1 = isSgBasicBlock(check)->get_statements();
  
  } else
  output_list_1.push_back(check);
  
  } else if (isSgBasicBlock(tnl_1))
  output_list_1 = isSgBasicBlock(tnl_1)->get_statements();
  else if (isSgExprStatement(tnl_1))
  output_list_1.push_back(isSgExprStatement(tnl_1));
  else
  //if (isSgIfStmt(tnl_1)) {
  
  throw ir_error(
  "Statement type not handled, (probably IF statement)!!");
  
  }
  
  }
  
  void IR_roseCode::populateScalars(const omega::CG_outputRepr *repr1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
  std::set<std::string> &indices, std::vector<std::string> &index) {
  
  //std::set<std::string> &def_vars) {
  std::vector<IR_ScalarRef *> scalars = FindScalarRef(repr1);
  
  for (int k = 0; k < index.size(); k++)
  indices.insert(index[k]);
  
  for (int k = 0; k < scalars.size(); k++)
  if (indices.find(scalars[k]->name()) == indices.end()) {
  if (scalars[k]->is_write()) {
  write_scalars_1.insert(
  std::pair<SgVarRefExp*, IR_ScalarRef*>(
  (isSgVarRefExp(
  static_cast<const omega::CG_roseRepr *>(scalars[k]->convert())->GetExpression())),
  scalars[k]));
  
  } else
  
  read_scalars_1.insert(
  std::pair<SgVarRefExp*, IR_ScalarRef*>(
  (isSgVarRefExp(
  static_cast<const omega::CG_roseRepr *>(scalars[k]->convert())->GetExpression())),
  scalars[k]));
  }
  
  }
  
  
  void IR_roseCode::checkWriteDependency(SgStatementPtrList &output_list_1,
  std::vector<DependenceVector> &dvs1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
  std::vector<std::string> &index, int i, int j) {
  
  for (std::map<SgVarRefExp*, IR_ScalarRef*>::iterator it =
  read_scalars_1.begin(); it != read_scalars_1.end(); it++) {
  SgVarRefExp* var__ = it->first;
  
  ssa_unfiltered_cfg::SSA_UnfilteredCfg::NodeReachingDefTable to_compare =
  main_ssa->getReachingDefsBefore(isSgNode(var__));
  
  for (ssa_unfiltered_cfg::SSA_UnfilteredCfg::NodeReachingDefTable::iterator it4 =
  to_compare.begin(); it4 != to_compare.end(); it4++) {
  ssa_unfiltered_cfg::SSA_UnfilteredCfg::VarName var_ = it4->first;
  for (int j = 0; j < var_.size(); j++) {
  int found = 0;
  if (var_[j] == var__->get_symbol()->get_declaration()) {
  
  ssa_unfiltered_cfg::ReachingDef::ReachingDefPtr to_compare_2 =
  it4->second;
  
  if (to_compare_2->isPhiFunction()) {
  std::set<VirtualCFG::CFGNode> to_compare_set =
  to_compare_2->getActualDefinitions();
  for (std::set<VirtualCFG::CFGNode>::iterator cfg_it =
  to_compare_set.begin();
  cfg_it != to_compare_set.end(); cfg_it++) {
  
  if (isSgAssignOp(cfg_it->getNode())
  || isSgCompoundAssignOp(cfg_it->getNode()))
  if (SgVarRefExp* variable =
  isSgVarRefExp(
  isSgBinaryOp(cfg_it->getNode())->get_lhs_operand())) {
  
  if (write_scalars_1.find(variable)
  != write_scalars_1.end()) {
  
  
  //end debug
  found = 1;
  DependenceVector dv1;
  dv1.sym = it->second->symbol();
  dv1.is_scalar_dependence = true;
  
  int max = (j > i) ? j : i;
  int start = index.size() - max;
  
  //1.lbounds.push_back(0);
  //1.ubounds.push_back(0);
  //dv2.sym =
  //        read_scalars_2.find(*di)->second->symbol();
  for (int k = 0; k < index.size(); k++) {
  if (k >= max) {
  dv1.lbounds.push_back(
  negInfinity);
  dv1.ubounds.push_back(-1);
  } else {
  dv1.lbounds.push_back(0);
  dv1.ubounds.push_back(0);
  
  }
  
  }
  dvs1.push_back(dv1);
  break;
  }
  }
  }
  
  }
  
  }
  if (found == 1)
  break;
  }
  }
  }
  }
  void IR_roseCode::checkDependency(SgStatementPtrList &output_list_1,
  std::vector<DependenceVector> &dvs1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
  std::vector<std::string> &index, int i, int j) {
  
  for (SgStatementPtrList::iterator it2 = output_list_1.begin();
  it2 != output_list_1.end(); it2++) {
  
  std::set<SgVarRefExp*> vars_1 = main_ssa->getUsesAtNode(
  isSgNode(isSgExprStatement(*it2)->get_expression()));
  
  std::set<SgVarRefExp*>::iterator di;
  
  for (di = vars_1.begin(); di != vars_1.end(); di++) {
  int found = 0;
  if (read_scalars_1.find(*di) != read_scalars_1.end()) {
  
  ssa_unfiltered_cfg::ReachingDef::ReachingDefPtr to_compare =
  main_ssa->getDefinitionForUse(*di);
  if (to_compare->isPhiFunction()) {
  
  std::set<VirtualCFG::CFGNode> to_compare_set =
  to_compare->getActualDefinitions();
  
  for (std::set<VirtualCFG::CFGNode>::iterator cfg_it =
  to_compare_set.begin();
  cfg_it != to_compare_set.end(); cfg_it++) {
  
  
  if (SgAssignOp* definition = isSgAssignOp(
  cfg_it->getNode()))
  if (SgVarRefExp* variable = isSgVarRefExp(
  definition->get_lhs_operand())) {
  
  if (write_scalars_1.find(variable)
  != write_scalars_1.end()) {
  
  found = 1;
  DependenceVector dv1;
  //DependenceVector dv2;
  dv1.sym =
  read_scalars_1.find(*di)->second->symbol();
  dv1.is_scalar_dependence = true;
  
  int max = (j > i) ? j : i;
  int start = index.size() - max;
  
  //1.lbounds.push_back(0);
  //1.ubounds.push_back(0);
  //dv2.sym =
  //        read_scalars_2.find(*di)->second->symbol();
  for (int k = 0; k < index.size(); k++) {
  if (k >= max) {
  dv1.lbounds.push_back(negInfinity);
  dv1.ubounds.push_back(-1);
  } else {
  dv1.lbounds.push_back(0);
  dv1.ubounds.push_back(0);
  
  }
  
  }
  dvs1.push_back(dv1);
  break;
  }
  }
  }
  }
  if (found == 1)
  break;
  }
  }
  }
  
  }
  
  void IR_roseCode::checkSelfDependency(SgStatementPtrList &output_list_1,
  std::vector<DependenceVector> &dvs1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
  std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
  std::vector<std::string> &index, int i, int j) {
  
  for (SgStatementPtrList::iterator it2 = output_list_1.begin();
  it2 != output_list_1.end(); it2++) {
  
  std::set<SgVarRefExp*> vars_1 = main_ssa->getUsesAtNode(
  isSgNode(isSgExprStatement(*it2)->get_expression()));
  
  std::set<SgVarRefExp*>::iterator di;
  
  for (di = vars_1.begin(); di != vars_1.end(); di++) {
  
  if (read_scalars_1.find(*di) != read_scalars_1.end()) {
  
  ssa_unfiltered_cfg::ReachingDef::ReachingDefPtr to_compare =
  main_ssa->getDefinitionForUse(*di);
  if (to_compare->isPhiFunction()) {
  
  std::set<VirtualCFG::CFGNode> to_compare_set =
  to_compare->getActualDefinitions();
  int found = 0;
  for (std::set<VirtualCFG::CFGNode>::iterator cfg_it =
  to_compare_set.begin();
  cfg_it != to_compare_set.end(); cfg_it++) {
  
  if (isSgAssignOp(cfg_it->getNode())
  || isSgCompoundAssignOp(cfg_it->getNode()))
  if (SgVarRefExp* variable =
  isSgVarRefExp(
  isSgBinaryOp(cfg_it->getNode())->get_lhs_operand())) {
  
  if (write_scalars_1.find(variable)
  == write_scalars_1.end()) {
  
  
  found = 1;
  DependenceVector dv1;
  dv1.sym =
  read_scalars_1.find(*di)->second->symbol();
  dv1.is_scalar_dependence = true;
  
  int max = (j > i) ? j : i;
  int start = index.size() - max;
  
  //1.lbounds.push_back(0);
  //1.ubounds.push_back(0);
  //dv2.sym =
  //        read_scalars_2.find(*di)->second->symbol();
  for (int k = 0; k < index.size(); k++) {
  if (k >= max) {
  dv1.lbounds.push_back(negInfinity);
  dv1.ubounds.push_back(-1);
  } else {
  dv1.lbounds.push_back(0);
  dv1.ubounds.push_back(0);
  
  }
  
  }
  dvs1.push_back(dv1);
  break;
  }
  }
  }
  }
  
  }
  }
  }
  
  }
*/
IR_CONDITION_TYPE IR_roseCode::QueryBooleanExpOperation(
  const omega::CG_outputRepr *repr) const {
  SgExpression* op2 =
    static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
  SgNode* op;
  
  if (op2 == NULL) {
    op = static_cast<const omega::CG_roseRepr *>(repr)->GetCode();
    
    if (op != NULL) {
      if (isSgExprStatement(op))
        op2 = isSgExprStatement(op)->get_expression();
      else
        return IR_COND_UNKNOWN;
    } else
      return IR_COND_UNKNOWN;
  }
  
  if (isSgEqualityOp(op2))
    return IR_COND_EQ;
  else if (isSgNotEqualOp(op2))
    return IR_COND_NE;
  else if (isSgLessThanOp(op2))
    return IR_COND_LT;
  else if (isSgLessOrEqualOp(op2))
    return IR_COND_LE;
  else if (isSgGreaterThanOp(op2))
    return IR_COND_GT;
  else if (isSgGreaterOrEqualOp(op2))
    return IR_COND_GE;
  
  return IR_COND_UNKNOWN;
  
}

std::vector<omega::CG_outputRepr *> IR_roseCode::QueryExpOperand(
  const omega::CG_outputRepr *repr) const {
  std::vector<omega::CG_outputRepr *> v;
  SgExpression* op1;
  SgExpression* op2;
  SgExpression* op =
    static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
  omega::CG_roseRepr *repr1;
  
  if (isSgValueExp(op) || isSgVarRefExp(op)) {
    omega::CG_roseRepr *repr = new omega::CG_roseRepr(op);
    v.push_back(repr);
  } else if (isSgAssignOp(op)) {
    op1 = isSgAssignOp(op)->get_rhs_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
    /*may be a problem as assignOp is a binaryop destop might be needed */
  } else if (isSgMinusOp(op)) {
    op1 = isSgMinusOp(op)->get_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
  } else if (isSgUnaryAddOp(op)) {
    op1 = isSgUnaryAddOp(op)->get_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
  } else if ((isSgAddOp(op) || isSgSubtractOp(op))
             || (isSgMultiplyOp(op) || isSgDivideOp(op))) {
    op1 = isSgBinaryOp(op)->get_lhs_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
    
    op2 = isSgBinaryOp(op)->get_rhs_operand();
    repr1 = new omega::CG_roseRepr(op2);
    v.push_back(repr1);
  } else if (isSgConditionalExp(op)) {
    SgExpression* cond = isSgConditionalExp(op)->get_conditional_exp();
    op1 = isSgBinaryOp(cond)->get_lhs_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
    
    op2 = isSgBinaryOp(cond)->get_rhs_operand();
    repr1 = new omega::CG_roseRepr(op2);
    v.push_back(repr1);
  } else if (isSgCompoundAssignOp(op)) {
    SgExpression* cond = isSgCompoundAssignOp(op);
    op1 = isSgBinaryOp(cond)->get_lhs_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
    
    op2 = isSgBinaryOp(cond)->get_rhs_operand();
    repr1 = new omega::CG_roseRepr(op2);
    v.push_back(repr1);
    
  } else if (isSgBinaryOp(op)) {
    
    op1 = isSgBinaryOp(op)->get_lhs_operand();
    repr1 = new omega::CG_roseRepr(op1);
    v.push_back(repr1);
    
    op2 = isSgBinaryOp(op)->get_rhs_operand();
    repr1 = new omega::CG_roseRepr(op2);
    v.push_back(repr1);
  }
  
  else
    throw ir_error("operation not supported");
  
  return v;
}

IR_Ref *IR_roseCode::Repr2Ref(const omega::CG_outputRepr *repr) const {
  SgExpression* op =
    static_cast<const omega::CG_roseRepr *>(repr)->GetExpression();
  
  if (SgValueExp* im = isSgValueExp(op)) {
    if (isSgIntVal(im))
      return new IR_roseConstantRef(this,
                                    static_cast<omega::coef_t>(isSgIntVal(im)->get_value()));
    else if (isSgUnsignedIntVal(im))
      return new IR_roseConstantRef(this,
                                    static_cast<omega::coef_t>(isSgUnsignedIntVal(im)->get_value()));
    else if (isSgLongIntVal(im))
      return new IR_roseConstantRef(this,
                                    static_cast<omega::coef_t>(isSgLongIntVal(im)->get_value()));
    else if (isSgFloatVal(im))
      return new IR_roseConstantRef(this, isSgFloatVal(im)->get_value());
    else
      assert(0);
    
  } else if (isSgVarRefExp(op))
    return new IR_roseScalarRef(this, isSgVarRefExp(op));
  else
    assert(0);
  
}

