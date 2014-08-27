/*****************************************************************************
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's compiler intermediate representation interface that extends
 Omega's builder interface to accomodate compiler analyses and
 extra code generation.
.
 Notes:
   Unlike CG_outputRepr, IR_Symbol,IR_Ref and IR_Control are place holders
 to the underlying code, thus deleting or duplicating them does not affect
 the actual code.  Similar to Omega builder's memory allocation strategy,
 all non-const pointer parameters of CG_outputRepr/IR_Symbol/IR_Ref/IR_Control
 are destroyed after the call.

 History:
   02/2009 Created by Chun Chen.
   06/2010 Add IR_Control interface, by chun.  
*****************************************************************************/

#ifndef IR_CODE_HH
#define IR_CODE_HH

#include <code_gen/CG_outputRepr.h>
#include <code_gen/CG_outputBuilder.h>
#include <vector>

enum IR_OPERATION_TYPE    {IR_OP_CONSTANT, IR_OP_VARIABLE,
                           IR_OP_PLUS, IR_OP_MINUS, IR_OP_MULTIPLY, IR_OP_DIVIDE,
                           IR_OP_POSITIVE, IR_OP_NEGATIVE,
                           IR_OP_MIN, IR_OP_MAX,
                           IR_OP_ASSIGNMENT,
                           IR_OP_NULL, IR_OP_UNKNOWN};
enum IR_CONTROL_TYPE      {IR_CONTROL_LOOP, IR_CONTROL_IF, IR_CONTROL_WHILE, IR_CONTROL_BLOCK};
enum IR_CONSTANT_TYPE     {IR_CONSTANT_INT, IR_CONSTANT_FLOAT,
                           IR_CONSTANT_UNKNOWN};
enum IR_CONDITION_TYPE    {IR_COND_LT, IR_COND_LE,
                           IR_COND_GT, IR_COND_GE,
                           IR_COND_EQ, IR_COND_NE,
                           IR_COND_UNKNOWN};
enum IR_ARRAY_LAYOUT_TYPE {IR_ARRAY_LAYOUT_ROW_MAJOR,
                           IR_ARRAY_LAYOUT_COLUMN_MAJOR,
                           IR_ARRAY_LAYOUT_SPACE_FILLING};

class IR_Code;


// Base abstract class for scalar and array symbols.  This is a place
// holder for related declaration in IR code.
struct IR_Symbol {
  const IR_Code *ir_;
  
  virtual ~IR_Symbol() {/* ir_ is not the responsibility of this object */}
  virtual int n_dim() const = 0;
  virtual std::string name() const = 0;
  virtual bool operator==(const IR_Symbol &that) const = 0;
  virtual bool operator!=(const IR_Symbol &that) const {return !(*this == that);}
  virtual IR_Symbol *clone() const = 0;  /* shallow copy */
};


struct IR_ScalarSymbol: public IR_Symbol {
  virtual ~IR_ScalarSymbol() {}
  int n_dim() const {return 0;}
  virtual int size() const = 0;
};


struct IR_ArraySymbol: public IR_Symbol {
  virtual ~IR_ArraySymbol() {}
  virtual int elem_size() const = 0;
  virtual omega::CG_outputRepr *size(int dim) const = 0;
  virtual IR_ARRAY_LAYOUT_TYPE layout_type() const = 0;
};


// Base abstract class for scalar and array references.  This is a
// place holder for related code in IR code.
struct IR_Ref {
  const IR_Code *ir_;
  
  virtual ~IR_Ref() {/* ir_ is not the responsibility of this object */}
  virtual int n_dim() const = 0;
  virtual bool is_write() const = 0;
  virtual std::string name() const = 0;
  virtual bool operator==(const IR_Ref &that) const = 0;
  virtual bool operator!=(const IR_Ref &that) const {return !(*this == that);}
  virtual omega::CG_outputRepr *convert() = 0;
  virtual IR_Ref *clone() const = 0;  /* shallow copy */
};


struct IR_ConstantRef: public IR_Ref {
  IR_CONSTANT_TYPE type_;

  virtual ~IR_ConstantRef() {}
  int n_dim() const {return 0;}
  bool is_write() const {return false;}
  std::string name() const {return std::string();}
  virtual bool is_integer() const {return type_ == IR_CONSTANT_INT;}
  virtual omega::coef_t integer() const = 0;
};
  

struct IR_ScalarRef: public IR_Ref {
  virtual ~IR_ScalarRef() {}
  int n_dim() const {return 0;}
  virtual IR_ScalarSymbol *symbol() const = 0;
  std::string name() const {
    IR_ScalarSymbol *sym = symbol();
    std::string s = sym->name();
    delete sym;
    return s;
  }
  virtual int size() const {
    IR_ScalarSymbol *sym = symbol();
    int s = sym->size();
    delete sym;
    return s;
  }
};


struct IR_ArrayRef: public IR_Ref {
  virtual ~IR_ArrayRef() {}
  int n_dim() const {
    IR_ArraySymbol *sym = symbol();
    int n = sym->n_dim();
    delete sym;
    return n;
  }
  virtual omega::CG_outputRepr *index(int dim) const = 0;
  virtual IR_ArraySymbol *symbol() const = 0;
  std::string name() const {
    IR_ArraySymbol *sym = symbol();
    std::string s = sym->name();
    delete sym;
    return s;
  }
  virtual int elem_size() const {
    IR_ArraySymbol *sym = symbol();
    int s = sym->elem_size();
    delete sym;
    return s;
  }
  virtual IR_ARRAY_LAYOUT_TYPE layout_type() const {
    IR_ArraySymbol *sym = symbol();
    IR_ARRAY_LAYOUT_TYPE t = sym->layout_type();
    delete sym;
    return t;
  }
};


struct IR_Block;

// Base abstract class for code structures.  This is a place holder
// for the actual structure in the IR code.  However, in cases that
// original source code may be transformed during loop initialization
// such as converting a while loop to a for loop or reconstructing the
// loop from low level IR code, the helper loop class (NOT
// IMPLEMENTED) must contain the transformed code that needs to be
// freed when out of service.
struct IR_Control {
  const IR_Code *ir_;

  virtual ~IR_Control() {/* ir_ is not the responsibility of this object */}
  virtual IR_CONTROL_TYPE type() const = 0;
  virtual IR_Block *convert() = 0;
  virtual IR_Control *clone() const = 0;  /* shallow copy */
};


struct IR_Loop: public IR_Control {  
  virtual ~IR_Loop() {}
  virtual IR_ScalarSymbol *index() const = 0;
  virtual omega::CG_outputRepr *lower_bound() const = 0;
  virtual omega::CG_outputRepr *upper_bound() const = 0;
  virtual IR_CONDITION_TYPE stop_cond() const = 0;
  virtual IR_Block *body() const = 0;
  virtual int step_size() const = 0;
  IR_CONTROL_TYPE type() const { return IR_CONTROL_LOOP; }
};


struct IR_Block: public IR_Control {
  virtual ~IR_Block() {}
  virtual omega::CG_outputRepr *extract() const = 0;
  IR_Block *convert() {return this;}
  IR_CONTROL_TYPE type() const { return IR_CONTROL_BLOCK; }
  virtual omega::CG_outputRepr *original() const = 0;
};


struct IR_If: public IR_Control {
  virtual ~IR_If() {}
  virtual omega::CG_outputRepr *condition() const = 0;
  virtual IR_Block *then_body() const = 0;
  virtual IR_Block *else_body() const = 0;
  IR_CONTROL_TYPE type() const { return IR_CONTROL_IF; }
};


  
struct IR_While: public IR_Control {
  // NOT IMPLEMENTED
};


// Abstract class for compiler IR.
class IR_Code {
protected:
  omega::CG_outputBuilder *ocg_;
  omega::CG_outputRepr *init_code_;
  omega::CG_outputRepr *cleanup_code_;

public:
  IR_Code() {ocg_ = NULL; init_code_ = cleanup_code_ = NULL;}
  virtual ~IR_Code() { delete ocg_; delete init_code_; delete cleanup_code_; } /* the content of init and cleanup code have already been released in derived classes */
  
  // memory_type is for differentiating the location of where the new memory is allocated.
  // this is useful for processors with heterogeneous memory hierarchy.
  virtual IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int memory_type) = 0;
  virtual IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int memory_type) = 0;
  
  virtual IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym) = 0;
  virtual IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index) = 0;
  virtual int ArrayIndexStartAt() {return 0;}

  // Array references should be returned in their accessing order.
  // e.g. s1: A[i] = A[i-1]
  //      s2: B[C[i]] = D[i] + E[i]
  // return A[i-1], A[i], D[i], E[i], C[i], B[C[i]] in this order.
  virtual std::vector<IR_ArrayRef *> FindArrayRef(const omega::CG_outputRepr *repr) const = 0;
  virtual std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const = 0;

  // If there is no sub structure interesting inside the block, return empty,
  // so we know when to stop looking inside.
  virtual std::vector<IR_Control *> FindOneLevelControlStructure(const IR_Block *block) const = 0;

  // All controls must be in the same block, at the same level and in
  // contiguous lexical order as appeared in parameter vector.
  virtual IR_Block *MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const = 0;
  
  virtual IR_Block *GetCode() const = 0;
  virtual void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr) = 0;
  virtual void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr) = 0;
  
  virtual IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const = 0;
  virtual IR_CONDITION_TYPE QueryBooleanExpOperation(const omega::CG_outputRepr *repr) const = 0;
  virtual std::vector<omega::CG_outputRepr *> QueryExpOperand(const omega::CG_outputRepr *repr) const = 0;
  virtual IR_Ref *Repr2Ref(const omega::CG_outputRepr *repr) const = 0;
  
  //---------------------------------------------------------------------------
  // CC Omega code builder interface here
  //---------------------------------------------------------------------------
  omega::CG_outputBuilder *builder() const {return ocg_;}

};

#endif  
  
