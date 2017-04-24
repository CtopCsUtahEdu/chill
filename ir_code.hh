/*****************************************************************************
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's compiler intermediate representation interface that extends
 Omega's builder interface to accomodate compiler analyses and
 extra code generation.
.
 Notes:
   Unlike CG_outputRepr, IR_Symbol, IR_Ref and IR_Control are place holders
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

#include <ir_enums.hh>

#include <chill_ast.hh>
#include "chill_io.hh"

#include <vector>

// needed for omega::coef_t below
#include <basic/util.h>   // in omega ... why is this not in CG_output*.h?

#include <code_gen/CG_outputRepr.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_stringBuilder.h>

/*!
 * \file
 * \brief CHiLL's compiler intermediate representation interface that extends Omega's builder interface to accomodate compiler analyses and extra code generation.
 *
 * Unlike CG_outputRepr, IR_Symbol,IR_Ref and IR_Control are place holders
 * to the underlying code, thus deleting or duplicating them does not affect
 * the actual code.  Similar to Omega builder's memory allocation strategy,
 * all non-const pointer parameters of CG_outputRepr/IR_Symbol/IR_Ref/IR_Control
 * are destroyed after the call.
 */

class IR_Code; // forward declaration

/*!
 * @brief Base abstract class for scalar and array symbols.
 *
 * This is a place holder for related declaration in IR code.
 */
struct IR_Symbol {
  const IR_Code *ir_;

  //! ir_ is not the responsibility of this object
  virtual ~IR_Symbol() { }

  virtual int n_dim() const = 0;
  virtual std::string name() const = 0;
  virtual bool operator==(const IR_Symbol &that) const = 0;
  virtual bool operator!=(const IR_Symbol &that) const {return !(*this == that);}

  //! shallow copy
  virtual IR_Symbol *clone() const = 0;  /* shallow copy */

  virtual bool isScalar()  const { return false; } // default
  virtual bool isArray()   const { return false; } // default
  virtual bool isPointer() const { return false; } // default

  //IR_SYMBOL_TYPE symtype; // base type: int, float, double, struct, .... typedef'd something
  //IR_SYMBOL_TYPE getDatatype() ; 
};


struct IR_ScalarSymbol: public IR_Symbol {
  virtual ~IR_ScalarSymbol() {}
  int n_dim() const {return 0;} // IR_ScalarSymbol
  virtual int size() const = 0;
  bool isScalar() const { return true; }
};

struct IR_FunctionSymbol: public IR_Symbol {
  virtual ~IR_FunctionSymbol() {}
  int n_dim() const { return 0; }
};

struct IR_ArraySymbol: public IR_Symbol {
  virtual ~IR_ArraySymbol() {}
  virtual int elem_size() const = 0;
  virtual omega::CG_outputRepr *size(int dim) const = 0;
  virtual IR_ARRAY_LAYOUT_TYPE layout_type() const = 0;
  virtual IR_CONSTANT_TYPE elem_type() const = 0;
  bool isArray() const { return true; }
};


struct IR_PointerSymbol: public IR_Symbol {
  virtual ~IR_PointerSymbol() {}
  virtual omega::CG_outputRepr *size(int dim) const = 0;
  virtual void set_size(int dim, omega::CG_outputRepr*) = 0;
  virtual IR_CONSTANT_TYPE elem_type() const = 0;
  bool isPointer() const { return true; }
};


/*!
 * @brief Base abstract class for scalar and array references.
 *
 * This is a place holder for related code in IR code.
 */
struct IR_Ref {
  const IR_Code *ir_;
  //! ir_ is not the responsibility of this object
  virtual ~IR_Ref() { }

  virtual int n_dim() const = 0;
  virtual bool is_write() const = 0;
  virtual std::string name() const = 0;
  virtual bool operator==(const IR_Ref &that) const = 0;
  virtual bool operator!=(const IR_Ref &that) const {return !(*this == that);}
  virtual omega::CG_outputRepr *convert() = 0;
  //! shallow copy
  virtual IR_Ref *clone() const = 0;  /* shallow copy */
  virtual void Dump() const { debug_fprintf(stderr, "some IR_*Ref needs to implement Dump()\n"); int *i=0; int j=i[0]; };
};


struct IR_ConstantRef: public IR_Ref {
  IR_CONSTANT_TYPE type_;

  virtual ~IR_ConstantRef() {}
  int n_dim() const {return 0;} // IR_ConstantRef
  bool is_write() const {return false;}
  std::string name() const {return std::string();}
  virtual bool is_integer() const {return type_ == IR_CONSTANT_INT;}
  virtual omega::coef_t integer() const = 0;
};
  

struct IR_ScalarRef: public IR_Ref {
  virtual ~IR_ScalarRef() {}
  int n_dim() const {return 0;} // IR_ScalarRef
  virtual IR_ScalarSymbol *symbol() const = 0;
  std::string name() const {
    IR_ScalarSymbol *sym = symbol(); // really inefficient. MAKE a symbol, just to get a name
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

struct IR_FunctionRef: public IR_Ref {
  virtual ~IR_FunctionRef() {}
  int n_dim() const { return 0; }
  virtual IR_FunctionSymbol *symbol() const = 0;
  std::string name() const {
    IR_FunctionSymbol *sym = symbol();
    std::string s = sym->name();
    delete sym;
    return s;
  }
};

struct IR_ArrayRef: public IR_Ref {
  virtual ~IR_ArrayRef() {}
  int n_dim() const {  // IR_ArrayRef  returns the size of the array 
    IR_ArraySymbol *sym = symbol();
    int n = sym->n_dim();
    // ?? delete sym;
    return n;
  }
  virtual omega::CG_outputRepr *index(int dim) const = 0;
  virtual IR_ArraySymbol *symbol() const = 0;
  virtual std::string name() const {
    // makes (constructs!) a symbol, just to copy a string!
    IR_ArraySymbol *sym = symbol(); // TODO exceedingly wasteful 
    std::string s = sym->name();
    // ?? delete sym; (goes out of scope, so deletes itself)
    return s; // s ALSO goes out of scope but perhaps the info is copied at the other end
  }
  virtual int elem_size() const {
    IR_ArraySymbol *sym = symbol();
    int s = sym->elem_size();
    // ?? delete sym;
    return s;
  }
  virtual IR_ARRAY_LAYOUT_TYPE layout_type() const {
    IR_ArraySymbol *sym = symbol();
    IR_ARRAY_LAYOUT_TYPE t = sym->layout_type();
    // ?? delete sym;
    return t;
  }
  virtual void Dump() const { debug_fprintf(stderr, "IR_ArrayRef needs to implement Dump()\n"); };
};

struct IR_PointerArrayRef: public IR_Ref {

  const IR_Code *ir_;

  virtual ~IR_PointerArrayRef() {}
  int n_dim() const {  // IR_PointerArrayRef returns size of the ... symbol?
    IR_PointerSymbol *sym = symbol();
    int n = sym->n_dim();
    //Anand: Hack, fix later
    //delete sym;
    return n;
  }
  virtual omega::CG_outputRepr *index(int dim) const = 0;
  virtual IR_PointerSymbol *symbol() const = 0;
  std::string name() const {
    IR_PointerSymbol *sym = symbol();
    std::string s = sym->name();
//Anand: Hack, fix later
    //delete sym;
    return s;
  }


};


struct IR_Block;

//! Base abstract class for code structures.
/*!
 * This is a place holder for the actual structure in the IR code.
 * However, in cases that original source code may be transformed during
 * loop initialization such as converting a while loop to a for loop or
 * reconstructing the loop from low level IR code, the helper loop class (NOT
 * IMPLEMENTED) must contain the transformed code that needs to be
 * freed when out of service.
 */
struct IR_Control {
  const IR_Code *ir_; // hate this

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
  // the only data members in IR_Code are Omega classes
  omega::CG_outputBuilder *ocg_;    // Omega Code Gen
  // TODO does stringBuilder have internal state that needs to be global?
  omega::CG_stringBuilder ocgs;
  omega::CG_outputRepr *init_code_;
  omega::CG_outputRepr *cleanup_code_;

  // OK, I lied
  static int ir_pointer_counter;
  static int ir_array_counter;

public:

  int getPointerCounter() { return ir_pointer_counter; }
  int getArrayCounter()   { return ir_array_counter; }

  // TODO can't get the initialize of counters to work !! 
  int getAndIncrementPointerCounter() { if (ir_pointer_counter == 0) ir_pointer_counter= 1; ir_pointer_counter++;  return ir_pointer_counter-1; }
  int getAndIncrementArrayCounter()   { if (ir_array_counter == 0) ir_array_counter= 1; ir_array_counter += 1;   return ir_array_counter-1; }

  // if all flavors of ir_code use chillAST internally ... 
  chillAST_FunctionDecl * func_defn;     // the function we're modifying
  chillAST_FunctionDecl *GetChillFuncDefinition() { return func_defn; }; 

  IR_Code() {ocg_ = NULL; init_code_ = cleanup_code_ = NULL;}
  virtual ~IR_Code() { delete ocg_; delete init_code_; delete cleanup_code_; } /* the content of init and cleanup code have already been released in derived classes */
  
  omega::CG_outputRepr* init_code(){ return init_code_; }
  virtual omega::CG_outputRepr *RetrieveMacro(std::string s) = 0;
  /*!
   * \param memory_type is for differentiating the location of
   *    where the new memory is allocated. this is useful for
   *    processors with heterogeneous memory hierarchy.
   */
  virtual IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int memory_type) = 0;
  virtual IR_ScalarSymbol *CreateScalarSymbol(IR_CONSTANT_TYPE type, int memory_type, std::string name="" ) =0;

  virtual IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, 
                                            std::vector<omega::CG_outputRepr *> &size, 
                                            int memory_type) = 0;
  virtual IR_ArraySymbol *CreateArraySymbol(omega::CG_outputRepr *type,
                                            std::vector<omega::CG_outputRepr *> &size_repr) =0;

  virtual IR_PointerSymbol *CreatePointerSymbol(const IR_Symbol *sym,
                                                std::vector<omega::CG_outputRepr *> &size_repr) =0;
  virtual IR_PointerSymbol *CreatePointerSymbol(const IR_CONSTANT_TYPE type,
                                                std::vector<omega::CG_outputRepr *> &size_repr, 
                                                std::string name="") =0;

  virtual IR_PointerSymbol *CreatePointerSymbol(omega::CG_outputRepr *type,
                                                std::vector<omega::CG_outputRepr *> &size_repr) =0;

  virtual IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym) = 0;
  virtual IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index) = 0;

  virtual omega::CG_outputRepr* CreateArrayRefRepr(const IR_ArraySymbol *sym,
                                            std::vector<omega::CG_outputRepr *> &index) {
    //IR_ArrayRef *AR = CreateArrayRef(sym, index);
    //return new omega::CG_outputRepr(AR);
    debug_fprintf(stderr, "ir_code.hh  SOME SUBCLASS OF ir_code did not implement CreateArrayRefRepr()\n"); 
    return NULL; 
  }

  virtual IR_PointerArrayRef *CreatePointerArrayRef( IR_PointerSymbol *sym,
      std::vector<omega::CG_outputRepr *> &index) =0;
  virtual int ArrayIndexStartAt() {return 0;}

  virtual void CreateDefineMacro(std::string s,std::string args,  omega::CG_outputRepr *repr) = 0;
  virtual void CreateDefineMacro(std::string s,std::string args, std::string repr) = 0;
  virtual void CreateDefineMacro(std::string s, std::vector<std::string> args,omega::CG_outputRepr *repr) {};  // TODO make pure virtual  

  virtual omega::CG_outputRepr *CreateArrayType(IR_CONSTANT_TYPE type, omega::CG_outputRepr* size)=0;
  virtual omega::CG_outputRepr *CreatePointerType(IR_CONSTANT_TYPE type)=0;
  virtual omega::CG_outputRepr *CreatePointerType(omega::CG_outputRepr *type)=0;
  virtual omega::CG_outputRepr *CreateScalarType(IR_CONSTANT_TYPE type)=0;
  /*!
  * Array references should be returned in their accessing order.
  *
  * ~~~
  * e.g. s1: A[i] = A[i-1]
  *      s2: B[C[i]] = D[i] + E[i]
  * return A[i-1], A[i], D[i], E[i], C[i], B[C[i]] in this order.
  * ~~~
  */
  virtual std::vector<IR_ArrayRef *> FindArrayRef(const omega::CG_outputRepr *repr) const = 0;
  virtual std::vector<IR_PointerArrayRef *> FindPointerArrayRef(const omega::CG_outputRepr *repr) const = 0 ;
  virtual std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const = 0;
  virtual std::vector<IR_Loop *> FindLoops(omega::CG_outputRepr *repr)= 0;
  virtual bool parent_is_array(IR_ArrayRef *a)=0;

  // If there is no sub structure interesting inside the block, return empty,
  // so we know when to stop looking inside.
  virtual std::vector<IR_Control *> FindOneLevelControlStructure(const IR_Block *block) const = 0;

  /*!
   * All controls must be in the same block, at the same level and in
   * contiguous lexical order as appeared in parameter vector.
   */
  virtual IR_Block *MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const = 0;

  virtual IR_Block *GetCode() const = 0;
  virtual IR_Control *GetCode(omega::CG_outputRepr *code) const = 0;
  virtual void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr) = 0;
  virtual void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr) = 0;
  
  virtual IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const = 0;
  virtual IR_CONDITION_TYPE QueryBooleanExpOperation(const omega::CG_outputRepr *repr) const = 0;
  virtual std::vector<omega::CG_outputRepr *> QueryExpOperand(const omega::CG_outputRepr *repr) const = 0;
  virtual IR_Ref *Repr2Ref(const omega::CG_outputRepr *repr) const = 0;

  // Manu:: Added functions required for reduction operation
 // virtual omega::CG_outputRepr * FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B) = 0;
  virtual bool FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B) = 0;
  virtual void printStmt(const omega::CG_outputRepr *repr) = 0;
  virtual int getStmtType(const omega::CG_outputRepr *repr) = 0;
  virtual IR_OPERATION_TYPE getReductionOp(const omega::CG_outputRepr *repr) = 0;
  virtual IR_Control *  FromForStmt(const omega::CG_outputRepr *repr) = 0;

  // Manu:: Added functions for scalar expansion
  virtual IR_ArraySymbol *CreateArraySymbol(omega::CG_outputRepr *size, const IR_Symbol *sym) = 0;
  virtual bool ReplaceRHSExpression(omega::CG_outputRepr *code, IR_Ref *ref) = 0;
  virtual omega::CG_outputRepr * GetRHSExpression(omega::CG_outputRepr *code) = 0;
  virtual omega::CG_outputRepr * GetLHSExpression(omega::CG_outputRepr *code) = 0;
  virtual omega::CG_outputRepr *CreateMalloc(const IR_CONSTANT_TYPE type, std::string lhs,
      omega::CG_outputRepr * size_repr)=0;
  virtual omega::CG_outputRepr *CreateMalloc(omega::CG_outputRepr *type, std::string variable,
      omega::CG_outputRepr * size_repr)=0;
  virtual omega::CG_outputRepr *CreateFree(omega::CG_outputRepr * exp)=0;

  //void Dump() { ocg_->Dump(); }; 
  //---------------------------------------------------------------------------
  // CC Omega code builder interface here

  //---------------------------------------------------------------------------
  omega::CG_outputBuilder *builder() const { return ocg_;}
  omega::CG_stringBuilder builder_s() const { return ocgs; }

};

#endif  
  
