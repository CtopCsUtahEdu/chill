/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   abstract base class of compiler IR code builder

 Notes:
   All "CG_outputRepr *" parameters are consumed inside the the function
 unless explicitly stated otherwise, i.e., not valid after the call.
   Parameter "indent" normally not used except it is used in unstructured
 string output for correct indentation.
 
 History:
   04/17/96 created - Lei Zhou
   05/02/08 clarify integer floor/mod/ceil definitions, -chen
   05/31/08 use virtual clone to implement CreateCopy, -chun
   08/05/10 clarify NULL parameter allowance, -chun
*****************************************************************************/

#ifndef _CG_OUTPUTBUILDER_H
#define _CG_OUTPUTBUILDER_H

#include <code_gen/CG_outputRepr.h>

#include <string>
#include <vector>

namespace omega {

class CG_outputBuilder {
private:

public:

  CG_outputBuilder() {}
  virtual ~CG_outputBuilder() {}


  virtual CG_outputRepr *CreateStruct(const std::string class_name,
                                      std::vector<std::string> class_data_members,  // why not just vector< CG_outputRepr> subparts;
                                      std::vector<CG_outputRepr *> class_data_types )=0;
  virtual CG_outputRepr *CreateClassInstance(std::string name , CG_outputRepr *class_def)=0;
	virtual CG_outputRepr *lookup_member_data(CG_outputRepr* scope, std::string varName, CG_outputRepr *instance)=0;
  virtual CG_outputRepr* CreatePointer(std::string  &name) const = 0;
	virtual CG_outputRepr* ObtainInspectorRange(const std::string &_s, const std::string &_name) const=0;



  //---------------------------------------------------------------------------
  // substitute variables in stmt
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateSubstitutedStmt(int indent, 
                                               CG_outputRepr *stmt,
                                               const std::vector<std::string> &vars,
                                               std::vector<CG_outputRepr *> &subs, 
                                               bool actuallyPrint =true) const = 0;

  //---------------------------------------------------------------------------
  // assignment stmt generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateAssignment(int indent, 
                                          CG_outputRepr *lhs,
                                          CG_outputRepr *rhs) const = 0;

  virtual CG_outputRepr *CreatePlusAssignment(int indent, 
                                              CG_outputRepr *lhs,
                                              CG_outputRepr *rhs) const = 0;

  //---------------------------------------------------------------------------
  // function invocation generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateInvoke(const std::string &funcName,
                                      std::vector<CG_outputRepr *> &argList,
                                      bool is_array=false) const = 0;

  //---------------------------------------------------------------------------
  // comment generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateComment(int indent,
                                       const std::string &commentText) const = 0;

  //---------------------------------------------------------------------------
  // Attribute generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr* CreateAttribute(CG_outputRepr  *control,
                                           const std::string &commentText) const = 0;
  //---------------------------------------------------------------------------
  // Pragma Attribute
  // --------------------------------------------------------------------------
  virtual CG_outputRepr* CreatePragmaAttribute(CG_outputRepr *scopeStmt, int looplevel, const std::string &pragmaText) const = 0;
  
  //---------------------------------------------------------------------------
  // Prefetch Attribute
  //---------------------------------------------------------------------------
  virtual CG_outputRepr* CreatePrefetchAttribute(CG_outputRepr *scopeStmt, int looplevel, const std::string &arrName, int hint) const = 0;

  //---------------------------------------------------------------------------
  // generate if stmt, true/false stmt can be NULL but not the condition
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateIf(int indent, CG_outputRepr *guardCondition,
                                  CG_outputRepr *true_stmtList,
                                  CG_outputRepr *false_stmtList) const = 0;

  //---------------------------------------------------------------------------
  // generate loop inductive variable (loop control structure)
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateInductive(CG_outputRepr *index,
                                         CG_outputRepr *lower,
                                         CG_outputRepr *upper,
                                         CG_outputRepr *step) const = 0;

  //---------------------------------------------------------------------------
  // generate loop stmt from loop control and loop body, NULL parameter allowed
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateLoop(int indent, CG_outputRepr *control,
                                    CG_outputRepr *stmtList) const = 0;

  //---------------------------------------------------------------------------
  // copy operation, NULL parameter allowed. this function makes pointer
  // handling uniform regardless NULL status
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateCopy(CG_outputRepr *original) const {
    if (original == NULL)
      return NULL;
    else
      return original->clone();
  }

  //---------------------------------------------------------------------------
  // basic integer number creation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateInt(int num) const = 0;
  virtual CG_outputRepr *CreateFloat(float num) const = 0;
  virtual CG_outputRepr *CreateDouble(double num) const = 0;

  virtual bool isInteger(CG_outputRepr *op) const = 0;
  virtual bool QueryInspectorType(const std::string &varName) const = 0;

  //---------------------------------------------------------------------------
  // basic identity/variable creation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateIdent(const std::string &varName) const = 0;
  virtual CG_outputRepr* CreateDotExpression(CG_outputRepr *lop,
  		CG_outputRepr *rop) const =0;
  virtual CG_outputRepr* CreateArrayRefExpression(const std::string &_s,
  		CG_outputRepr *rop) const =0;
  virtual CG_outputRepr* CreateArrayRefExpression(CG_outputRepr *lop,
  		CG_outputRepr *rop) const=0;
  virtual CG_outputRepr* ObtainInspectorData(const std::string &_s, const std::string &member_name) const=0;
  virtual CG_outputRepr* CreateNullStatement() const=0;

  //---------------------------------------------------------------------------
  // binary arithmetic operations, NULL parameter means 0,
  // Note:
  //   integer division truncation method undefined, only use when lop is known
  //   to be multiple of rop, otherwise use integer floor instead
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreatePlus(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateMinus(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateTimes(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateDivide(CG_outputRepr *lop, CG_outputRepr *rop) const {
    return CreateIntegerFloor(lop, rop);
  }
  
  //---------------------------------------------------------------------------
  // integer arithmetic functions, NULL parameter means 0, second parameter
  // must be postive (i.e. b > 0 below), otherwise function undefined
  // Note:
  //   ceil(a, b) = -floor(-a, b) or floor(a+b-1, b) or floor(a-1, b)+1
  //   mod(a, b) = a-b*floor(a, b) 
  //     where result must lie in range [0,b)
  //   floor(a, b) = a/b if a >= 0
  //                 (a-b+1)/b if a < 0
  //     where native '/' operator behaves as 5/2 = 2, (-5)/2 = -2
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateIntegerFloor(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateIntegerMod(CG_outputRepr *lop, CG_outputRepr *rop) const {
    CG_outputRepr *lop2 = CreateCopy(lop);
    CG_outputRepr *rop2 = CreateCopy(rop);
    return CreateMinus(lop2, CreateTimes(rop2, CreateIntegerFloor(lop, rop)));
  }
  virtual CG_outputRepr *CreateIntegerCeil(CG_outputRepr *lop, CG_outputRepr *rop) const {
    return CreateMinus(NULL, CreateIntegerFloor(CreateMinus(NULL, lop), rop));
  }

  //---------------------------------------------------------------------------
  // binary logical operation, NULL parameter means TRUE
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateAnd(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;

  //---------------------------------------------------------------------------
  // binary condition operations
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *CreateGE(CG_outputRepr *lop, CG_outputRepr *rop) const {
    return CreateLE(rop, lop);
  } 
  virtual CG_outputRepr *CreateLE(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateEQ(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateNEQ(CG_outputRepr *lop, CG_outputRepr *rop) const = 0;
  virtual CG_outputRepr *CreateAddressOf(CG_outputRepr *op) const = 0;
  virtual CG_outputRepr *CreateBreakStatement(void) const = 0;
 
  //---------------------------------------------------------------------------
  // join stmts together, NULL parameter allowed
  //---------------------------------------------------------------------------
  virtual CG_outputRepr *StmtListAppend(CG_outputRepr *list1, CG_outputRepr *list2) const = 0;
  virtual CG_outputRepr *CreateStatementFromExpression(CG_outputRepr *exp) const = 0;

  virtual const char *ClassName() { return "UNKNOWN"; } 
};

}

#endif
