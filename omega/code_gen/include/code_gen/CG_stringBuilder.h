#ifndef _CG_STRINGBUILDER_H
#define _CG_STRINGBUILDER_H

#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_stringRepr.h>

namespace omega {

class CG_stringBuilder: public CG_outputBuilder { 
public:

  CG_stringBuilder() {}
  ~CG_stringBuilder() {}
  bool isInteger(CG_outputRepr *op) const;
  bool QueryInspectorType(const std::string &varName) const;

  CG_stringRepr *CreateInt(int num) const;
  CG_stringRepr *CreateFloat(float num) const;
  CG_stringRepr *CreateDouble(double num) const;

  CG_stringRepr *CreateSubstitutedStmt(int indent, CG_outputRepr *stmt, const std::vector<std::string> &vars, std::vector<CG_outputRepr *> &subs, bool actuallyPrint) const;
  CG_stringRepr *CreateAssignment(int indent, CG_outputRepr *lhs, CG_outputRepr *rhs) const;
  CG_stringRepr *CreatePlusAssignment(int indent, CG_outputRepr *lhs, CG_outputRepr *rhs) const;
  CG_stringRepr *CreateInvoke(const std::string &funcName, std::vector<CG_outputRepr *> &argList,bool is_array=false) const;
  CG_stringRepr *CreateComment(int indent, const std::string &commentText) const;
  CG_stringRepr* CreateAttribute(CG_outputRepr *control,
                                          const std::string &commentText) const;
  CG_outputRepr *CreatePragmaAttribute(CG_outputRepr *scopeStmt, int looplevel, const std::string &pragmaText) const;
  CG_outputRepr *CreatePrefetchAttribute(CG_outputRepr *scopeStmt, int looplevel, const std::string &arrName, int hint) const;
  CG_stringRepr* CreateNullStatement() const;
  CG_stringRepr *CreateIf(int indent, CG_outputRepr *guardCondition, CG_outputRepr *true_stmtList, CG_outputRepr *false_stmtList) const;
  CG_stringRepr *CreateInductive(CG_outputRepr *index, CG_outputRepr *lower, CG_outputRepr *upper, CG_outputRepr *step) const;
  CG_stringRepr *CreateLoop(int indent, CG_outputRepr *control, CG_outputRepr *stmtList) const;

  CG_stringRepr *CreateAddressOf(CG_outputRepr *op) const ;
  CG_stringRepr *CreateIdent(const std::string &varName) const;
  CG_stringRepr *CreateDotExpression(CG_outputRepr *lop,
  		CG_outputRepr *rop) const;
  CG_stringRepr *CreateArrayRefExpression(const std::string &_s,
  		CG_outputRepr *rop) const;
  CG_stringRepr* CreateArrayRefExpression(CG_outputRepr *lop,
  		CG_outputRepr *rop) const;
  CG_stringRepr *ObtainInspectorData(const std::string &_s, const std::string &member_name) const;
  CG_stringRepr *CreatePlus(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateMinus(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateTimes(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateDivide(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateIntegerFloor(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateIntegerMod(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateIntegerCeil(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateAnd(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateGE(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateLE(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateEQ(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateNEQ(CG_outputRepr *lop, CG_outputRepr *rop) const;
  CG_stringRepr *CreateBreakStatement(void) const;
  CG_stringRepr *StmtListAppend(CG_outputRepr *list1, CG_outputRepr *list2) const;
  CG_stringRepr *CreateStatementFromExpression(CG_outputRepr *exp) const;

  CG_outputRepr *CreateStruct(const std::string struct_name,
                              std::vector<std::string> data_members,
                              std::vector<CG_outputRepr *> data_types);

  CG_outputRepr *CreateClassInstance(std::string name , CG_outputRepr *class_def);
	CG_outputRepr *lookup_member_data(CG_outputRepr* scope, std::string varName, CG_outputRepr *instance);
  const char *ClassName() { return "stringBuilder"; }; 
  CG_outputRepr* CreatePointer(std::string  &name) const;
	CG_outputRepr* ObtainInspectorRange(const std::string &_s, const std::string &_name) const;

};


}

#endif
