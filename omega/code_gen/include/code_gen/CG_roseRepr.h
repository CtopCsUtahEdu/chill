#ifndef CG_roseRepr_h
#define CG_roseRepr_h

#include <code_gen/CG_outputRepr.h>
#include "rose.h"

namespace omega {

class CG_roseRepr : public CG_outputRepr {
  friend class CG_roseBuilder;
public:
  CG_roseRepr();
  CG_roseRepr(SgNode *tnl);
  CG_roseRepr(SgExpression *exp);
  CG_roseRepr(SgStatementPtrList* stmtlist);

  ~CG_roseRepr();
  CG_outputRepr *clone() const;
  void clear();

  SgNode* GetCode() const;
  SgStatementPtrList* GetList() const;
  SgExpression *GetExpression() const;




  //---------------------------------------------------------------------------
  // Dump operations
  //---------------------------------------------------------------------------
  void Dump() const;
  //void DumpToFile(FILE *fp = stderr) const;
private:
  // only one of _tnl and _op would be active at any time, depending on
  // whether it is building a statement list or an expression tree
  SgNode  *tnl_;
  SgExpression  *op_;
  SgStatementPtrList *list_;
  void DumpFileHelper(SgNode* node, FILE* fp) const; 
  //operand op_;
};



} // namespace

#endif
