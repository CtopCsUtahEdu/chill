#ifndef CG_suifRepr_h
#define CG_suifRepr_h

#include <code_gen/CG_outputRepr.h>
#include <suif1.h>

namespace omega {

class CG_suifRepr : public CG_outputRepr {
  friend class CG_suifBuilder;
public:
  CG_suifRepr();
  CG_suifRepr(tree_node_list *tnl);
  CG_suifRepr(operand op);
  virtual ~CG_suifRepr();
  virtual CG_outputRepr *clone();
  virtual void clear();

  tree_node_list* GetCode() const;
  operand GetExpression() const;

  //---------------------------------------------------------------------------
  // Dump operations
  //---------------------------------------------------------------------------
  virtual void Dump() const;
  virtual void DumpToFile(FILE *fp = stderr) const;
private:
  // only one of _tnl and _op would be active at any time, depending on
  // whether it is building a statement list or an expression tree
  tree_node_list *tnl_;
  operand op_;
};

} // namespace

#endif
