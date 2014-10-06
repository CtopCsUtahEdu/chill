#ifndef CG_suifBuilder_h
#define CG_suifBuilder_h

#include <basic/Tuple.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_suifRepr.h>
#include <string>

namespace omega {


class CG_suifBuilder: public CG_outputBuilder { 
public:
  //CG_suifBuilder(proc_symtab *symtab) {symtab_ = symtab;}
  CG_suifBuilder(proc_symtab *symtab); 
  //protonu--initializing code_gen stuff for cuda
  //this looks like a flaw in my design	
  //end--protonu
  ~CG_suifBuilder() {}

  CG_outputRepr* CreatePlaceHolder(int indent, CG_outputRepr *stmt,
                                           Tuple<CG_outputRepr*> &funcList,
                                           Tuple<std::string> &loop_vars) const;
  CG_outputRepr* CreateAssignment(int indent, CG_outputRepr* lhs,
                                  CG_outputRepr* rhs) const;
  CG_outputRepr* CreateInvoke(const std::string &fname,
                              Tuple<CG_outputRepr*> &argList) const;
  CG_outputRepr* CreateComment(int indent, const std::string &commentText) const;
  CG_outputRepr* CreateAttribute(CG_outputRepr *control,
                                          const std::string &commentText) const;

  CG_outputRepr* CreateIf(int indent, CG_outputRepr* guardCondition,
                          CG_outputRepr* true_stmtList, CG_outputRepr* false_stmtList) const;
  CG_outputRepr* CreateInductive(CG_outputRepr* index,
                                 CG_outputRepr* lower,
                                 CG_outputRepr* upper,
                                 CG_outputRepr* step) const;
  CG_outputRepr* CreateLoop(int indent, CG_outputRepr* control,
                            CG_outputRepr* stmtList) const;
  CG_outputRepr* CreateInt(int) const;
  CG_outputRepr* CreateIdent(const std::string &idStr) const;
  CG_outputRepr* CreatePlus(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateMinus(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateTimes(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateIntegerDivide(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateIntegerMod(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateAnd(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateGE(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateLE(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* CreateEQ(CG_outputRepr*, CG_outputRepr*) const;
  CG_outputRepr* StmtListAppend(CG_outputRepr* list1, CG_outputRepr* list2) const;
  //---------------------------------------------------------------------------
  // pragma generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr* CreatePragma(int indent, const std::string &pragmaText) const;

  //---------------------------------------------------------------------------
  // dim3 generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr* CreateDim3(immed varName, immed arg1, immed arg2) const;
  virtual CG_outputRepr* CreateDim3(immed varName, immed arg1, immed arg2, immed arg3) const;


  //---------------------------------------------------------------------------
  // kernel generation
  //---------------------------------------------------------------------------
  virtual CG_outputRepr* CreateKernel(immed_list* iml) const;

  //---------------------------------------------------------------------------
  // Add a modifier to a type (for things like __global__)
  //---------------------------------------------------------------------------
  type_node* ModifyType(type_node* base, const char* modifier) const;
private:
  proc_symtab *symtab_;
};

extern char *k_ocg_comment;

bool substitute(instruction *in, var_sym *sym, operand expr,
                base_symtab *st=NULL);
bool substitute(tree_node *tn, var_sym *sym, operand expr,
                base_symtab *st=NULL);
bool substitute(tree_node_list *tnl, var_sym *sym, operand expr,
                base_symtab *st = NULL);

}

#endif
