#ifndef CG_roseBuilder_h
#define CG_roseBuilder_h

#include <basic/Tuple.h>
#include <code_gen/rose_attributes.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_roseRepr.h>
#include <string>

namespace omega {

class CG_roseBuilder : public CG_outputBuilder { 
public:
  CG_roseBuilder(int isFortran, SgGlobal* global, SgGlobal* global_scope, SgSymbolTable* symtab1, SgSymbolTable* symtab2,  SgNode* root);
  ~CG_roseBuilder();
   
  //---------------------------------------------------------------------------
  // place holder generation
  //---------------------------------------------------------------------------
  // CG_outputRepr* CreatePlaceHolder(int indent, CG_outputRepr *stmt,
  //                                         Tuple<CG_outputRepr*> &funcList,
  //                                         Tuple<std::string> &loop_vars) const;


  //---------------------------------------------------------------------------
  // substitute variables in stmt
  //---------------------------------------------------------------------------


 
   CG_outputRepr *CreateSubstitutedStmt(int indent, CG_outputRepr *stmt,
                                               const std::vector<std::string> &vars,
                                               std::vector<CG_outputRepr *> &subs) const;


  
  //---------------------------------------------------------------------------
  // assignment generation
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateAssignment(int indent, CG_outputRepr* lhs,
                                          CG_outputRepr* rhs) const;

  //---------------------------------------------------------------------------
  // function invocation generation
  //---------------------------------------------------------------------------
    CG_outputRepr* CreateInvoke(const std::string &funcName,
    		std::vector<CG_outputRepr *> &argList) const;
  
  //---------------------------------------------------------------------------
  // comment generation
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateComment(int indent, const std::string &commentText) const;
  //---------------------------------------------------------------------------
  // Attribute generation
  //---------------------------------------------------------------------------
    CG_outputRepr* CreateAttribute(CG_outputRepr  *control,
                                          const std::string &commentText) const;
  //---------------------------------------------------------------------------
  // Pragma Attribute
  //---------------------------------------------------------------------------
  CG_outputRepr* CreatePragmaAttribute(CG_outputRepr *scopeStmt, int looplevel,
                                          const std::string &pragmaText) const;
  
  //---------------------------------------------------------------------------
  // Prefetch Attribute
  //---------------------------------------------------------------------------
  CG_outputRepr* CreatePrefetchAttribute(CG_outputRepr *scopeStmt, int looplevel,
                                          const std::string &arrName, int hint) const;

  //---------------------------------------------------------------------------
  // if stmt gen operations
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateIf(int indent, CG_outputRepr* guardCondition,
                                  CG_outputRepr* true_stmtList, CG_outputRepr* false_stmtList) const;
   
  //---------------------------------------------------------------------------
  // inductive variable generation, to be used in CreateLoop as control
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateInductive(CG_outputRepr* index,
                                         CG_outputRepr* lower,
                                         CG_outputRepr* upper,
                                         CG_outputRepr* step) const;

  //---------------------------------------------------------------------------
  // loop stmt generation
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateLoop(int indent, CG_outputRepr* control,
                                    CG_outputRepr* stmtList) const;

  //---------------------------------------------------------------------------
  // basic operations
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateInt(int num ) const;
   bool isInteger(CG_outputRepr *op) const;
   CG_outputRepr* CreateIdent(const std::string &varName) const;

  //---------------------------------------------------------------------------
  // binary arithmetic operations
  //---------------------------------------------------------------------------
   CG_outputRepr* CreatePlus(CG_outputRepr* lop, CG_outputRepr* rop) const;
   CG_outputRepr* CreateMinus(CG_outputRepr* lop, CG_outputRepr* rop) const;
   CG_outputRepr* CreateTimes(CG_outputRepr* lop, CG_outputRepr* rop) const;
   CG_outputRepr* CreateIntegerFloor(CG_outputRepr* lop, CG_outputRepr* rop) const;
   CG_outputRepr* CreateIntegerMod(CG_outputRepr* lop, CG_outputRepr* rop) const;

  //---------------------------------------------------------------------------
  // binary logical operations
  //---------------------------------------------------------------------------
   CG_outputRepr* CreateAnd(CG_outputRepr* lop, CG_outputRepr* rop) const;

  //---------------------------------------------------------------------------
  // binary relational operations
  //---------------------------------------------------------------------------
  // CG_outputRepr* CreateGE(CG_outputRepr*, CG_outputRepr*) const;
   CG_outputRepr* CreateLE(CG_outputRepr* lop, CG_outputRepr* rop) const;
   CG_outputRepr* CreateEQ(CG_outputRepr* lop, CG_outputRepr* rop) const;
     
  //---------------------------------------------------------------------------
  // stmt list gen operations
  //---------------------------------------------------------------------------
  // CG_outputRepr*
  //  CreateStmtList(CG_outputRepr *singleton = NULL) const;
  // CG_outputRepr*
  //  StmtListInsertLast(CG_outputRepr* list, CG_outputRepr* node) const;
   CG_outputRepr*
    StmtListAppend(CG_outputRepr* list1, CG_outputRepr* list2) const;

   //CG_outputRepr* CreateDim3(const char* varName, int  arg1, int  arg2) const;
   CG_outputRepr* CreateDim3(const char* varName, CG_outputRepr* arg1, CG_outputRepr*  arg2, CG_outputRepr* arg3 = NULL) const;

   // Manu:: added for fortran support
   bool isInputFortran() const;

  //---------------------------------------------------------------------------
  // kernel generation
  //---------------------------------------------------------------------------
  // CG_outputRepr* CreateKernel(immed_list* iml) const;

  //---------------------------------------------------------------------------
  // Add a modifier to a type (for things like __global__)
  //---------------------------------------------------------------------------
  //type_node* ModifyType(type_node* base, const char* modifier) const;


private:
  SgSymbolTable *symtab_;
  SgSymbolTable *symtab2_;
  SgNode* root_;
  SgGlobal* global_;
  SgGlobal* global_scope;
  int isFortran; // Manu:: added for fortran support
};

extern char *k_ocg_comment;
  //bool substitute(SgExpression *in, SgVariableSymbol *sym, SgExpression *expr, SgNode* root, SgExpression* parent);  
  //bool substitute(SgStatement *tn, SgVariableSymbol *sym, SgExpression* expr, SgNode* root, SgSymbolTable* symtab);  
std::vector<SgVarRefExp *>substitute(SgNode *tnl, const SgVariableSymbol *sym, SgExpression *expr,SgNode* root) ;




} // namespace

#endif
