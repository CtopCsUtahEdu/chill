#ifndef CG_chillBuilder_h
#define CG_chillBuilder_h

#include <basic/Tuple.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_chillRepr.h>
#include <string>
 



namespace omega {

class CG_chillBuilder : public CG_outputBuilder { 
private:

  chillAST_SourceFile   *toplevel; 
  chillAST_FunctionDecl *currentfunction;

  chillAST_SymbolTable  *symtab_;  // symbol table for FUNC (parameters?)
  chillAST_SymbolTable  *symtab2_; // symbol table for func BODY 

public:
  CG_chillBuilder() ; 
  CG_chillBuilder(chillAST_SourceFile *top, chillAST_FunctionDecl *func ) ; 
  ~CG_chillBuilder();

  const char *ClassName() { return "chillBuilder"; }; 

  //---------------------------------------------------------------------------
  // place holder generation  not in  CG_outputBuilder ?? or CG_roseBuilder
  //---------------------------------------------------------------------------
  CG_outputRepr* CreatePlaceHolder(int indent, 
                                   CG_outputRepr *stmt,
                                   Tuple<CG_outputRepr*> &funcList,
                                   Tuple<std::string> &loop_vars) const;

  //---------------------------------------------------------------------------
  // substitute variables in stmt
  //---------------------------------------------------------------------------


   CG_outputRepr *CreateSubstitutedStmt(int indent, 
                                        CG_outputRepr *stmt,
                                        const std::vector<std::string> &vars,
                                        std::vector<CG_outputRepr *> &subs,
                                        bool actuallyPrint =true) const;



  //---------------------------------------------------------------------------
  // assignment generation
  //---------------------------------------------------------------------------
  CG_outputRepr* CreateAssignment(int indent, CG_outputRepr* lhs, CG_outputRepr* rhs) const;

  CG_outputRepr *CreatePlusAssignment(int indent, 
                                      CG_outputRepr *lhs,
                                      CG_outputRepr *rhs) const;


  //---------------------------------------------------------------------------
  // function invocation generation
  //---------------------------------------------------------------------------
  CG_outputRepr* CreateInvoke(const std::string &funcName,
                              std::vector<CG_outputRepr*> &argList,
                              bool is_array = true) const;

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
  CG_outputRepr* CreateIf(int indent, 
                          CG_outputRepr* guardCondition,
                          CG_outputRepr* true_stmtList, 
                          CG_outputRepr* false_stmtList) const;

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
  CG_outputRepr* CreateLoop(int indent, 
                            CG_outputRepr* control,
                            CG_outputRepr* stmtList) const;

  //---------------------------------------------------------------------------
  // basic operations
  //---------------------------------------------------------------------------
  CG_outputRepr* CreateInt(int num ) const;  // create Integer constant?
  CG_outputRepr* CreateFloat(float num ) const;  // 
  CG_outputRepr* CreateDouble(double num ) const;  // should these all be chillRepr ???




  bool isInteger(CG_outputRepr *op) const;
  CG_outputRepr* CreateIdent(const std::string &idStr) const; // create a new INTEGER identifier


  CG_outputRepr* CreateDotExpression(CG_outputRepr *l, CG_outputRepr *r) const;
	CG_outputRepr* CreateArrayRefExpression(const std::string &_s,
			CG_outputRepr *rop) const;
	CG_outputRepr* CreateArrayRefExpression(CG_outputRepr *lop,
			CG_outputRepr *rop) const;
	CG_outputRepr* ObtainInspectorData(const std::string &_s, const std::string &member_name) const;
  bool QueryInspectorType(const std::string &varName) const; 
  CG_outputRepr* CreateNullStatement() const;

  //---------------------------------------------------------------------------
  // binary arithmetic operations
  //---------------------------------------------------------------------------
  CG_outputRepr* CreatePlus(         CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateMinus(        CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateTimes(        CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateIntegerDivide(CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateIntegerFloor( CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateIntegerMod(   CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateIntegerCeil(  CG_outputRepr* lop, CG_outputRepr* rop) const;

  //---------------------------------------------------------------------------
  // binary logical operations
  //---------------------------------------------------------------------------
  CG_outputRepr* CreateAnd(CG_outputRepr* lop, CG_outputRepr* rop) const;

  //---------------------------------------------------------------------------
  // binary relational operations
  //---------------------------------------------------------------------------
  //CG_outputRepr* CreateGE(CG_outputRepr* lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateLE(CG_outputRepr*  lop, CG_outputRepr* rop) const;
  CG_outputRepr* CreateEQ(CG_outputRepr*  lop, CG_outputRepr* rop) const;
	CG_outputRepr* CreateNEQ(CG_outputRepr* lop, CG_outputRepr* rop) const;
     
  //---------------------------------------------------------------------------
  // stmt list gen operations
  //---------------------------------------------------------------------------
  CG_outputRepr* 
    CreateStmtList(CG_outputRepr *singleton = NULL) const;
  CG_outputRepr* 
    StmtListInsertLast(CG_outputRepr* list, CG_outputRepr* node) const;
  CG_outputRepr* 
    StmtListAppend(CG_outputRepr* list1, CG_outputRepr* list2) const;

	CG_outputRepr *CreateAddressOf(CG_outputRepr *op) const ;
	CG_outputRepr *CreateBreakStatement(void) const;
	CG_outputRepr *CreateStatementFromExpression(CG_outputRepr *exp) const;

  //---------------------------------------------------------------------------
  // Utility Functions
  //---------------------------------------------------------------------------
  //CompoundStmt *StmtV2Compound(StmtList *slist) const // ?? 
    //{ return new (astContext_)CompoundStmt(*astContext_, &(*slist)[0], (*slist).size(), SourceLocation(), SourceLocation()); }

  //bool substitute(clang::Expr *in, std::string sym, clang::Expr* expr, clang::Expr *parent) const;

  CG_outputRepr *CreateStruct(const std::string struct_name,
                              std::vector<std::string> data_members,
                              std::vector<CG_outputRepr *> data_types);

  CG_outputRepr *CreateClassInstance(std::string name , CG_outputRepr *class_def);
	CG_outputRepr *lookup_member_data(CG_outputRepr* scope, std::string varName, CG_outputRepr *instance);
  CG_outputRepr* CreatePointer(std::string  &name) const;
	CG_outputRepr* ObtainInspectorRange(const std::string &_s, const std::string &_name) const;

  CG_outputRepr *CreateLinkedListStruct(const std::string class_name,
                                        std::vector<std::string> class_data_members,
                                        std::vector<CG_outputRepr *> class_data_types) {
		// TODO Implement
    throw std::runtime_error(__PRETTY_FUNCTION__);
  }

	CG_outputRepr *CreateClass(const std::string class_name,
			std::vector<std::string> class_data_members = std::vector<std::string>(), std::vector<CG_outputRepr *> class_data_types = std::vector<
					CG_outputRepr *>(), std::vector<std::string> methods =
					std::vector<std::string>(),
			std::vector<std::vector<std::string> > method_params = std::vector<
					std::vector<std::string> >(),
			std::vector<CG_outputRepr *> method_return_types = std::vector<
					CG_outputRepr *>(),
			std::vector<CG_outputRepr *> method_bodies = std::vector<
					CG_outputRepr *>()) {
		// TODO Implement
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}

	CG_outputRepr* CreateArrowRefExpression(const std::string &_s,
																					CG_outputRepr *rop) const {
		// TODO Implement
    throw std::runtime_error(__PRETTY_FUNCTION__);
	}
	CG_outputRepr* CreateArrowRefExpression(CG_outputRepr *lop,
																					CG_outputRepr *rop) const {
    // TODO Implement
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
};

} // namespace omega

#endif
