#ifndef IR_ROSE_HH
#define IR_ROSE_HH

#include <omega.h>
#include <code_gen/CG_chillRepr.h>  // just for CreateArrayRefRepr.  probably a bad idea 

#include "chill_ast.hh"  // needed now that we're going immediately from rose ast to chill ast 
#include "chill_io.hh"

#include "ir_code.hh"
#include "ir_rose_utils.hh"
#include <AstInterface_ROSE.h>
#include "chill_error.hh"
#include "staticSingleAssignment.h"
#include "VariableRenaming.h"
#include "ssaUnfilteredCfg.h"
#include "virtualCFG.h"


// IR_roseCode is built on IR_chillCode
#include "ir_chill.hh" 

// forward declarations 
chillAST_node * ConvertRoseFile(  SgGlobal *sg, const char *filename ); // the entire file 
chillAST_node * ConvertRoseFunctionDecl( SgFunctionDeclaration *D );
chillAST_node * ConvertRoseParamVarDecl( SgInitializedName *vardecl );
chillAST_node * ConvertRoseInitName( SgInitializedName *vardecl );
chillAST_node * ConvertRoseVarDecl( SgVariableDeclaration *vardecl ); // stupid name TODO
chillAST_node * ConvertRoseForStatement( SgForStatement *forstatement );
chillAST_node * ConvertRoseWhileStatement( SgWhileStmt *whilestmt );
chillAST_node * ConvertRoseExprStatement( SgExprStatement *exprstatement );
chillAST_node * ConvertRoseBinaryOp( SgBinaryOp *binaryop );
chillAST_node * ConvertRoseMemberExpr( SgBinaryOp *binaryop); // binop! a.b
chillAST_node * ConvertRoseArrowExp  ( SgBinaryOp *binaryop); // binop! a->b
char *          ConvertRoseMember( SgVarRefExp* memb ); // TODO
chillAST_node * ConvertRoseUnaryOp( SgUnaryOp *unaryop );
chillAST_node * ConvertRoseVarRefExp( SgVarRefExp *varrefexp );
chillAST_node * ConvertRoseIntVal( SgIntVal *riseintval );
chillAST_node * ConvertRoseFloatVal( SgFloatVal *rosefloatval );
chillAST_node * ConvertRoseDoubleVal( SgDoubleVal *rosecdoubleval );
chillAST_node * ConvertRoseBasicBlock( SgBasicBlock *bb );
chillAST_node * ConvertRoseFunctionCallExp( SgFunctionCallExp* );
chillAST_node * ConvertRoseReturnStmt( SgReturnStmt *rs );
chillAST_node * ConvertRoseArrayRefExp( SgPntrArrRefExp *roseARE );
chillAST_node * ConvertRoseCastExp( SgCastExp *roseCE );
chillAST_node * ConvertRoseAssignInitializer( SgAssignInitializer *roseAI );
// TODO 
chillAST_node * ConvertRoseStructDefinition( SgClassDefinition *def );
chillAST_node * ConvertRoseStructDeclaration( SgClassDeclaration *dec );


chillAST_node * ConvertRoseIfStmt( SgIfStmt *ifstatement );

chillAST_node * ConvertRoseTypeDefDecl( SgTypedefDeclaration *TDD );

//chillAST_node * ConvertRoseRecordDecl( clang::RecordDecl *D, chillAST_node * );
//chillAST_node * ConvertRoseDeclStmt( clang::DeclStmt *clangDS, chillAST_node * );
//chillAST_node * ConvertRoseCompoundStmt( clang::CompoundStmt *clangCS, chillAST_node * );

//chillAST_node * ConvertRoseDeclRefExpr( clang::DeclRefExpr * clangDRE, chillAST_node * );
//chillAST_node * ConvertRoseCStyleCastExpr( clang::CStyleCastExpr *clangICE, chillAST_node * );
//chillAST_node * ConvertRoseIfStmt( clang::IfStmt *clangIS , chillAST_node *);
chillAST_node * ConvertRoseGenericAST( SgNode *n );

extern SgProject *OneAndOnlySageProject;  // a global

class IR_roseCode: public IR_chillCode {
protected:
  // things that start with Sg are rose-specific (Sage, actually)
  SgProject* project;
  SgSourceFile* file;
  SgGlobal *root;
  SgGlobal *firstScope;
  SgSymbolTable* symtab_;
  SgSymbolTable* symtab2_;
  SgSymbolTable* symtab3_;
  SgDeclarationStatementPtrList::iterator p;
  SgFunctionDeclaration *func;
  
  
  bool is_fortran_;
  int i_;
  StaticSingleAssignment *ssa_for_scalar;
  ssa_unfiltered_cfg::SSA_UnfilteredCfg *main_ssa;
  VariableRenaming *varRenaming_for_scalar;
  
  
  std::vector<chillAST_VarDecl> entire_file_symbol_table;
  
public:

  void print() { chillfunc->print(); printf("\n"); fflush(stdout); }; 
  
  IR_roseCode(const char *filename, const char* proc_name, const char* dest_name = NULL );
  ~IR_roseCode();

  int ArrayIndexStartAt() {
    if (is_fortran_)
      return 1;
    else
      return 0;
  }
  
  void populateLists(SgNode* tnl_1, SgStatementPtrList* list_1,
                     SgStatementPtrList& output_list_1);
  void populateScalars(const omega::CG_outputRepr *repr1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
                       std::set<std::string> &indices, std::vector<std::string> &index);
  //        std::set<std::string> &def_vars);
  /*void findDefinitions(SgStatementPtrList &list_1,
    std::set<VirtualCFG::CFGNode> &reaching_defs_1,
    std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
    std::set<std::string> &def_vars);
  */
  /*    void checkDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkSelfDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkWriteDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
  */

  /*    std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> >
        FindScalarDeps(const omega::CG_outputRepr *repr1,
        const omega::CG_outputRepr *repr2, std::vector<std::string> index,
        int i, int j);
  */
  void finalizeRose();
  
  // Manu:: Added functions required for reduction operation
  // virtual omega::CG_outputRepr * FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B) = 0;

  //std::vector<IR_PointerArrayRef *> FindPointerArrayRef(const omega::CG_outputRepr *repr) const; // inherit from chillcode ??

  bool ReplaceLHSExpression(omega::CG_outputRepr *code, IR_ArrayRef *ref);

  //static int rose_pointer_counter ; // for manufactured arrays
  
  friend class IR_roseArraySymbol;
  friend class IR_roseArrayRef;
};


#endif
