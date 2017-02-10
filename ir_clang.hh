#ifndef IR_CLANG_HH
#define IR_CLANG_HH

#include <omega.h>
#include "ir_chill.hh"
//#include <AstInterface_CLANG.h>
#include "chill_error.hh"
#include "chill_io.hh"

#define __STDC_CONSTANT_MACROS
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMap.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"

#include "chill_ast.hh"

// using namespace clang;          // NEVER EVER do this in a header file 
// using namespace clang::driver;  // NEVER EVER do this in a header file 

extern vector<chillAST_VarDecl *> VariableDeclarations;  // a global.   TODO 

typedef llvm::SmallVector<clang::Stmt *, 16> StmtList;  // TODO delete 

class aClangCompiler {
private:
  //Chill_ASTConsumer *astConsumer_;
  clang::ASTContext *astContext_;
  
  clang::DiagnosticOptions *diagnosticOptions;
  clang::TextDiagnosticPrinter *pTextDiagnosticPrinter;
  clang::DiagnosticIDs *diagID ;
  clang::DiagnosticsEngine *diagnosticsEngine;
  clang::CompilerInstance *Clang;
  clang::Preprocessor *preprocessor;
  //FileManager *FileMgr;
  //clang::CompilerInvocation *CI;

  clang::FileManager *fileManager;
  clang::SourceManager *sourceManager;

  // UNUSED? 
  clang::Diagnostic *diagnostic;
  clang::LangOptions *languageOptions;
  clang::HeaderSearchOptions *headerSearchOptions;
  //clang::HeaderSearch *headerSearch;
  std::shared_ptr<clang::TargetOptions> targetOptions;
  clang::TargetInfo *pTargetInfo;
  clang::PreprocessorOptions *preprocessorOptions;
  clang::FrontendOptions *frontendOptions;
  clang::IdentifierTable *idTable;
  clang::SelectorTable *selTable;
  clang::Builtin::Context *builtinContext;


public:
  char *SourceFileName; 
  chillAST_SourceFile * entire_file_AST; // TODO move out of public

  aClangCompiler( char *filename ); // constructor
  chillAST_FunctionDecl *findprocedurebyname( char *name );   // someday, return the chill AST
  clang::FunctionDecl *FD;  
  //Chill_ASTConsumer *getASTConsumer() { return astConsumer_; }
  clang::ASTContext    *getASTContext()       { return astContext_; }
  clang::SourceManager *getASTSourceManager() { return sourceManager; }; 
};


// singleton class for global clang initialization
// TODO: Add support for multiple files in same script
class IR_clangCode_Global_Init {
private:
  static IR_clangCode_Global_Init *pinstance;   // the one and only
  // protecting the constructor is the SINGLETON PATTERN.   a global by any other name
  // IR_clangCode_Global_Init(); 
  ~IR_clangCode_Global_Init(); // is this hidden, too?
  chillAST_FunctionDecl * chillFD; // the original C code

  clang::ASTContext    *astContext_;  
  clang::SourceManager *sourceManager;
public:
  clang::ASTContext *getASTContext() { return astContext_; }
  clang::SourceManager *getSourceManager() { return sourceManager; }; 
  static IR_clangCode_Global_Init *Instance(char **argv);
  static IR_clangCode_Global_Init *Instance() { return pinstance; } ;
  aClangCompiler *ClangCompiler; // this is the thing we really just want one of


  void setCurrentFunction( chillAST_node *F ) { chillFD = (chillAST_FunctionDecl *)F; } ;
  chillAST_FunctionDecl *getCurrentFunction( ) { return chillFD;  } ;


  void setCurrentASTContext( clang::ASTContext *A ) { astContext_ = A;};
  clang::ASTContext    *getCurrentASTContext() { return astContext_; } ; 

  void setCurrentASTSourceManager( clang::SourceManager *S ) { sourceManager = S; } ;
  clang::SourceManager *getCurrentASTSourceManager() { return sourceManager; } ; 
};



class IR_clangCode: public IR_chillCode{   // for an entire file?  A single function?
protected:

  clang::FunctionDecl *func_;   // a clang construct   the function we're currenly modifying
  clang::ASTContext *astContext_;
  clang::SourceManager *sourceManager;

  // firstScope;
  //   symboltable1,2,3 ??
  // topleveldecls
  // 

public:
  clang::ASTContext    *getASTContext() { return astContext_; } ;
  clang::SourceManager *getASTSourceManager() { return sourceManager; } ; 

  IR_clangCode(const char *filename, const char *proc_name, const char *dest_name = NULL);
  ~IR_clangCode();

  int ArrayIndexStartAt() { return 0;} // TODO FORTRAN

  friend class IR_chillArraySymbol;
  friend class IR_chillArrayRef;
};


#endif
