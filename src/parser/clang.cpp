

/*****************************************************************************
  Copyright (C) 2009-2010 University of Utah
  All Rights Reserved.

Purpose:
CHiLL's CLANG interface.
convert from CLANG AST to chill AST

Notes:
Array supports mixed pointer and array type in a single declaration.

History:
12/10/2010 LLVM/CLANG Interface created by Saurav Muralidharan.
 *****************************************************************************/

#include <typeinfo>
#include <sstream>
#include "parser/clang.h"
#include "chill_error.hh"
#include "scanner/sanityCheck.h"
#include "scanner/definitionLinker.h"

#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/ASTContext.h>
#include <clang/Lex/Lexer.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>

#define UNWRAP(x) ((x)[0])
#define WRAP(x) (chillAST_NodeList(1,x))

// fwd declarations
chillAST_NodeList ConvertVarDecl( clang::VarDecl *D );
chillAST_NodeList ConvertTypeDefDecl( clang::TypedefDecl *TDD );
chillAST_NodeList ConvertRecordDecl( clang::RecordDecl *D );
chillAST_NodeList ConvertDeclStmt( clang::DeclStmt *clangDS );
chillAST_NodeList ConvertCompoundStmt( clang::CompoundStmt *clangCS );
chillAST_NodeList ConvertFunctionDecl( clang::FunctionDecl *D );
chillAST_NodeList ConvertForStmt( clang::ForStmt *clangFS );
chillAST_NodeList ConvertWhileStmt( clang::WhileStmt *clangWS );
chillAST_NodeList ConvertUnaryOperator( clang::UnaryOperator * clangU );
chillAST_NodeList ConvertBinaryOperator( clang::BinaryOperator * clangBO );
chillAST_NodeList ConvertArraySubscriptExpr( clang::ArraySubscriptExpr *clangASE );
chillAST_NodeList ConvertDeclRefExpr( clang::DeclRefExpr * clangDRE );
chillAST_NodeList ConvertIntegerLiteral( clang::IntegerLiteral *clangIL );
chillAST_NodeList ConvertFloatingLiteral( clang::FloatingLiteral *clangFL );
chillAST_NodeList ConvertImplicitCastExpr( clang::ImplicitCastExpr *clangICE );
chillAST_NodeList ConvertCStyleCastExpr( clang::CStyleCastExpr *clangICE );
chillAST_NodeList ConvertReturnStmt( clang::ReturnStmt *clangRS );
chillAST_NodeList ConvertCallExpr( clang::CallExpr *clangCE );
chillAST_NodeList ConvertIfStmt( clang::IfStmt *clangIS );
chillAST_NodeList ConvertMemberExpr( clang::MemberExpr *clangME );
chillAST_NodeList ConvertConditionalOperator( clang::ConditionalOperator * clangCO );


chillAST_node * ConvertTranslationUnit(  clang::TranslationUnitDecl *TUD, char *filename );
chillAST_NodeList ConvertGenericClangAST( clang::Stmt *s );


using namespace clang;
using namespace std;

extern vector<chillAST_VarDecl *> VariableDeclarations;  // a global.   TODO

class aClangCompiler {
private:
  clang::ASTContext *astContext_;
  clang::DiagnosticOptions *diagnosticOptions;
  clang::TextDiagnosticPrinter *pTextDiagnosticPrinter;
  clang::DiagnosticIDs *diagID ;
  clang::DiagnosticsEngine *diagnosticsEngine;
  clang::CompilerInstance *Clang;
  clang::SourceManager *sourceManager;
  std::shared_ptr<clang::TargetOptions> targetOptions;

public:
  char *SourceFileName;
  chillAST_SourceFile * entire_file_AST; // TODO move out of public

  aClangCompiler(const char *filename ); // constructor
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
  static IR_clangCode_Global_Init *Instance(const char **argv);
  static IR_clangCode_Global_Init *Instance() { return pinstance; } ;
  aClangCompiler *ClangCompiler; // this is the thing we really just want one of


  void setCurrentFunction( chillAST_node *F ) { chillFD = (chillAST_FunctionDecl *)F; } ;
  chillAST_FunctionDecl *getCurrentFunction( ) { return chillFD;  } ;


  void setCurrentASTContext( clang::ASTContext *A ) { astContext_ = A;};
  clang::ASTContext    *getCurrentASTContext() { return astContext_; } ;

  void setCurrentASTSourceManager( clang::SourceManager *S ) { sourceManager = S; } ;
  clang::SourceManager *getCurrentASTSourceManager() { return sourceManager; } ;
};

// forward defs
SourceManager * globalSRCMAN;

chillAST_NodeList ConvertVarDecl( VarDecl *D ) {
   bool isParm = false;

   QualType T0 = D->getType();
   QualType T  = T0;
   if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
     T = Parm->getOriginalType();
     isParm = true;
   }
  ASTContext *ctx = IR_clangCode_Global_Init::Instance()->getASTContext();
  chillAST_NodeList arr;
  bool restrict = T.isRestrictQualified();
  while (isa<PointerType>(T) || ctx->getAsVariableArrayType(T) || ctx->getAsArrayType(T)) {
    while (isa<PointerType>(T)) {
      const PointerType *PTR = cast<PointerType>(T);
      arr.push_back(new chillAST_NULL());
      T = PTR->getPointeeType();
    }
    const VariableArrayType *VLA = ctx->getAsVariableArrayType(T);
    while (VLA) {
      Expr *SE = VLA->getSizeExpr();
      arr.push_back(UNWRAP(ConvertGenericClangAST(SE)));
      T = VLA->getElementType();
      VLA = ctx->getAsVariableArrayType(T);
    }
    const ArrayType *AT = ctx->getAsArrayType(T);
    while (AT) {
      int size = (int) cast<ConstantArrayType>(AT)->getSize().getZExtValue();
      arr.push_back(new chillAST_IntegerLiteral(size));
      T = AT->getElementType();
      AT = ctx->getAsArrayType(T);
    }
  }
  string TypeStr = T.getAsString();

  char *otype =  strdup( TypeStr.c_str());
  char *arraypart = parseArrayParts( otype );
  char *varname = strdup(D->getNameAsString().c_str());
  char *vartype = parseUnderlyingType(restricthack(otype));

  chillAST_VarDecl * chillVD = new chillAST_VarDecl( vartype, arraypart,  varname, arr, (void *)D );

  chillVD->isAParameter = isParm;
  chillVD->isRestrict = restrict;

  Expr *Init = D->getInit();
  if (Init)
    chillVD->setInit(UNWRAP(ConvertGenericClangAST(Init)));

  free (vartype);
  free (varname);

  // store this away for declrefexpr that references it!
  VariableDeclarations.push_back(chillVD);
  return WRAP(chillVD);
}



chillAST_NodeList ConvertRecordDecl( clang::RecordDecl *RD ) { // for structs and unions

  int count = 0;
  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) count++;

  char blurb[128];
  sprintf(blurb, "struct %s", RD->getNameAsString().c_str());
  debug_fprintf(stderr, "blurb is '%s'\n", blurb);

  chillAST_TypedefDecl *astruct = new chillAST_TypedefDecl( blurb, "");
  astruct->setStruct( true );
  astruct->setStructName( RD->getNameAsString().c_str() );

  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) {
    clang::FieldDecl *FD = (*fi);
    string TypeStr = FD->getType().getAsString();

    const char *typ  = TypeStr.c_str();
    const char *name = FD->getNameAsString().c_str();
    debug_fprintf(stderr, "(typ) %s (name) %s\n", typ, name);

    chillAST_VarDecl *VD = NULL;
    // very clunky and incomplete
    VD = new chillAST_VarDecl( astruct, "", name ); // can't handle arrays yet

    astruct->subparts.push_back(VD);
  }

  return WRAP(astruct);
}


chillAST_NodeList ConvertTypeDefDecl( TypedefDecl *TDD ) {
  char *under =  strdup( TDD->getUnderlyingType().getAsString().c_str());
  char *arraypart = splitTypeInfo(under);
  char *alias = strdup(TDD->getName().str().c_str());

  chillAST_TypedefDecl *CTDD = new chillAST_TypedefDecl( under, alias, arraypart );

  free(under);
  free(arraypart);

  return WRAP(CTDD);
}



chillAST_NodeList ConvertDeclStmt( DeclStmt *clangDS ) {
  chillAST_VarDecl *chillvardecl; // the thing we'll return if this is a single declaration
  chillAST_NodeList nl;

  DeclGroupRef dgr = clangDS->getDeclGroup();
  clang::DeclGroupRef::iterator DI = dgr.begin();
  clang::DeclGroupRef::iterator DE = dgr.end();

  for ( ; DI != DE; ++DI) {
    Decl *D = *DI;
    const char *declT =  D->getDeclKindName();
    //debug_fprintf(stderr, "a decl of type %s\n", D->getDeclKindName());

    if (!strcmp("Var", declT)) {
      VarDecl *V = dyn_cast<VarDecl>(D);
      nl.push_back(UNWRAP(ConvertVarDecl(V)));
    }
  }  // for each of possibly multiple decls

  return nl;  // OR a single decl
}



chillAST_NodeList ConvertCompoundStmt( CompoundStmt *clangCS ) {
  chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt;
  // for each clang child
  for (auto I = clangCS->child_begin(); I != clangCS->child_end(); ++I) { // ?? loop looks WRONG
    // create the chill ast for each child
    Stmt *child = *I;
    chillAST_NodeList nl =  ConvertGenericClangAST( child );
    chillCS->addChildren( nl );
  }
  return WRAP(chillCS);
}

chillAST_NodeList ConvertFunctionDecl( FunctionDecl *D ) {
  QualType QT = D->getReturnType();
  string ReturnTypeStr = QT.getAsString();

  // Function name
  DeclarationName DeclName = D->getNameInfo().getName();
  string FuncName = DeclName.getAsString();

  chillAST_FunctionDecl *chillFD = new chillAST_FunctionDecl( ReturnTypeStr.c_str(),  FuncName.c_str(), D);


  int numparams = D->getNumParams();

  for (int i=0; i<numparams; i++) {
    if (i) debug_fprintf(stderr, ", ");
    VarDecl *clangvardecl = D->getParamDecl(i);  // the ith parameter  (CLANG)
    ParmVarDecl *pvd = D->getParamDecl(i);
    QualType T = pvd->getOriginalType();
    debug_fprintf(stderr, "OTYPE %s\n", T.getAsString().c_str());

    chillAST_VarDecl *chillPVD = (chillAST_VarDecl*)UNWRAP(ConvertVarDecl( clangvardecl ));
    //chillPVD->print();  fflush(stdout);

    //chillPVD->isAParameter = 1;
    VariableDeclarations.push_back(chillPVD);

    chillFD->addParameter(chillPVD);
    debug_fprintf(stderr, "chillAST ParmVarDecl for %s from chill location 0x%x\n",chillPVD->varname, clangvardecl);
  } // for each parameter



  //debug_fprintf(stderr, ")\n{\n"); // beginning of function body
  //if (D->isExternC())    { chillFD->setExtern();  debug_fprintf(stderr, "%s is extern\n", FuncName.c_str()); };
  if (D->getBuiltinID()) { chillFD->setExtern();  debug_fprintf(stderr, "%s is builtin (extern)\n", FuncName.c_str()); };

  Stmt *clangbody = D->getBody();
  if (clangbody) { // may just be fwd decl or external, without an actual body
    chillAST_node *CB = UNWRAP(ConvertGenericClangAST( clangbody ));
    chillFD->setBody ( CB );
  }

  //debug_fprintf(stderr, "adding function %s  0x%x to FunctionDeclarations\n", chillFD->functionName, chillFD);
  FunctionDeclarations.push_back(chillFD);
  return WRAP(chillFD);
}


chillAST_NodeList ConvertForStmt( ForStmt *clangFS ) {

  Stmt *init = clangFS->getInit();
  Expr *cond = clangFS->getCond();
  Expr *incr = clangFS->getInc();
  Stmt *body = clangFS->getBody();

  chillAST_node *ini = UNWRAP(ConvertGenericClangAST( init ));
  chillAST_node *con = UNWRAP(ConvertGenericClangAST( cond ));
  chillAST_node *inc = UNWRAP(ConvertGenericClangAST( incr ));
  chillAST_node *bod = UNWRAP(ConvertGenericClangAST( body ));
  if (bod->getType() != CHILLAST_NODETYPE_COMPOUNDSTMT) {
    // make single statement loop bodies loop like other loops
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt( );
    cs->addChild( bod );
    bod = cs;
  }
  chillAST_ForStmt *chill_loop = new  chillAST_ForStmt( ini, con, inc, bod );
  return WRAP(chill_loop);
}

chillAST_NodeList ConvertWhileStmt( WhileStmt *clangWS ) {
  Expr *cond = clangWS->getCond();
  Stmt *body = clangWS->getBody();

  chillAST_node *cnd = UNWRAP(ConvertGenericClangAST( cond ));
  chillAST_node *bod = UNWRAP(ConvertGenericClangAST( body ));
  chillAST_WhileStmt *chill_loop = new  chillAST_WhileStmt( cnd, bod );
  return WRAP(chill_loop);
}


chillAST_NodeList ConvertIfStmt( IfStmt *clangIS ) {
  Expr *cond = clangIS->getCond();
  Stmt *thenpart = clangIS->getThen();
  Stmt *elsepart = clangIS->getElse();

  chillAST_node *con = UNWRAP(ConvertGenericClangAST( cond ));
  chillAST_node *thn = NULL;
  if (thenpart) thn = UNWRAP(ConvertGenericClangAST( thenpart ));
  chillAST_node *els = NULL;
  if (elsepart) els = UNWRAP(ConvertGenericClangAST( elsepart ));

  chillAST_IfStmt *ifstmt = new chillAST_IfStmt( con, thn, els );
  return WRAP(ifstmt);
}



chillAST_NodeList ConvertUnaryOperator( UnaryOperator * clangUO ) {
  const char *op = clangUO->getOpcodeStr(clangUO->getOpcode()).str().c_str();
  bool pre = !(clangUO->isPostfix());
  chillAST_node *sub = UNWRAP(ConvertGenericClangAST( clangUO->getSubExpr()));

  chillAST_UnaryOperator *chillUO = new chillAST_UnaryOperator( op, pre, sub );
  sub->setParent( chillUO );
  return WRAP(chillUO);
}


chillAST_NodeList ConvertBinaryOperator( BinaryOperator * clangBO ) {

  // get the clang parts
  Expr *lhs = clangBO->getLHS();
  Expr *rhs = clangBO->getRHS();

  // convert to chill equivalents
  chillAST_node *l = UNWRAP(ConvertGenericClangAST( lhs ));
  const char *opstring = clangBO->getOpcodeStr().str().c_str();
  chillAST_node *r = UNWRAP(ConvertGenericClangAST( rhs ));
  // TODO chill equivalent for numeric op.

  // build up the chill Binary Op AST node
  chillAST_BinaryOperator * binop = new chillAST_BinaryOperator( l, opstring, r );

  return WRAP(binop);
}




chillAST_NodeList ConvertArraySubscriptExpr( ArraySubscriptExpr *clangASE ) {

  Expr *clangbase  = clangASE->getBase();
  Expr *clangindex = clangASE->getIdx();
  //debug_fprintf(stderr, "clang base: "); clangbase->dump(); debug_fprintf(stderr, "\n");

  chillAST_node *bas  = UNWRAP(ConvertGenericClangAST( clangbase));
  chillAST_node *indx = UNWRAP(ConvertGenericClangAST( clangindex));

  chillAST_ArraySubscriptExpr * chillASE = new chillAST_ArraySubscriptExpr( bas, indx, clangASE);
  return WRAP(chillASE);
}



chillAST_NodeList ConvertDeclRefExpr( DeclRefExpr * clangDRE ) {
  DeclarationNameInfo DNI = clangDRE->getNameInfo();

  ValueDecl *vd = static_cast<ValueDecl *>(clangDRE->getDecl());

  QualType QT = vd->getType();
  string TypeStr = QT.getAsString();

  DeclarationName DN = DNI.getName();
  const char *varname = DN.getAsString().c_str() ;
  chillAST_DeclRefExpr * chillDRE = new chillAST_DeclRefExpr(TypeStr.c_str(),  varname);

  //debug_fprintf(stderr, "%s\n", DN.getAsString().c_str());
  return WRAP(chillDRE);
}



chillAST_NodeList ConvertIntegerLiteral( IntegerLiteral *clangIL ) {
  bool isSigned = clangIL->getType()->isSignedIntegerType();
  const char *printable = clangIL->getValue().toString(10, isSigned).c_str();
  int val = atoi( printable );
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val );
  return WRAP(chillIL);
}


chillAST_NodeList ConvertFloatingLiteral( FloatingLiteral *clangFL ) {
  double val = clangFL->getValueAsApproximateDouble();

  auto sr = clangFL->getSourceRange();
  auto pr = llvm::APFloat::getSizeInBits(clangFL->getValue().getSemantics());
  string lit = Lexer::getSourceText(CharSourceRange::getTokenRange(sr), *globalSRCMAN, LangOptions());

  return WRAP(new chillAST_FloatingLiteral(val, pr/32, lit.empty()? NULL:lit.c_str()));
}


chillAST_NodeList ConvertImplicitCastExpr( ImplicitCastExpr *clangICE ) {
  CastExpr *CE = dyn_cast<ImplicitCastExpr>(clangICE);
  chillAST_node * sub = UNWRAP(ConvertGenericClangAST( clangICE->getSubExpr() ));
  chillAST_ImplicitCastExpr *chillICE = new chillAST_ImplicitCastExpr( sub );
  // ignore the ImplicitCastExpr !!  TODO (probably a bad idea)
  return WRAP(sub);
}




chillAST_NodeList ConvertCStyleCastExpr( CStyleCastExpr *clangCSCE ) {
  CastExpr *CE = dyn_cast<CastExpr>(clangCSCE);

  const char * towhat = strdup( clangCSCE->getTypeAsWritten().getAsString().c_str() );

  chillAST_node * sub = UNWRAP(ConvertGenericClangAST( clangCSCE->getSubExprAsWritten()));
  chillAST_CStyleCastExpr *chillCSCE = new chillAST_CStyleCastExpr( towhat, sub );
  sub->setParent( chillCSCE );
  return WRAP(chillCSCE);
}




chillAST_NodeList ConvertReturnStmt( ReturnStmt *clangRS ) {
  chillAST_node * retval = UNWRAP(ConvertGenericClangAST( clangRS->getRetValue())); // NULL is handled

  chillAST_ReturnStmt * chillRS = new chillAST_ReturnStmt( retval );
  if (retval) retval->setParent( chillRS );
  return WRAP(chillRS);
}


chillAST_NodeList ConvertCallExpr( CallExpr *clangCE ) {
  chillAST_node *callee = UNWRAP(ConvertGenericClangAST( clangCE->getCallee() ));

  chillAST_CallExpr *chillCE = new chillAST_CallExpr( callee );
  callee->setParent( chillCE );

  int numargs = clangCE->getNumArgs();
  Expr **clangargs =  clangCE->getArgs();
  for (int i=0; i<numargs; i++) {
    chillCE->addArg( UNWRAP(ConvertGenericClangAST( clangargs[i] )) );
  }

  return WRAP(chillCE);
}


chillAST_NodeList ConvertParenExpr( ParenExpr *clangPE ) {
  chillAST_node *sub = UNWRAP(ConvertGenericClangAST( clangPE->getSubExpr()));
  chillAST_ParenExpr *chillPE = new chillAST_ParenExpr( sub );

  return WRAP(chillPE);
}

chillAST_NodeList ConvertConditionalOperator( clang::ConditionalOperator * clangCO ) {
  chillAST_node *cond = UNWRAP(ConvertGenericClangAST(clangCO->getCond()));
  chillAST_node *trueExpr = UNWRAP(ConvertGenericClangAST(clangCO->getTrueExpr()));
  chillAST_node *falseExpr = UNWRAP(ConvertGenericClangAST(clangCO->getFalseExpr()));
  chillAST_TernaryOperator *chillTO = new chillAST_TernaryOperator("?", cond, trueExpr, falseExpr);

  return WRAP(chillTO);
}


chillAST_node * ConvertTranslationUnit(  TranslationUnitDecl *TUD, char *filename ) {
  // TUD derived from Decl and DeclContext
  static DeclContext *DC = TUD->castToDeclContext( TUD );

  chillAST_SourceFile * topnode = new chillAST_SourceFile( filename  );
  topnode->setFrontend("clang");
  topnode->chill_array_counter  = 1;
  topnode->chill_scalar_counter = 0;

  // now recursively build clang AST from the children of TUD
  DeclContext::decl_iterator start = DC->decls_begin();
  DeclContext::decl_iterator end   = DC->decls_end();
  for (DeclContext::decl_iterator DI=start; DI != end; ++DI) {
    Decl *D = *DI;
    chillAST_NodeList child;

    if (isa<FunctionDecl>(D))
      child = ConvertFunctionDecl( dyn_cast<FunctionDecl>(D) );
    else if (isa<VarDecl>(D))
      child = ConvertVarDecl( dyn_cast<VarDecl>(D) );
    else if (isa<TypedefDecl>(D))
      child = ConvertTypeDefDecl( dyn_cast<TypedefDecl>(D) );
    else if (isa<RecordDecl>(D))
      child = ConvertRecordDecl( dyn_cast<RecordDecl>(D) );
    else if (isa<TypeAliasDecl>(D))
      throw std::runtime_error("TUD TypeAliasDecl  TODO \n");
    else
      throw std::runtime_error(std::string("TUD a declaration of type which I can't handle: ") + D->getDeclKindName());

    topnode->addChildren(child);
    if (D->isImplicit() || !globalSRCMAN->getFilename(D->getLocation()).equals(filename))
      for (auto i = child.begin(); i != child.end(); ++i)
        (*i)->isFromSourceFile = false;
  }
  return ( chillAST_node *)  topnode;
}



 chillAST_NodeList ConvertGenericClangAST( Stmt *s ) {

   chillAST_NodeList ret;
   if (s == NULL) return WRAP(NULL);
   if (isa<CompoundStmt>(s))              {ret = ConvertCompoundStmt( dyn_cast<CompoundStmt>(s));
   } else if (isa<DeclStmt>(s))           {ret = ConvertDeclStmt(dyn_cast<DeclStmt>(s));
   } else if (isa<ForStmt>(s))            {ret = ConvertForStmt(dyn_cast<ForStmt>(s));
   } else if (isa<WhileStmt>(s))          {ret = ConvertWhileStmt(dyn_cast<WhileStmt>(s));
   } else if (isa<BinaryOperator>(s))     {ret = ConvertBinaryOperator(dyn_cast<BinaryOperator>(s));
   } else if (isa<ArraySubscriptExpr>(s)) {ret = ConvertArraySubscriptExpr(dyn_cast<ArraySubscriptExpr>(s));
   } else if (isa<DeclRefExpr>(s))        {ret = ConvertDeclRefExpr(dyn_cast<DeclRefExpr>(s));
   } else if (isa<FloatingLiteral>(s))    {ret = ConvertFloatingLiteral(dyn_cast<FloatingLiteral>(s));
   } else if (isa<IntegerLiteral>(s))     {ret = ConvertIntegerLiteral(dyn_cast<IntegerLiteral>(s));
   } else if (isa<UnaryOperator>(s))      {ret = ConvertUnaryOperator(dyn_cast<UnaryOperator>(s));
   } else if (isa<ImplicitCastExpr>(s))   {ret = ConvertImplicitCastExpr(dyn_cast<ImplicitCastExpr>(s));
   } else if (isa<CStyleCastExpr>(s))     {ret = ConvertCStyleCastExpr(dyn_cast<CStyleCastExpr>(s));
   } else if (isa<ReturnStmt>(s))         {ret = ConvertReturnStmt(dyn_cast<ReturnStmt>(s));
   } else if (isa<CallExpr>(s))           {ret = ConvertCallExpr(dyn_cast<CallExpr>(s));
   } else if (isa<ParenExpr>(s))          {ret = ConvertParenExpr(dyn_cast<ParenExpr>(s));
   } else if (isa<IfStmt>(s))             {ret = ConvertIfStmt(dyn_cast<IfStmt>(s));
   } else if (isa<MemberExpr>(s))         {ret = ConvertMemberExpr(dyn_cast<MemberExpr>(s));
   } else if (isa<ConditionalOperator>(s)){ret = ConvertConditionalOperator(dyn_cast<ConditionalOperator>(s));


     // these can only happen at the top level?
     //   } else if (isa<FunctionDecl>(D))       { ret = ConvertFunctionDecl( dyn_cast<FunctionDecl>(D));
     //} else if (isa<VarDecl>(D))            { ret =      ConvertVarDecl( dyn_cast<VarDecl>(D) );
     //} else if (isa<TypedefDecl>(D))        { ret =  ConvertTypeDefDecl( dyn_cast<TypedefDecl>(D));
     //  else if (isa<TranslationUnitDecl>(s))  // need filename




     //   } else if (isa<>(s))                  {         Convert ( dyn_cast<>(s));

     /*
     */

   } else {
     std::string err = "ConvertGenericClangAST() UNHANDLED";
     if (isa<Stmt>(s)) err = err + "Stmt of type " + s->getStmtClassName();
     throw std::runtime_error(err.c_str());
   }

   return ret;
 }

class NULLASTConsumer : public ASTConsumer {
};


// ----------------------------------------------------------------------------
// Class: IR_clangCode_Global_Init
// ----------------------------------------------------------------------------

IR_clangCode_Global_Init *IR_clangCode_Global_Init::pinstance = 0;


IR_clangCode_Global_Init *IR_clangCode_Global_Init::Instance(const char **argv) {
  debug_fprintf(stderr, "in IR_clangCode_Global_Init::Instance(), ");
  if (pinstance == 0) {
    // this is the only way to create an IR_clangCode_Global_Init
    pinstance = new IR_clangCode_Global_Init;
    pinstance->ClangCompiler = new aClangCompiler( argv[1] );

  }
  return pinstance;
}


aClangCompiler::aClangCompiler(const char *filename ) {
  SourceFileName = strdup( filename );

  // Arguments to pass to the clang frontend
  std::vector<const char *> args;
  args.push_back(strdup(filename));

  // The compiler invocation needs a DiagnosticsEngine so it can report problems
  diagnosticOptions =  new DiagnosticOptions(); // private member of aClangCompiler

  pTextDiagnosticPrinter = new clang::TextDiagnosticPrinter(llvm::errs(), diagnosticOptions); // private member of aClangCompiler

  diagID =  new clang::DiagnosticIDs(); // private member of IR_clangCode_Global_Init

  diagnosticsEngine = new clang::DiagnosticsEngine(diagID, diagnosticOptions, pTextDiagnosticPrinter);

  // Create the compiler invocation
  // This class is designed to represent an abstract "invocation" of the compiler,
  // including data such as the include paths, the code generation options,
  // the warning flags, and so on.
  std::shared_ptr<clang::CompilerInvocation> CI(new clang::CompilerInvocation);
  clang::CompilerInvocation::CreateFromArgs(*CI, &args[0], &args[0] + args.size(), *diagnosticsEngine);

  // Create the compiler instance
  Clang = new clang::CompilerInstance();  // TODO should have a better name ClangCompilerInstance

  // Get ready to report problems
  Clang->createDiagnostics(nullptr, true);
  targetOptions = std::make_shared<clang::TargetOptions>();
  targetOptions->Triple = llvm::sys::getDefaultTargetTriple();

  TargetInfo *pti = TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), targetOptions);

  Clang->setTarget(pti);
  Clang->createFileManager();
  FileManager &FileMgr = Clang->getFileManager();
  clang::FileManager *fileManager = &FileMgr;
  Clang->createSourceManager(FileMgr);
  SourceManager &SourceMgr = Clang->getSourceManager();
  sourceManager = &SourceMgr; // ?? aclangcompiler copy
  globalSRCMAN = &SourceMgr; //  TODO   global bad

  // Replace the current invocation
#if CLANG_VERSION_MAJOR > 3
  Clang->setInvocation(CI);
#else
  Clang->setInvocation(CI.get());
#endif

  Clang->createPreprocessor(TU_Prefix);

  Clang->createASTContext();                              // needs preprocessor
  astContext_ = &Clang->getASTContext();
  const FileEntry *FileIn = FileMgr.getFile(filename); // needs preprocessor
  SourceMgr.setMainFileID(SourceMgr.createFileID(FileIn, clang::SourceLocation(), clang::SrcMgr::C_User));
  Clang->getDiagnosticClient().BeginSourceFile(Clang->getLangOpts(), &Clang->getPreprocessor());

  NULLASTConsumer TheConsumer; // must pass a consumer in to ParseAST(). This one does nothing
  ParseAST(Clang->getPreprocessor(), &TheConsumer, Clang->getASTContext());
  // Translation Unit is contents of a file
  TranslationUnitDecl *TUD = astContext_->getTranslationUnitDecl();
  chillAST_node *wholefile = ConvertTranslationUnit(TUD, SourceFileName);
  entire_file_AST = (chillAST_SourceFile *) wholefile;
  chill::scanner::DefinitionLinker dl;
  dl.exec(entire_file_AST);
  chill::scanner::SanityCheck sc;
  sc.run(entire_file_AST,std::cout);
  entire_file_AST->print();
  astContext_ = &Clang->getASTContext();
}

IR_clangCode_Global_Init::~IR_clangCode_Global_Init()
{
  /*
  delete pTextDiagnosticPrinter;
  delete diagnostic;
  delete sourceManager;
  delete preprocessor;
  delete idTable;
  delete builtinContext;
  delete astContext_;
  delete astConsumer_;
  */
}


chillAST_NodeList ConvertMemberExpr( clang::MemberExpr *clangME ) {
  debug_fprintf(stderr, "ConvertMemberExpr()\n");

  chillAST_node *base = UNWRAP(ConvertGenericClangAST( clangME->getBase() ));

  DeclarationNameInfo memnameinfo = clangME->getMemberNameInfo();
  DeclarationName DN = memnameinfo.getName();
  const char *member = DN.getAsString().c_str();

  chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, clangME );

  return WRAP(ME);
}



// ----------------------------------------------------------------------------
// Class: IR_clangCode
// ----------------------------------------------------------------------------

void chill::parser::Clang::parse(std::string fname, std::string proc_name) {
  debug_fprintf(stderr, "\nIR_xxxxCode::IR_xxxxCode()\n\n");
  const char *argv[2];
  argv[0] = "chill";
  argv[1] = strdup(fname.c_str());
  // this causes opening and parsing of the file.
  // this is the only call to Instance that has an argument list or file name
  IR_clangCode_Global_Init *pInstance = IR_clangCode_Global_Init::Instance(argv);

  if(pInstance) {
    aClangCompiler *Clang = pInstance->ClangCompiler;
    entire_file_AST = Clang->entire_file_AST;  // ugly that same name, different classes
  } else throw std::runtime_error("Failed to initialize Clang interface, abort!");
}

