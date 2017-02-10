

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
#include "ir_clang.hh"
#include "loop.hh"
#include "chill_error.hh"

#define DUMPFUNC(x, y) std::cerr << "In function " << x << "\n"; y->dump(); 

#include "clang/Frontend/FrontendActions.h"
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/Decl.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Basic/TargetInfo.h>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/Host.h>

#include "code_gen/CG_chillRepr.h"
#include "code_gen/CG_chillBuilder.h"

#include "chill_ast.hh"

// fwd declarations
chillAST_node * ConvertVarDecl( clang::VarDecl *D, chillAST_node * );
chillAST_node * ConvertTypeDefDecl( clang::TypedefDecl *TDD, chillAST_node * );
chillAST_node * ConvertRecordDecl( clang::RecordDecl *D, chillAST_node * );
chillAST_node * ConvertDeclStmt( clang::DeclStmt *clangDS, chillAST_node * );
chillAST_node * ConvertCompoundStmt( clang::CompoundStmt *clangCS, chillAST_node * );
chillAST_node * ConvertFunctionDecl( clang::FunctionDecl *D , chillAST_node *);
chillAST_node * ConvertForStmt( clang::ForStmt *clangFS, chillAST_node * );
chillAST_node * ConvertUnaryOperator( clang::UnaryOperator * clangU, chillAST_node *O ); 
chillAST_node * ConvertBinaryOperator( clang::BinaryOperator * clangBO, chillAST_node *B );
chillAST_node * ConvertArraySubscriptExpr( clang::ArraySubscriptExpr *clangASE, chillAST_node * ); 
chillAST_node * ConvertDeclRefExpr( clang::DeclRefExpr * clangDRE, chillAST_node * );
chillAST_node * ConvertIntegerLiteral( clang::IntegerLiteral *clangIL, chillAST_node * );
chillAST_node * ConvertFloatingLiteral( clang::FloatingLiteral *clangFL, chillAST_node * );
chillAST_node * ConvertImplicitCastExpr( clang::ImplicitCastExpr *clangICE, chillAST_node * );
chillAST_node * ConvertCStyleCastExpr( clang::CStyleCastExpr *clangICE, chillAST_node * );
chillAST_node * ConvertReturnStmt( clang::ReturnStmt *clangRS, chillAST_node * );
chillAST_node * ConvertCallExpr( clang::CallExpr *clangCE , chillAST_node *);
chillAST_node * ConvertIfStmt( clang::IfStmt *clangIS , chillAST_node *);
chillAST_node * ConvertMemberExpr( clang::MemberExpr *clangME , chillAST_node *);


chillAST_node * ConvertTranslationUnit(  clang::TranslationUnitDecl *TUD, char *filename );
chillAST_node * ConvertGenericClangAST( clang::Stmt *s, chillAST_node *  );


using namespace clang;
using namespace clang::driver;
using namespace omega;
using namespace std;

namespace {
  static string binops[] = {
      " ", " ",             // BO_PtrMemD, BO_PtrMemI,       // [C++ 5.5] Pointer-to-member operators.
      "*", "/", "%",        // BO_Mul, BO_Div, BO_Rem,       // [C99 6.5.5] Multiplicative operators.
      "+", "-",             // BO_Add, BO_Sub,               // [C99 6.5.6] Additive operators.
      "<<", ">>",           // BO_Shl, BO_Shr,               // [C99 6.5.7] Bitwise shift operators.
      "<", ">", "<=", ">=", // BO_LT, BO_GT, BO_LE, BO_GE,   // [C99 6.5.8] Relational operators.
      "==", "!=",           // BO_EQ, BO_NE,                 // [C99 6.5.9] Equality operators.
      "&",                  // BO_And,                       // [C99 6.5.10] Bitwise AND operator.
      "^",                 // BO_Xor,                       // [C99 6.5.11] Bitwise XOR operator.
      "|",                  // BO_Or,                        // [C99 6.5.12] Bitwise OR operator.
      "&&",                 // BO_LAnd,                      // [C99 6.5.13] Logical AND operator.
      "||",                 // BO_LOr,                       // [C99 6.5.14] Logical OR operator.
      "=", "*=",            // BO_Assign, BO_MulAssign,      // [C99 6.5.16] Assignment operators.
      "/=", "%=",           // BO_DivAssign, BO_RemAssign,
      "+=", "-=",           // BO_AddAssign, BO_SubAssign,
      "<<=", ">>=",         // BO_ShlAssign, BO_ShrAssign,
      "&&=", "^=",         // BO_AndAssign, BO_XorAssign,
      "||=",                // BO_OrAssign,
      ","};                 // BO_Comma                      // [C99 6.5.17] Comma operator.


  static string unops[] = {
      "++", "--",           // [C99 6.5.2.4] Postfix increment and decrement
      "++", "--",           // [C99 6.5.3.1] Prefix increment and decrement
      "@", "*",            // [C99 6.5.3.2] Address and indirection
      "+", "-",             // [C99 6.5.3.3] Unary arithmetic
      "~", "!",             // [C99 6.5.3.3] Unary arithmetic
      "__real", "__imag",   // "__real expr"/"__imag expr" Extension.
      "__extension"          // __extension__ marker.
  };
}

// forward defs
SourceManager * globalSRCMAN;

chillAST_node * ConvertVarDecl( VarDecl *D, chillAST_node *p ) {
   bool isParm = false;

   QualType T0 = D->getType();
   QualType T  = T0;
   if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
     T = Parm->getOriginalType();
     isParm = true;
   }

  char *vartype =  strdup( T.getAsString().c_str());
  char *arraypart = splitTypeInfo(vartype);

  char *varname = strdup(D->getName().str().c_str()); 

  chillAST_VarDecl * chillVD = new chillAST_VarDecl( vartype,  varname, arraypart, (void *)D, p /* , initializer */ );

  chillVD->isAParameter = isParm; 

  int numdim = 0;
  chillVD-> knownArraySizes = true;
  if (index(vartype, '*')) chillVD->knownArraySizes = false;  // float *a;   for example
  if (index(arraypart, '*'))  chillVD->knownArraySizes = false;
  
  // note: vartype here, arraypart in next code..    is that right?
  if (index(vartype, '*')) { 
    for (int i = 0; i<strlen(vartype); i++) if (vartype[i] == '*') numdim++;
    chillVD->numdimensions = numdim;
  }

  if (index(arraypart, '[')) {  // JUST [12][34][56]  no asterisks
    char *dupe = strdup(arraypart);

    int len = strlen(arraypart);
    for (int i=0; i<len; i++) if (dupe[i] == '[') numdim++;

    chillVD->numdimensions = numdim;
    int *as =  (int *)malloc(sizeof(int *) * numdim );
    if (!as) { 
      debug_fprintf(stderr, "can't malloc array sizes in ConvertVarDecl()\n");
      exit(-1);
    }
    chillVD->arraysizes = as; // 'as' changed later!

    
    char *ptr = dupe;
    while (ptr = index(ptr, '[')) {
      ptr++;
      int dim;
      sscanf(ptr, "%d", &dim);
      *as++ = dim;
      
      ptr =  index(ptr, ']');
    }
    free(dupe);
  }
  
  Expr *Init = D->getInit();
  if (Init) {
    throw std::runtime_error(" = VARDECL HAS INIT.  (TODO) (RIGHT NOW)");
  }

  free (vartype);
  free (varname);

  // store this away for declrefexpr that references it! 
  VariableDeclarations.push_back(chillVD);
  return chillVD;
}



chillAST_node * ConvertRecordDecl( clang::RecordDecl *RD, chillAST_node *p ) { // for structs and unions

  int count = 0;
  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) count++; 

  char blurb[128];
  sprintf(blurb, "struct %s", RD->getNameAsString().c_str()); 
  debug_fprintf(stderr, "blurb is '%s'\n", blurb); 

  chillAST_TypedefDecl *astruct = new chillAST_TypedefDecl( blurb, "", p);
  astruct->setStruct( true ); 
  astruct->setStructName( RD->getNameAsString().c_str() );

  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) { 
    clang::FieldDecl *FD = (*fi);
    FD->dump(); printf(";\n"); fflush(stdout); 
    string TypeStr = FD->getType().getAsString(); 

    const char *typ  = TypeStr.c_str();
    const char *name = FD->getNameAsString().c_str();
    debug_fprintf(stderr, "(typ) %s (name) %s\n", typ, name);

    chillAST_VarDecl *VD = NULL;
    // very clunky and incomplete
    VD = new chillAST_VarDecl( typ, name, "", astruct ); // can't handle arrays yet 
    
    astruct->subparts.push_back(VD); 
  }


  debug_fprintf(stderr, "I just defined a struct\n"); 
  astruct->print(0, stderr); 

  return astruct; 
}


chillAST_node * ConvertTypeDefDecl( TypedefDecl *TDD, chillAST_node *p ) {
  char *under =  strdup( TDD->getUnderlyingType().getAsString().c_str());
  char *arraypart = splitTypeInfo(under);
  char *alias = strdup(TDD->getName().str().c_str());

  chillAST_TypedefDecl *CTDD = new chillAST_TypedefDecl( under, alias, arraypart, p );

  free(under);        
  free(arraypart);   

  return CTDD; 
}



chillAST_node * ConvertDeclStmt( DeclStmt *clangDS, chillAST_node *p ) {
  chillAST_VarDecl *chillvardecl; // the thing we'll return if this is a single declaration
  
  bool multiples = !clangDS->isSingleDecl();

  DeclGroupRef dgr = clangDS->getDeclGroup();
  clang::DeclGroupRef::iterator DI = dgr.begin();
  clang::DeclGroupRef::iterator DE = dgr.end();
  
  for ( ; DI != DE; ++DI) {
    Decl *D = *DI;
    const char *declT =  D->getDeclKindName();
    //debug_fprintf(stderr, "a decl of type %s\n", D->getDeclKindName()); 
    
    if (!strcmp("Var", declT)) {
      VarDecl *V = dyn_cast<VarDecl>(D);
      // ValueDecl *VD = dyn_cast<ValueDecl>(D); // not needed? 
      std::string Name = V->getNameAsString();
      char *varname = strdup( Name.c_str()); 
      
      //debug_fprintf(stderr, "variable named %s\n", Name.c_str()); 
      QualType T = V->getType();
      string TypeStr = T.getAsString();
      char *vartype =  strdup( TypeStr.c_str());
      
      //debug_fprintf(stderr, "%s %s\n", td, varname); 
      char *arraypart = splitTypeInfo( vartype );
      
      chillvardecl = new chillAST_VarDecl(vartype, varname, arraypart, (void *)D, p );
      //debug_fprintf(stderr, "DeclStmt (clang 0x%x) for %s %s%s\n", D, vartype,  varname, arraypart);

      // store this away for declrefexpr that references it! 
      VariableDeclarations.push_back(chillvardecl);
      
      if (multiples) p->addChild( chillvardecl ); 

      // TODO 
      if (V->hasInit()) { 
        debug_fprintf(stderr, " ConvertDeclStmt()  UNHANDLED initialization\n");
        exit(-1); 
      }
    }
  }  // for each of possibly multiple decls 
  
  if (multiples) return NULL;    // multiple decls added themselves already
  return chillvardecl;  // OR a single decl
}



chillAST_node * ConvertCompoundStmt( CompoundStmt *clangCS, chillAST_node *p ) {
  chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt;
  chillCS->setParent(p);
  // for each clang child
  for (auto I = clangCS->child_begin(); I != clangCS->child_end(); ++I) { // ?? loop looks WRONG
    // create the chill ast for each child
    Stmt *child = *I;
    chillAST_node *n =  ConvertGenericClangAST( child, chillCS );
    // usually n will be a statement. We just add it as a child.
    // SOME DeclStmts have multiple declarations. They will add themselves and return NULL
    if (n) chillCS->addChild( n );
  }
  return chillCS;
}

chillAST_node * ConvertFunctionDecl( FunctionDecl *D, chillAST_node *p ) {
  QualType QT = D->getReturnType();
  string ReturnTypeStr = QT.getAsString();

  // Function name
  DeclarationName DeclName = D->getNameInfo().getName();
  string FuncName = DeclName.getAsString();

  chillAST_FunctionDecl *chillFD = new chillAST_FunctionDecl( ReturnTypeStr.c_str(),  FuncName.c_str(), p, D);
  

  int numparams = D->getNumParams();

  for (int i=0; i<numparams; i++) {
    if (i) debug_fprintf(stderr, ", ");
    VarDecl *clangvardecl = D->getParamDecl(i);  // the ith parameter  (CLANG)
    ParmVarDecl *pvd = D->getParamDecl(i); 
    QualType T = pvd->getOriginalType();
    debug_fprintf(stderr, "OTYPE %s\n", T.getAsString().c_str()); 

    chillAST_VarDecl *chillPVD = (chillAST_VarDecl *)ConvertVarDecl( clangvardecl, chillFD ) ; 
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
    //debug_fprintf(stderr, "body of type %s\n", clangbody->getStmtClassName()); 
    //chillAST_node *CB = ConvertCompoundStmt(  dyn_cast<CompoundStmt>(clangbody) ); // always a compound statement?
    chillAST_node *CB = ConvertGenericClangAST( clangbody, chillFD ); 
    //debug_fprintf(stderr, "FunctionDecl body = 0x%x of type %s\n", CB, CB->getTypeString());
    chillFD->setBody ( CB ); 
  }

  //debug_fprintf(stderr, "adding function %s  0x%x to FunctionDeclarations\n", chillFD->functionName, chillFD); 
  FunctionDeclarations.push_back(chillFD); 
  return  chillFD; 
}


chillAST_node * ConvertForStmt( ForStmt *clangFS, chillAST_node *p ) {

  Stmt *init = clangFS->getInit();
  Expr *cond = clangFS->getCond();
  Expr *incr = clangFS->getInc();
  Stmt *body = clangFS->getBody();

  chillAST_node *ini = ConvertGenericClangAST( init, NULL ); 
  chillAST_node *con = ConvertGenericClangAST( cond, NULL); 
  chillAST_node *inc = ConvertGenericClangAST( incr, NULL); 
  chillAST_node *bod = ConvertGenericClangAST( body, NULL); 
  if (bod->getType() != CHILLAST_NODETYPE_COMPOUNDSTMT) {
    //debug_fprintf(stderr, "ForStmt body of type %s\n", bod->getTypeString()); 
    // make single statement loop bodies loop like other loops
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt( );
    cs->addChild( bod );
    bod = cs;
  }


  chillAST_ForStmt *chill_loop = new  chillAST_ForStmt( ini, con, inc, bod, p ); 
  ini->setParent( chill_loop );
  con->setParent( chill_loop );
  inc->setParent( chill_loop );
  bod->setParent( chill_loop );

  return chill_loop; 
}


chillAST_node * ConvertIfStmt( IfStmt *clangIS, chillAST_node *p ) {
  Expr *cond = clangIS->getCond();
  Stmt *thenpart = clangIS->getThen();
  Stmt *elsepart = clangIS->getElse();
  
  chillAST_node *con = ConvertGenericClangAST( cond, NULL);
  chillAST_node *thn = NULL;
  if (thenpart) thn = ConvertGenericClangAST( thenpart, NULL);
  chillAST_node *els = NULL;
  if (elsepart) els = ConvertGenericClangAST( elsepart, NULL);
  
  chillAST_IfStmt *ifstmt = new chillAST_IfStmt( con, thn, els, NULL);
  return ifstmt; 
}



chillAST_node * ConvertUnaryOperator( UnaryOperator * clangUO, chillAST_node *p ) {
  const char *op = unops[clangUO->getOpcode()].c_str();
  bool pre = clangUO->isPrefix();
  chillAST_node *sub = ConvertGenericClangAST( clangUO->getSubExpr(), NULL ); 

  chillAST_UnaryOperator *chillUO = new chillAST_UnaryOperator( op, pre, sub, p ); 
  sub->setParent( chillUO );
  return chillUO; 
}


chillAST_node * ConvertBinaryOperator( BinaryOperator * clangBO, chillAST_node *p ) {

  // get the clang parts
  Expr *lhs = clangBO->getLHS();
  Expr *rhs = clangBO->getRHS();
  BinaryOperator::Opcode op = clangBO->getOpcode(); // this is CLANG op, not CHILL op


  // convert to chill equivalents
  chillAST_node *l = ConvertGenericClangAST( lhs, NULL ); 
  const char *opstring = binops[op].c_str();
  chillAST_node *r = ConvertGenericClangAST( rhs, NULL ); 
  // TODO chill equivalent for numeric op. 

  // build up the chill Binary Op AST node
  chillAST_BinaryOperator * binop = new chillAST_BinaryOperator( l, opstring, r, p );
  l->setParent( binop );
  r->setParent( binop );

  return binop; 
}




chillAST_node * ConvertArraySubscriptExpr( ArraySubscriptExpr *clangASE, chillAST_node *p ) { 

  Expr *clangbase  = clangASE->getBase();
  Expr *clangindex = clangASE->getIdx();
  //debug_fprintf(stderr, "clang base: "); clangbase->dump(); debug_fprintf(stderr, "\n"); 

  chillAST_node *bas  = ConvertGenericClangAST( clangbase, NULL ); 
  chillAST_node *indx = ConvertGenericClangAST( clangindex, NULL ); 
  
  chillAST_ArraySubscriptExpr * chillASE = new chillAST_ArraySubscriptExpr( bas, indx, p, clangASE);
  bas->setParent( chillASE );
  indx->setParent( chillASE );
  return chillASE; 
}



chillAST_node * ConvertDeclRefExpr( DeclRefExpr * clangDRE, chillAST_node *p ) { 
  DeclarationNameInfo DNI = clangDRE->getNameInfo();

  ValueDecl *vd = static_cast<ValueDecl *>(clangDRE->getDecl()); // ValueDecl ?? VarDecl ??

  QualType QT = vd->getType();
  string TypeStr = QT.getAsString();
  //debug_fprintf(stderr, "\n\n*** type %s ***\n\n", TypeStr.c_str()); 
  //debug_fprintf(stderr, "kind %s\n", vd->getDeclKindName()); 

  DeclarationName DN = DNI.getName();
  const char *varname = DN.getAsString().c_str() ; 
  chillAST_DeclRefExpr * chillDRE = new chillAST_DeclRefExpr(TypeStr.c_str(),  varname, p ); 

  //debug_fprintf(stderr, "clang DeclRefExpr refers to declaration of %s @ 0x%x\n", varname, vd);
  //debug_fprintf(stderr, "clang DeclRefExpr refers to declaration of %s of kind %s\n", varname, vd->getDeclKindName()); 
  
  // find the definition (we hope)
  if ( (!strcmp("Var",  vd->getDeclKindName())) || (!strcmp("ParmVar",  vd->getDeclKindName()))) { 
    // it's a variable reference 
    int numvars = VariableDeclarations.size();
    chillAST_VarDecl *chillvd = NULL;
    for (int i=0; i<numvars; i++) { 
      if (VariableDeclarations[i]->uniquePtr == vd) {
        chillvd = VariableDeclarations[i];
        //debug_fprintf(stderr, "found it at variabledeclaration %d of %d\n", i, numvars);
      }
    }
    if (!chillvd) { 
      debug_fprintf(stderr, "\nWARNING, ir_clang.cc clang DeclRefExpr %s refers to a declaration I can't find! at ox%x\n", varname, vd); 
      debug_fprintf(stderr, "variables I know of are:\n");
      for (int i=0; i<numvars; i++) { 
        chillAST_VarDecl *adecl = VariableDeclarations[i];
        if (adecl->isParmVarDecl()) debug_fprintf(stderr, "(parameter) ");
        debug_fprintf(stderr, "%s %s at location 0x%x\n", adecl->vartype, adecl->varname, adecl->uniquePtr); 
      }  
      debug_fprintf(stderr, "\n"); 
    }
    
    if (chillvd == NULL) { debug_fprintf(stderr, "chillDRE->decl = 0x%x\n", chillvd); exit(-1); }

    chillDRE->decl = (chillAST_node *)chillvd; // start of spaghetti pointers ...
  }
  else  if (!strcmp("Function",  vd->getDeclKindName())) { 
    //debug_fprintf(stderr, "declrefexpr of type Function\n");
    int numfuncs = FunctionDeclarations.size();
    chillAST_FunctionDecl *chillfd = NULL;
    for (int i=0; i<numfuncs; i++) { 
      if (FunctionDeclarations[i]->uniquePtr == vd) {
        chillfd = FunctionDeclarations[i];
        //debug_fprintf(stderr, "found it at functiondeclaration %d of %d\n", i, numfuncs);
      }
    }
    if (chillfd == NULL) { debug_fprintf(stderr, "chillDRE->decl = 0x%x\n", chillfd); exit(-1); }

    chillDRE->decl = (chillAST_node *)chillfd; // start of spaghetti pointers ...
    
  }
  else { 
  debug_fprintf(stderr, "clang DeclRefExpr refers to declaration of %s of kind %s\n", varname, vd->getDeclKindName()); 
    debug_fprintf(stderr, "chillDRE->decl = UNDEFINED\n"); 
    exit(-1); 
  }

  //debug_fprintf(stderr, "%s\n", DN.getAsString().c_str()); 
  return chillDRE; 
}



chillAST_node * ConvertIntegerLiteral( IntegerLiteral *clangIL, chillAST_node *p ) { 
  bool isSigned = clangIL->getType()->isSignedIntegerType();
  //int val = clangIL->getIntValue();
  const char *printable = clangIL->getValue().toString(10, isSigned).c_str(); 
  int val = atoi( printable ); 
  //debug_fprintf(stderr, "int value %s  (%d)\n", printable, val); 
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val, p );
  return chillIL; 
}


chillAST_node * ConvertFloatingLiteral( FloatingLiteral *clangFL, chillAST_node *p ) { 
  //debug_fprintf(stderr, "\nConvertFloatingLiteral()\n"); 
  float val = clangFL->getValueAsApproximateDouble(); // TODO approx is a bad idea!
  string WHAT; 
  SmallString<16> Str;
  clangFL->getValue().toString( Str );
  const char *printable = Str.c_str(); 
  //debug_fprintf(stderr, "literal %s\n", printable); 

  SourceLocation sloc = clangFL->getLocStart();
  SourceLocation eloc = clangFL->getLocEnd();

  std::string start = sloc.printToString( *globalSRCMAN ); 
  std::string end   = eloc.printToString( *globalSRCMAN ); 
  //debug_fprintf(stderr, "literal try2 start %s end %s\n", start.c_str(), end.c_str()); 
  //printlines( sloc, eloc, globalSRCMAN ); 
  unsigned int startlineno = globalSRCMAN->getPresumedLineNumber( sloc );
  unsigned int   endlineno = globalSRCMAN->getPresumedLineNumber( eloc ); ;
  const char     *filename = globalSRCMAN->getBufferName( sloc );

  std::string  fname = globalSRCMAN->getFilename( sloc );
  //debug_fprintf(stderr, "fname %s\n", fname.c_str()); 

  if (filename && strlen(filename) > 0) {} // debug_fprintf(stderr, "literal file '%s'\n", filename);
  else { 
    debug_fprintf(stderr, "\nConvertFloatingLiteral() filename is NULL?\n"); 

    //sloc =  globalSRCMAN->getFileLoc( sloc );  // should get spelling loc? 
    sloc =  globalSRCMAN->getSpellingLoc( sloc );  // should get spelling loc? 
    //eloc =  globalSRCMAN->getFileLoc( eloc );

    start = sloc.printToString( *globalSRCMAN ); 
    //end   = eloc.printToString( *globalSRCMAN );  
    //debug_fprintf(stderr, "literal try3 start %s end %s\n", start.c_str(), end.c_str()); 
   
    startlineno = globalSRCMAN->getPresumedLineNumber( sloc );
    //endlineno = globalSRCMAN->getPresumedLineNumber( eloc ); ;    
    //debug_fprintf(stderr, "start, end line numbers %d %d\n", startlineno, endlineno); 
    
    filename = globalSRCMAN->getBufferName( sloc );

    //if (globalSRCMAN->isMacroBodyExpansion( sloc )) { 
    //  debug_fprintf(stderr, "IS MACRO\n");
    //} 
  }
  
  unsigned int  offset = globalSRCMAN->getFileOffset( sloc );
  //debug_fprintf(stderr, "literal file offset %d\n", offset); 

  FILE *fp = fopen (filename, "r");
  fseek(fp, offset, SEEK_SET); // go to the part of the file where the float is defined
  
  char buf[10240];
  fgets (buf, sizeof(buf), fp); // read a line starting where the float starts
  fclose(fp);

  // buf has the line we want   grab the float constant out of it
  //debug_fprintf(stderr, "\nbuf '%s'\n", buf);
  char *ptr = buf;
  if (*ptr == '-') ptr++; // ignore possible minus sign
  int len = strspn(ptr, ".-0123456789f"); 
  buf[len] = '\0';
  //debug_fprintf(stderr, "'%s'\n", buf);

  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val, buf, p );
  
  //chillFL->print(); printf("\n"); fflush(stdout); 
  return chillFL; 
}


chillAST_node * ConvertImplicitCastExpr( ImplicitCastExpr *clangICE, chillAST_node *p ) {
  //debug_fprintf(stderr, "ConvertImplicitCastExpr()\n"); 
  CastExpr *CE = dyn_cast<ImplicitCastExpr>(clangICE);
  //debug_fprintf(stderr, "implicit cast of type %s\n", CE->getCastKindName());
  chillAST_node * sub = ConvertGenericClangAST( clangICE->getSubExpr(), p );
  chillAST_ImplicitCastExpr *chillICE = new chillAST_ImplicitCastExpr( sub, p ); 
  
  //sub->setParent( chillICE ); // these 2 lines work
  //return chillICE; 

  //sub->setParent(p);         // ignore the ImplicitCastExpr !!  TODO (probably a bad idea) 
  return sub; 

}




chillAST_node * ConvertCStyleCastExpr( CStyleCastExpr *clangCSCE, chillAST_node *p ) {
  //debug_fprintf(stderr, "ConvertCStyleCastExpr()\n"); 
  //debug_fprintf(stderr, "C Style cast of kind ");
  CastExpr *CE = dyn_cast<CastExpr>(clangCSCE);
  //debug_fprintf(stderr, "%s\n", CE->getCastKindName());
  
  //clangCSCE->getTypeAsWritten().getAsString(Policy)
  const char * towhat = strdup( clangCSCE->getTypeAsWritten().getAsString().c_str() );
  //debug_fprintf(stderr, "before sub towhat (%s)\n", towhat);

  chillAST_node * sub = ConvertGenericClangAST( clangCSCE->getSubExprAsWritten(), NULL );
  //debug_fprintf(stderr, "after sub towhat (%s)\n", towhat);
  chillAST_CStyleCastExpr *chillCSCE = new chillAST_CStyleCastExpr( towhat, sub, p ); 
  //debug_fprintf(stderr, "after CSCE towhat (%s)\n", towhat);
  sub->setParent( chillCSCE );
  return chillCSCE; 
}




chillAST_node * ConvertReturnStmt( ReturnStmt *clangRS, chillAST_node *p ) {
  chillAST_node * retval = ConvertGenericClangAST( clangRS->getRetValue(), NULL ); // NULL is handled
  //if (retval == NULL) debug_fprintf(stderr, "return stmt returns nothing\n");

  chillAST_ReturnStmt * chillRS = new chillAST_ReturnStmt( retval, p );
  if (retval) retval->setParent( chillRS );
  return chillRS; 
}


chillAST_node * ConvertCallExpr( CallExpr *clangCE, chillAST_node *p ) {
  //debug_fprintf(stderr, "ConvertCallExpr()\n"); 

  chillAST_node *callee = ConvertGenericClangAST( clangCE->getCallee(), NULL ); 
  //debug_fprintf(stderr, "callee is of type %s\n", callee->getTypeString()); 

  //chillAST_node *next = ((chillAST_ImplicitCastExpr *)callee)->subexpr;
  //debug_fprintf(stderr, "callee is of type %s\n", next->getTypeString()); 

  chillAST_CallExpr *chillCE = new chillAST_CallExpr( callee, p ); 
  callee->setParent( chillCE );

  int numargs = clangCE->getNumArgs();
  //debug_fprintf(stderr, "CallExpr has %d args\n", numargs);
  Expr **clangargs =  clangCE->getArgs(); 
  for (int i=0; i<numargs; i++) { 
    chillCE->addArg( ConvertGenericClangAST( clangargs[i], chillCE ) );
  }
  
  return chillCE; 
}


chillAST_node * ConvertParenExpr( ParenExpr *clangPE, chillAST_node *p ) {
  chillAST_node *sub = ConvertGenericClangAST( clangPE->getSubExpr(), NULL);
  chillAST_ParenExpr *chillPE = new chillAST_ParenExpr( sub, p); 
  sub->setParent( chillPE );

  return chillPE; 
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
    chillAST_node *child;
       
    if (isa<FunctionDecl>(D))
      child = ConvertFunctionDecl( dyn_cast<FunctionDecl>(D), topnode );
    else if (isa<VarDecl>(D))
      child = ConvertVarDecl( dyn_cast<VarDecl>(D), topnode );
    else if (isa<TypedefDecl>(D))
      child = ConvertTypeDefDecl( dyn_cast<TypedefDecl>(D), topnode );
    else if (isa<RecordDecl>(D))
      child = ConvertRecordDecl( dyn_cast<RecordDecl>(D), topnode );
    else if (isa<TypeAliasDecl>(D))
      throw std::runtime_error("TUD TypeAliasDecl  TODO \n");
    else
      throw std::runtime_error(std::string("TUD a declaration of type which I can't handle: ") + D->getDeclKindName());

    topnode->addChild(child);
    if (D->isImplicit() || !globalSRCMAN->getFilename(D->getLocation()).equals(filename))
      child->isFromSourceFile = false;
  }
  return ( chillAST_node *)  topnode;
}



 chillAST_node * ConvertGenericClangAST( Stmt *s, chillAST_node *p ) {
   
   if (s == NULL) return NULL;
   //debug_fprintf(stderr, "\nConvertGenericClangAST() Stmt of type %d (%s)\n", s->getStmtClass(),s->getStmtClassName()); 
   Decl *D = (Decl *) s;
   //if (isa<Decl>(D)) debug_fprintf(stderr, "Decl of kind %d (%s)\n",  D->getKind(),D->getDeclKindName() );
   

   chillAST_node *ret = NULL;

   if (isa<CompoundStmt>(s))              {ret = ConvertCompoundStmt( dyn_cast<CompoundStmt>(s),p); 
   } else if (isa<DeclStmt>(s))           {ret = ConvertDeclStmt(dyn_cast<DeclStmt>(s),p); 
   } else if (isa<ForStmt>(s))            {ret = ConvertForStmt(dyn_cast<ForStmt>(s),p);
   } else if (isa<BinaryOperator>(s))     {ret = ConvertBinaryOperator(dyn_cast<BinaryOperator>(s),p);
   } else if (isa<ArraySubscriptExpr>(s)) {ret = ConvertArraySubscriptExpr(dyn_cast<ArraySubscriptExpr>(s),p);
   } else if (isa<DeclRefExpr>(s))        {ret = ConvertDeclRefExpr(dyn_cast<DeclRefExpr>(s),p); 
   } else if (isa<FloatingLiteral>(s))    {ret = ConvertFloatingLiteral(dyn_cast<FloatingLiteral>(s),p);
   } else if (isa<IntegerLiteral>(s))     {ret = ConvertIntegerLiteral(dyn_cast<IntegerLiteral>(s),p);
   } else if (isa<UnaryOperator>(s))      {ret = ConvertUnaryOperator(dyn_cast<UnaryOperator>(s),p);
   } else if (isa<ImplicitCastExpr>(s))   {ret = ConvertImplicitCastExpr(dyn_cast<ImplicitCastExpr>(s),p);
   } else if (isa<CStyleCastExpr>(s))     {ret = ConvertCStyleCastExpr(dyn_cast<CStyleCastExpr>(s),p);
   } else if (isa<ReturnStmt>(s))         {ret = ConvertReturnStmt(dyn_cast<ReturnStmt>(s),p); 
   } else if (isa<CallExpr>(s))           {ret = ConvertCallExpr(dyn_cast<CallExpr>(s),p); 
   } else if (isa<ParenExpr>(s))          {ret = ConvertParenExpr(dyn_cast<ParenExpr>(s),p); 
   } else if (isa<IfStmt>(s))             {ret = ConvertIfStmt(dyn_cast<IfStmt>(s),p);
   } else if (isa<MemberExpr>(s))         {ret = ConvertMemberExpr(dyn_cast<MemberExpr>(s),p);


     // these can only happen at the top level? 
     //   } else if (isa<FunctionDecl>(D))       { ret = ConvertFunctionDecl( dyn_cast<FunctionDecl>(D)); 
     //} else if (isa<VarDecl>(D))            { ret =      ConvertVarDecl( dyn_cast<VarDecl>(D) ); 
     //} else if (isa<TypedefDecl>(D))        { ret =  ConvertTypeDefDecl( dyn_cast<TypedefDecl>(D)); 
     //  else if (isa<TranslationUnitDecl>(s))  // need filename 




     //   } else if (isa<>(s))                  {         Convert ( dyn_cast<>(s)); 

     /*
     */

   } else {
     // more work to do his->chillvd == l_that->chillvd;
     debug_fprintf(stderr, "ir_clang.cc ConvertGenericClangAST() UNHANDLED ");
     //if (isa<Decl>(D)) debug_fprintf(stderr, "Decl of kind %s\n",  D->getDeclKindName() );
     if (isa<Stmt>(s))debug_fprintf(stderr, "Stmt of type %s\n", s->getStmtClassName()); 
     exit(-1); 
   }
   
   return ret; 
 }

class NULLASTConsumer : public ASTConsumer
{
};


// ----------------------------------------------------------------------------
// Class: IR_clangCode_Global_Init
// ----------------------------------------------------------------------------

IR_clangCode_Global_Init *IR_clangCode_Global_Init::pinstance = 0;


IR_clangCode_Global_Init *IR_clangCode_Global_Init::Instance(char **argv) {
  debug_fprintf(stderr, "in IR_clangCode_Global_Init::Instance(), "); 
  if (pinstance == 0) {  
    //debug_fprintf(stderr, "\n\n***  making the one and only instance ***\n\n\n"); 
    // this is the only way to create an IR_clangCode_Global_Init
    pinstance = new IR_clangCode_Global_Init; 
    pinstance->ClangCompiler = new aClangCompiler( argv[1] );

  }
  //debug_fprintf(stderr, "leaving  IR_clangCode_Global_Init::Instance()\n"); 
  return pinstance;
}


aClangCompiler::aClangCompiler( char *filename ) {
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
  std::unique_ptr<clang::CompilerInvocation> CI(new clang::CompilerInvocation);
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
  fileManager = &FileMgr;
  Clang->createSourceManager(FileMgr);
  SourceManager &SourceMgr = Clang->getSourceManager();
  sourceManager = &SourceMgr; // ?? aclangcompiler copy
  globalSRCMAN = &SourceMgr; //  TODO   global bad

  Clang->setInvocation(CI.get()); // Replace the current invocation

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
  // create another AST, very similar to the clang AST but not written by idiots
  chillAST_node *wholefile = ConvertTranslationUnit(TUD, filename);
  entire_file_AST = (chillAST_SourceFile *) wholefile;
  astContext_ = &Clang->getASTContext();
}




chillAST_FunctionDecl*  aClangCompiler::findprocedurebyname( char *procname ) {

  //debug_fprintf(stderr, "searching through files in the clang AST\n\n");
  //debug_fprintf(stderr, "astContext_  0x%x\n", astContext_);

  vector<chillAST_node*> procs;
  findmanually( entire_file_AST, procname, procs );

  //debug_fprintf(stderr, "procs has %d members\n", procs.size());

  if ( procs.size() == 0 ) { 
    debug_fprintf(stderr, "could not find function named '%s' in AST from file %s\n", procname, SourceFileName);
    exit(-1);
  }
  
  if ( procs.size() > 1 ) { 
    debug_fprintf(stderr, "oddly, found %d functions named '%s' in AST from file %s\n", procs.size(), procname, SourceFileName);
    debug_fprintf(stderr, "I am unsure what to do\n"); 
    exit(-1);
  }

  debug_fprintf(stderr, "found the procedure named %s\n", procname); 
  return (chillAST_FunctionDecl *)procs[0];

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



// ----------------------------------------------------------------------------
// Class: IR_clangCode
// ----------------------------------------------------------------------------

IR_clangCode::IR_clangCode(const char *fname, const char *proc_name, const char *dest_name) : IR_chillCode() {
  debug_fprintf(stderr, "\nIR_xxxxCode::IR_xxxxCode()\n\n"); 
  //debug_fprintf(stderr, "IR_clangCode::IR_clangCode( filename %s, procedure %s )\n", filename, proc_name);
  
  filename = strdup(fname); // filename is internal to IR_clangCode
  procedurename = strdup(proc_name);
  if (dest_name)
    outputname = strdup(dest_name);
  int argc = 2;
  char *argv[2];
  argv[0] = "chill";
  argv[1] = strdup(filename);
  
  // this causes opening and parsing of the file.
  // this is the only call to Instance that has an argument list or file name 
  IR_clangCode_Global_Init *pInstance = IR_clangCode_Global_Init::Instance(argv);
  
  if(pInstance) {
    
    aClangCompiler *Clang = pInstance->ClangCompiler;
    pInstance->setCurrentFunction( NULL );  // we have no function AST yet
    entire_file_AST = Clang->entire_file_AST;  // ugly that same name, different classes
    chillAST_FunctionDecl *localFD = Clang->findprocedurebyname( strdup(proc_name) );   // stored locally
    pInstance->setCurrentFunction( localFD );

    ocg_ = new omega::CG_chillBuilder(localFD->getSourceFile(), localFD);  // ocg == omega code gen
    chillfunc =  localFD; 
  }
}


IR_clangCode::~IR_clangCode() {
}


chillAST_node * ConvertMemberExpr( clang::MemberExpr *clangME , chillAST_node *) { 
  debug_fprintf(stderr, "ConvertMemberExpr()\n"); 
  
  clang::Expr *E = clangME->getBase(); 
  E->dump();

  chillAST_node *base = ConvertGenericClangAST( clangME->getBase(), NULL );

  DeclarationNameInfo memnameinfo = clangME->getMemberNameInfo(); 
  DeclarationName DN = memnameinfo.getName();
  const char *member = DN.getAsString().c_str();

  chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, NULL, clangME ); 

  debug_fprintf(stderr, "this is the Member Expresion\n"); 
  ME->print(); 
  debug_fprintf(stderr, "\n"); 

  return ME; 
  
} 
