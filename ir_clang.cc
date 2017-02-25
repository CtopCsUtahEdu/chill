

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
#include "scanner/sanityCheck.h"
#include "scanner/definitionLinker.h"

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


chillAST_node * ConvertTranslationUnit(  clang::TranslationUnitDecl *TUD, char *filename );
chillAST_NodeList ConvertGenericClangAST( clang::Stmt *s );


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

chillAST_NodeList ConvertVarDecl( VarDecl *D ) {
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

  chillAST_VarDecl * chillVD = new chillAST_VarDecl( vartype,  varname, arraypart, (void *)D );

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
      // ValueDecl *VD = dyn_cast<ValueDecl>(D); // not needed? 
      std::string Name = V->getNameAsString();
      char *varname = strdup( Name.c_str()); 
      
      //debug_fprintf(stderr, "variable named %s\n", Name.c_str()); 
      QualType T = V->getType();
      string TypeStr = T.getAsString();
      char *vartype =  strdup( TypeStr.c_str());
      
      //debug_fprintf(stderr, "%s %s\n", td, varname); 
      char *arraypart = splitTypeInfo( vartype );
      
      chillvardecl = new chillAST_VarDecl(vartype, varname, arraypart, (void *)D );
      //debug_fprintf(stderr, "DeclStmt (clang 0x%x) for %s %s%s\n", D, vartype,  varname, arraypart);

      nl.push_back(chillvardecl);

      // TODO 
      if (V->hasInit()) { 
        debug_fprintf(stderr, " ConvertDeclStmt()  UNHANDLED initialization\n");
        exit(-1); 
      }
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
  const char *op = unops[clangUO->getOpcode()].c_str();
  bool pre = clangUO->isPrefix();
  chillAST_node *sub = UNWRAP(ConvertGenericClangAST( clangUO->getSubExpr()));

  chillAST_UnaryOperator *chillUO = new chillAST_UnaryOperator( op, pre, sub );
  sub->setParent( chillUO );
  return WRAP(chillUO);
}


chillAST_NodeList ConvertBinaryOperator( BinaryOperator * clangBO ) {

  // get the clang parts
  Expr *lhs = clangBO->getLHS();
  Expr *rhs = clangBO->getRHS();
  BinaryOperator::Opcode op = clangBO->getOpcode(); // this is CLANG op, not CHILL op


  // convert to chill equivalents
  chillAST_node *l = UNWRAP(ConvertGenericClangAST( lhs ));
  const char *opstring = binops[op].c_str();
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
  //int val = clangIL->getIntValue();
  const char *printable = clangIL->getValue().toString(10, isSigned).c_str(); 
  int val = atoi( printable ); 
  //debug_fprintf(stderr, "int value %s  (%d)\n", printable, val); 
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val );
  return WRAP(chillIL);
}


chillAST_NodeList ConvertFloatingLiteral( FloatingLiteral *clangFL ) {
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

  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val, buf );
  
  //chillFL->print(); printf("\n"); fflush(stdout); 
  return WRAP(chillFL);
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
  //debug_fprintf(stderr, "CallExpr has %d args\n", numargs);
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
   //debug_fprintf(stderr, "\nConvertGenericClangAST() Stmt of type %d (%s)\n", s->getStmtClass(),s->getStmtClassName());
   Decl *D = (Decl *) s;
   //if (isa<Decl>(D)) debug_fprintf(stderr, "Decl of kind %d (%s)\n",  D->getKind(),D->getDeclKindName() );

   if (isa<CompoundStmt>(s))              {ret = ConvertCompoundStmt( dyn_cast<CompoundStmt>(s));
   } else if (isa<DeclStmt>(s))           {ret = ConvertDeclStmt(dyn_cast<DeclStmt>(s));
   } else if (isa<ForStmt>(s))            {ret = ConvertForStmt(dyn_cast<ForStmt>(s));
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


IR_clangCode_Global_Init *IR_clangCode_Global_Init::Instance(const char **argv) {
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
  chillAST_node *wholefile = ConvertTranslationUnit(TUD, SourceFileName);
  entire_file_AST = (chillAST_SourceFile *) wholefile;
  chill::scanner::DefinitionLinker dl;
  dl.exec(entire_file_AST);
  chill::scanner::SanityCheck sc;
  sc.run(entire_file_AST,std::cout);
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
  const char *argv[2];
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


chillAST_NodeList ConvertMemberExpr( clang::MemberExpr *clangME ) {
  debug_fprintf(stderr, "ConvertMemberExpr()\n"); 
  
  clang::Expr *E = clangME->getBase(); 
  E->dump();

  chillAST_node *base = UNWRAP(ConvertGenericClangAST( clangME->getBase() ));

  DeclarationNameInfo memnameinfo = clangME->getMemberNameInfo(); 
  DeclarationName DN = memnameinfo.getName();
  const char *member = DN.getAsString().c_str();

  chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, clangME );

  debug_fprintf(stderr, "this is the Member Expresion\n"); 
  ME->print(); 
  debug_fprintf(stderr, "\n"); 

  return WRAP(ME);
  
} 
