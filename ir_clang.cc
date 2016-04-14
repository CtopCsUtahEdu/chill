

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

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/OwningPtr.h>
#include <llvm/Module.h>
#include <llvm/Support/Host.h>

#include "code_gen/CG_chillRepr.h"
#include "code_gen/CG_chillBuilder.h"

#include "chill_ast.hh"

// TODO move to ir_clang.hh
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


vector<chillAST_VarDecl *> VariableDeclarations; 
vector<chillAST_FunctionDecl *> FunctionDeclarations; 

using namespace clang;
using namespace clang::driver;
using namespace omega;
using namespace std;


static string binops[] = {
  " ", " ",             // BO_PtrMemD, BO_PtrMemI,       // [C++ 5.5] Pointer-to-member operators.
  "*", "/", "%",        // BO_Mul, BO_Div, BO_Rem,       // [C99 6.5.5] Multiplicative operators.
  "+", "-",             // BO_Add, BO_Sub,               // [C99 6.5.6] Additive operators.
  "<<", ">>",           // BO_Shl, BO_Shr,               // [C99 6.5.7] Bitwise shift operators.
  "<", ">", "<=", ">=", // BO_LT, BO_GT, BO_LE, BO_GE,   // [C99 6.5.8] Relational operators.
  "==", "!=",           // BO_EQ, BO_NE,                 // [C99 6.5.9] Equality operators.
  "&",                  // BO_And,                       // [C99 6.5.10] Bitwise AND operator.
  "??",                 // BO_Xor,                       // [C99 6.5.11] Bitwise XOR operator.
  "|",                  // BO_Or,                        // [C99 6.5.12] Bitwise OR operator.
  "&&",                 // BO_LAnd,                      // [C99 6.5.13] Logical AND operator.
  "||",                 // BO_LOr,                       // [C99 6.5.14] Logical OR operator.
  "=", "*=",            // BO_Assign, BO_MulAssign,      // [C99 6.5.16] Assignment operators.
  "/=", "%=",           // BO_DivAssign, BO_RemAssign,
  "+=", "-=",           // BO_AddAssign, BO_SubAssign,
  "???", "???",         // BO_ShlAssign, BO_ShrAssign,
  "&&=", "???",         // BO_AndAssign, BO_XorAssign,
  "||=",                // BO_OrAssign,
  ","};                 // BO_Comma                      // [C99 6.5.17] Comma operator.


static string unops[] = {
  "++", "--",           // [C99 6.5.2.4] Postfix increment and decrement
  "++", "--",           // [C99 6.5.3.1] Prefix increment and decrement
  "@",  "*",            // [C99 6.5.3.2] Address and indirection
  "+", "-",             // [C99 6.5.3.3] Unary arithmetic
  "~", "!",             // [C99 6.5.3.3] Unary arithmetic
  "__real", "__imag",   // "__real expr"/"__imag expr" Extension.
  "__extension"          // __extension__ marker.
};

// forward defs
SourceManager * globalSRCMAN;  // ugly. shame. 

char *splitTypeInfo( char *underlyingtype );
void printsourceline( const char *filename, int line  );
void printlines(  SourceLocation &S, SourceLocation &E, SourceManager *SRCMAN );

void PrintBinaryOperator(  Stmt *s,         SourceManager *SRCMAN, int level );
void PrintDeclStmt(        Stmt *s,         SourceManager *SRCMAN, int level );
void PrintALoop(            Stmt *L,        SourceManager *SRCMAN, int level );
void PrintAUnaryOperator(   Stmt *s,        SourceManager *SRCMAN, int level );
void PrintAnIfStmt(         Stmt *s,        SourceManager *SRCMAN, int level );
void PrintCompoundStmt(     Stmt *s,        SourceManager *SRCMAN, int level );
void PrintDeclRefExpr(      Stmt *s,        SourceManager *SRCMAN, int level );
void PrintImplicitCastExpr( Stmt *s,        SourceManager *SRCMAN, int level );
void PrintReturnStmt(       Stmt *s,        SourceManager *SRCMAN, int level );
void PrintStmt(             Stmt *s,        SourceManager *SRCMAN, int level );

void PrintAnIntegerLiteral( Stmt *I );
void PrintAFloatingLiteral( Stmt *s );

void PrintFunctionDecl( FunctionDecl *D, SourceManager *SRCMAN, int level );
void PrintTypeDefDecl(  TypedefDecl *D,  SourceManager *SRCMAN, int level );
void PrintVarDecl(      VarDecl *D,      SourceManager *SRCMAN, int level );





void printlines(  SourceLocation &S, SourceLocation &E, SourceManager *SRCMAN ) {
  unsigned int startlineno = SRCMAN->getPresumedLineNumber( S );
  unsigned int   endlineno = SRCMAN->getPresumedLineNumber( E );
  const char     *filename = SRCMAN->getBufferName( S );
  fprintf(stderr, "\n");
  for (int l=startlineno; l<= endlineno; l++) printsourceline( filename, l);
}


void Indent( int level ) {
  for (int i=0; i<level; i++) fprintf(stderr, "    ");
}


char *splitTypeInfo( char *underlyingtype ) {
  char *arraypart = strdup("");  // leak
  if (index(underlyingtype, '[')) {
    // looks like an array 
    free(arraypart);
    char *start = index(underlyingtype, '[');
    arraypart = strdup( start );
    if (*(start-1) == ' ') start--;  // hack 
    *start = '\0';
  }
  return arraypart;  // leak unless caller frees this 
}


// really slow and bad. but I'm debugging
void printsourceline( const char *filename, int line  )
{
  FILE *fp = fopen (filename, "r");

  // Now read lines up to and including the line we want.
  char buf[10240];
  int l = 0;
  while (l < line) {
    if (fgets (buf, sizeof(buf), fp))
      ++l;
    else
      break;
  }
  fclose(fp);

  fprintf(stderr, "*  %s", buf);
}



void PrintBinaryOperator( Stmt *s, SourceManager *SRCMAN, int level ) {  // SOMETIMES SHOULD HAVE ; + return
  //rintf(stderr, "\nBinaryOperator(%d) ", level); 
  BinaryOperator *b = cast<BinaryOperator>(s);

  BinaryOperator::Opcode op = b->getOpcode();
  Expr *lhs = b->getLHS();
  Expr *rhs = b->getRHS();

  //fprintf(stderr, "binaryoperator lhs has type %s\n",   lhs->getStmtClassName());
  //fprintf(stderr, "binaryoperator rhs has type %s\n\n", rhs->getStmtClassName()); 

  PrintStmt( lhs, SRCMAN, level+1 );
  fprintf(stderr, " %s ",  binops[op].c_str());
  PrintStmt( rhs, SRCMAN, level+1);
  if (level == 1) fprintf(stderr, ";\n");
}



void PrintDeclStmt( Stmt *s, SourceManager *SRCMAN, int level ) {
  fprintf(stderr, "\nDeclaration Statement(%d)", level); 
  DeclStmt *D = cast<DeclStmt>(s);

      //QualType QT = D->getType();
      //string TypeStr = QT.getAsString();
      //fprintf(stderr, "type %s\n", TypeStr,c_str()); 

      //SourceLocation S = D->getStartLoc();
      //SourceLocation E = D->getEndLoc();
      //printlines(S, E, SRCMAN); 

      if (D->isSingleDecl()) {
        fprintf(stderr, "this is a single definition\n");
        Decl *d = D->getSingleDecl();
      }
      else {
        fprintf(stderr, "this is NOT a single definition\n");
        DeclGroupRef dg = D->getDeclGroup();
      }

      for (DeclStmt::decl_iterator DI = D->decl_begin(), DE = D->decl_end(); DI != DE; ++DI) {
        //fprintf(stderr, "a def\n"); 
        Decl *d = *DI;
        //fprintf(stderr, "\nstatement of type %s\n", d->getStmtClassName()); 
        //std::cout << (void *) d << "?";
        if (ValueDecl *VD = dyn_cast<ValueDecl>(d)) {
          if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
            if (V->getStorageClass() != SC_None) {
              fprintf(stderr, "%s ", VarDecl::getStorageClassSpecifierString(V->getStorageClass()));
            }
            // else fprintf(stderr, "no storage class? "); 

            QualType T = V->getType();
            string TypeStr = T.getAsString();
            std::string Name = VD->getNameAsString();
            //VD->getType().getAsStringInternal(Name,
            //                                  PrintingPolicy(VD->getASTContext().getLangOpts()));
            fprintf(stderr,"%s %s ", TypeStr.c_str(), Name.c_str());
            // If this is a vardecl with an initializer, emit it.
            if (Expr *E = V->getInit()) {
              fprintf(stderr, " = ");

              Stmt *s = dyn_cast<Stmt>(E);
              PrintStmt(s, SRCMAN, level+1);
              //fprintf(stderr, ";\n"); 

            }

          }

        }
        if (level <= 1) fprintf(stderr, ";\n");           // TODO wrong

      } // for each actual declaration 
}






void PrintAFloatingLiteral( Stmt *s ) {
  FloatingLiteral *F = dyn_cast<FloatingLiteral>(s);
  fprintf(stderr, "%f", F->getValueAsApproximateDouble()); // TODO approximate? 
}




void PrintALoop( Stmt *L, SourceManager *SRCMAN, int level ) {
  //fprintf(stderr, "\nA LOOP L=0x%x  SRCMAN 0x%x", L, &SRCMAN); 
  ForStmt *ForStatement = cast<ForStmt>(L);

  SourceLocation srcloc = ForStatement->getForLoc();
  unsigned int lineno   = SRCMAN->getPresumedLineNumber( srcloc );
  const char *filename  = SRCMAN->getBufferName( srcloc );
  //fprintf(stderr, " in file %s  at line %d   ", filename, lineno); 
  //printsourceline( filename, lineno); 

  Stmt *init = ForStatement->getInit();
  Expr *cond = ForStatement->getCond();
  Expr *incr = ForStatement->getInc();
  Stmt *body = ForStatement->getBody();

  fprintf(stderr, "for (");
  PrintStmt(init, SRCMAN, 0);
  fprintf(stderr, "; ");
  PrintStmt(cond, SRCMAN, 0);
  fprintf(stderr, "; ");
  PrintStmt(incr, SRCMAN, 0);
  fprintf(stderr, " )\n");
  Indent(level); 
  fprintf(stderr, "{\n");
  PrintStmt(body, SRCMAN, level+1);
  fprintf(stderr, "}\n\n");

}


void PrintAUnaryOperator( Stmt *s, SourceManager *SRCMAN, int level ) {
  //fprintf(stderr, "UnaryOperator  "); 
  UnaryOperator *u = cast<UnaryOperator>(s);

  const char *op = unops[u->getOpcode()].c_str();

  if (u->isPrefix()) {
    fprintf(stderr, "%s", op );
  }

  PrintStmt( u->getSubExpr(), SRCMAN, level+1 );

  if (u->isPostfix()) {
    fprintf(stderr, "%s", op );
  }
}



void PrintAnIfStmt( Stmt *s, SourceManager *SRCMAN, int level ) {
  //fprintf(stderr, "an IF statement\n"); 
  //    SourceLocation S = s->getLocStart();
  //    SourceLocation E = s->getLocEnd();
  //    printlines( S, E, SRCMAN);

  IfStmt *IfStatement = cast<IfStmt>(s);

  Stmt *Cond = IfStatement->getCond();
  Stmt *Then = IfStatement->getThen();
  Stmt *Else = IfStatement->getElse();

  fprintf(stderr, "if (");
  PrintStmt(Cond, SRCMAN, level+1);
  fprintf(stderr, ") {\n");
  PrintStmt(Then, SRCMAN, level+1);
  fprintf(stderr, "\n}\nelse\n{\n");
  PrintStmt(Else, SRCMAN, level+1);
  fprintf(stderr, "\n}\n\n");
}



void PrintAnIntegerLiteral(   Stmt *s ) {
  IntegerLiteral *I = dyn_cast<IntegerLiteral>(s);
  bool isSigned = I->getType()->isSignedIntegerType();
  fprintf(stderr, "%s", I->getValue().toString(10, isSigned).c_str());
}



void PrintArraySubscriptExpr( Stmt *s,  SourceManager *SRCMAN, int level ) {
  ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(s);

  Expr *Base = ASE->getBase();
  Expr *Index = ASE->getIdx();

  PrintStmt(Base, SRCMAN, level+1);
  fprintf(stderr, "[");
  PrintStmt(Index, SRCMAN, level+1);
  fprintf(stderr, "]");
}


void PrintCompoundStmt(  Stmt *s, SourceManager *SRCMAN, int level ) {
  //fprintf(stderr, "\nCompoundStmt(%d)", level);
  CompoundStmt *cs = dyn_cast<CompoundStmt>(s);
    int numchildren=cs->size();
    //fprintf(stderr, "CompoundStmt has %d children\n", numchildren);


#ifdef DEBUGGING    
BUH 
    for (Stmt::child_range I = cs->children(); I; ++I) {
      const char *classname =  I->getStmtClassName();
      if (!strcmp(classname, "BinaryOperator")) {
        BinaryOperator *b = cast<BinaryOperator>(*I);
        BinaryOperator::Opcode op = b->getOpcode();
        if (op == BO_Assign)  {
          fprintf(stderr, "compound statement has child of type ASSIGNMENT STATEMENT  ");
          SourceLocation S = I->getLocStart();
          SourceLocation E = I->getLocEnd();
          unsigned int startlineno = SRCMAN->getPresumedLineNumber( S );
          unsigned int   endlineno = SRCMAN->getPresumedLineNumber( E );
          fprintf(stderr, "(%d-%d)\n", startlineno, endlineno ); 
        }
        else
          fprintf(stderr, "compound statement has child of type %s\n", I->getStmtClassName());
      }
      else
        fprintf(stderr, "compound statement has child of type %s\n", I->getStmtClassName());
    }
#endif   // debugging 



    for (Stmt::child_range I = cs->children(); I; ++I) {
      Stmt *child = *I;
      PrintStmt( child, SRCMAN, level   );   // NOTE not level + 1
      
      fprintf(stderr, "\n"); // ***\n\n"); 
    }
    
}




void PrintDeclRefExpr( Stmt *s,  SourceManager *SRCMAN, int level ) {
  DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(s);

  //if (NestedNameSpecifier *Qualifier = DRE->getQualifier())
  //  Qualifier->print( raw_ostream nonstandard of course ); 
  DeclarationNameInfo DNI = DRE->getNameInfo();
  DeclarationName DN = DNI.getName();
  fprintf(stderr, "%s", DN.getAsString().c_str()); 

}



void PrintFunctionDecl( FunctionDecl *D, SourceManager *SRCMAN, int level ) {
  //rintf(stderr, "\nFunctionDecl(%d)  %s\n", level,  D->getNameInfo().getAsString().c_str()); 
  // Type name as string
  QualType QT = D->getResultType();
  string TypeStr = QT.getAsString();

  // Function name
  DeclarationName DeclName = D->getNameInfo().getName();
  string FuncName = DeclName.getAsString();
  //fprintf(stderr, "function %s has type %s ", FuncName.c_str(),  TypeStr.c_str()); 


  fprintf(stderr, "\n%s %s(",TypeStr.c_str(), FuncName.c_str());

  int numparams = D->getNumParams();
  //fprintf(stderr, "and %d parameters\n", numparams);
  for (int i=0; i<numparams; i++) {
    if (i) fprintf(stderr, ", ");
    ParmVarDecl *clangVardecl = D->getParamDecl(i);

    // from  DeclPrinter::VisitVarDecl(VarDecl *D)
    StorageClass SCAsWritten = clangVardecl->getStorageClassAsWritten();
    if (SCAsWritten != SC_None) {
      fprintf(stderr, "%s ", VarDecl::getStorageClassSpecifierString(SCAsWritten));
    } 
    //else fprintf(stderr, "(no storage class?) ");

    QualType T = clangVardecl->getType();
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(clangVardecl))
      T = Parm->getOriginalType();

    string Name    = clangVardecl->getName();
    char *td = strdup(T.getAsString().c_str());
    fprintf(stderr, "td = '%s'\n", td); 
    char *arraypart = splitTypeInfo(td);
    fprintf(stderr, "%s %s%s ", td, Name.c_str(), arraypart);

  }

  fprintf(stderr, ")\n{\n"); // beginning of function body 

  Stmt *body = D->getBody();
  PrintStmt( body, SRCMAN, level+1);
  fprintf(stderr, "}\n\n");   // end of function body
}


void PrintImplicitCastExpr( Stmt *s,  SourceManager *SRCMAN, int level ) {
  ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(s);
  PrintStmt( ICE->getSubExpr(), SRCMAN, level+1);
}



void PrintReturnStmt( Stmt *s,  SourceManager *SRCMAN, int level ) {
  ReturnStmt *r = dyn_cast<ReturnStmt>(s);

  fprintf(stderr, "return");
  if (r->getRetValue()) {
    fprintf(stderr, " ");
    PrintStmt( r->getRetValue(), SRCMAN, level+1);
  }
  fprintf(stderr, ";\n");
}


void PrintStmt( Stmt *s,  SourceManager *SRCMAN, int level ) {
  //fprintf(stderr, "\nprint statement 0x%x of type %s\n", s, s->getStmtClassName()); 


  //fprintf(stderr, "*"); 
  //SourceLocation srcloc = s->getStartLoc();
  //unsigned int lineno = SRCMAN->getPresumedLineNumber( srcloc ); 
  //const char *filename = SRCMAN->getBufferName( srcloc );

  //StmtClass getStmtClass() 
  if (isa<DeclStmt>(s))                  {          PrintDeclStmt(s, SRCMAN, level);
  } else if (isa<FloatingLiteral>(s))    {  PrintAFloatingLiteral(s               );
  } else if (isa<IntegerLiteral>(s))     {  PrintAnIntegerLiteral(s               );
  } else if (isa<UnaryOperator>(s))      {    PrintAUnaryOperator(s, SRCMAN, level);
  } else if (isa<BinaryOperator>(s))     {   PrintBinaryOperator(s, SRCMAN, level);
  } else if (isa<ForStmt>(s))            {             PrintALoop(s, SRCMAN, level);
  } else if (isa<IfStmt>(s))             {          PrintAnIfStmt(s, SRCMAN, level);
  } else if (isa<CompoundStmt>(s))       {      PrintCompoundStmt(s, SRCMAN, level);
  } else if (isa<ImplicitCastExpr>(s))   {  PrintImplicitCastExpr(s, SRCMAN, level);
  } else if (isa<DeclRefExpr>(s))        {       PrintDeclRefExpr(s, SRCMAN, level); 
  } else if (isa<ArraySubscriptExpr>(s)) {PrintArraySubscriptExpr(s, SRCMAN, level);
  } else if (isa<ReturnStmt>(s))         {        PrintReturnStmt(s, SRCMAN, level);
  } else  { 
    fprintf(stderr, "\nPrintStmt() UNHANDLED statement of type %s\n", s->getStmtClassName());
    exit(-1); 
  }
  //int numchildren=0;
  //for (Stmt::child_range I = s->children(); I; ++I, numchildren++) ;
  //if (numchildren) fprintf(stderr, "has %d children\n", numchildren);
  //if (numchildren) {
  //  for (Stmt::child_range I = s->children(); I; ++I
  //}
}


void PrintTranslationUnit(  TranslationUnitDecl *TUD,  ASTContext &CTX ) {
  // TUD derived from Decl and DeclContext
  static DeclContext *DC = TUD->castToDeclContext( TUD );
  //SourceManager  SM =  CTX.getSourceManager();

  for (DeclContext::decl_iterator DI = DC->decls_begin(), DE = DC->decls_end(); DI != DE; ++DI) {
    Decl *D = *DI;

    if (isa<FunctionDecl>(D))  { //fprintf(stderr, "FunctionDecl\n");
      PrintFunctionDecl( dyn_cast<FunctionDecl>(D), &CTX.getSourceManager(), 0);
    }
    else if (isa<VarDecl>(D)) { //fprintf(stderr, "VarDecl\n"); 
      PrintVarDecl( dyn_cast<VarDecl>(D), &CTX.getSourceManager(), 0 );
    }
    else if (isa<TypedefDecl>(D)) { //fprintf(stderr, "TypedefDecl\n"); 
      PrintTypeDefDecl( dyn_cast<TypedefDecl>(D), &CTX.getSourceManager(), 0 );
    }
    else if (isa<TypeAliasDecl>(D)) { fprintf(stderr, "TypeAliasDecl\n");
    }
    else { 
      fprintf(stderr, "\na declaration of type %s (%d) which I have no idea how to handle\n",   D->getDeclKindName(), D->getKind());
      exit(-1);
    }
      
    //else if (isa<TypedefNameDecl>(D)) { fprintf(stderr, "TypedefNameDecl\n");}
  }
}


void PrintTypeDefDecl( TypedefDecl *D, SourceManager *SRCMAN, int level ) {

  /* internal typedefs do not have a source file and this will die!
  SourceLocation S = D->getLocStart();  // NOT getStartLoc(), that's class DeclStmt
  SourceLocation E = D->getLocEnd();
  unsigned int startlineno = SRCMAN->getPresumedLineNumber( S ); 
  unsigned int   endlineno = SRCMAN->getPresumedLineNumber( E ); 
  const char *filename = SRCMAN-> etBufferName( S );
  fprintf(stderr, " in file %s  at lines %d-%d", filename, startlineno, endlineno); 
  for (int l=startlineno; l<= endlineno; l++) printsourceline( filename, l);
  */

  // arrays suck
  char *td =  strdup( D->getUnderlyingType().getAsString().c_str());
  //fprintf(stderr, "td = '%s'\n", td); 
  char *arraypart = splitTypeInfo(td);
  fprintf(stderr, "typedef %s %s%s;\n", td, D->getName().str().c_str(), arraypart);

  free(td);
  free(arraypart);
}



void PrintVarDecl( VarDecl *D, SourceManager *SRCMAN, int level ) {
  // arrays suck
  char *td =  strdup( D->getType().getAsString().c_str());  // leak 
  //fprintf(stderr, "td = '%s'\n", td); 
  char *arraypart = splitTypeInfo(td);
  fprintf(stderr, "%s %s%s", td,  D->getName().str().c_str(), arraypart);

  Expr *Init = D->getInit();
  if (Init) {
    //fprintf(stderr," = (TODO)");
    PrintStmt( Init, SRCMAN, level+1);
  }
  fprintf(stderr, ";\n");

  free(td);
  free(arraypart);
} //PrintVarDecl








chillAST_node * ConvertVarDecl( VarDecl *D, chillAST_node *p ) {
  //fprintf(stderr, "\nConvertVarDecl()\n"); 
  //fprintf(stderr, "Decl has type %s\n", D->getDeclKindName()); 
  //PrintVarDecl( D, globalSRCMAN, 0 );

   bool isParm = false; 

   QualType T0 = D->getType();
   QualType T  = T0;
   if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) { // My GOD clang stinks
     T = Parm->getOriginalType();
     isParm = true;
   }

  // arrays suck
  char *vartype =  strdup( T.getAsString().c_str());  
  //fprintf(stderr, "vartype = '%s'   T0  '%s\n", vartype, T0.getAsString().c_str() ); 
  char *arraypart = splitTypeInfo(vartype);
  //fprintf(stderr, "arraypart = '%s'\n", arraypart);


  char *varname = strdup(D->getName().str().c_str()); 
  //fprintf(stderr, "VarDecl (clang 0x%x) for %s %s%s\n", D, vartype,  varname, arraypart);
  
  chillAST_VarDecl * chillVD = new chillAST_VarDecl( vartype,  varname, arraypart, (void *)D, p /* , initializer */ );

  chillVD->isAParameter = isParm; 
  //fprintf(stderr, "\nthis is the vardecl\n"); 
  //chillVD->print();  printf("\n\n"); fflush(stdout); 

  //clang::QualType qtyp = D->getType(); 
  //clang::Expr *e = D->getInit();
  //fprintf(stderr, "e 0x%x\n", e); 

  //if (qtyp->isPointerType()) { 
  //  fprintf(stderr, "pointer type\n"); 
  //  clang::QualType ptyp = qtyp->getPointeeType(); 
  //  fprintf(stderr, "%s\n", ptyp.getAsString().c_str());  
  //} 

  //if (qtyp->isArrayType()) fprintf(stderr, "Array type\n"); 
  //if (qtyp->isConstantArrayType()) fprintf(stderr, "constant array type\n"); 

  //const clang::Type *typ = qtyp.getTypePtr(); 
  //clang::Expr *e = ((clang::VariableArrayType *)typ)->getSizeExpr();
  //e->dump(); 


  int numdim = 0;
  chillVD-> knownArraySizes = true;
  if (index(vartype, '*')) chillVD->knownArraySizes = false;  // float *a;   for example
  if (index(arraypart, '*'))  chillVD->knownArraySizes = false;
  
  // note: vartype here, arraypart in next code..    is that right?
  if (index(vartype, '*')) { 
    for (int i = 0; i<strlen(vartype); i++) if (vartype[i] == '*') numdim++;
    //fprintf(stderr, "numd %d\n", numd);
    chillVD->numdimensions = numdim; 
  }

  if (index(arraypart, '[')) {  // JUST [12][34][56]  no asterisks
    char *dupe = strdup(arraypart);

    int len = strlen(arraypart);
    for (int i=0; i<len; i++) if (dupe[i] == '[') numdim++;

    //fprintf(stderr, "numdim %d\n", numdim);

    chillVD->numdimensions = numdim; 
    int *as =  (int *)malloc(sizeof(int *) * numdim );
    if (!as) { 
      fprintf(stderr, "can't malloc array sizes in ConvertVarDecl()\n");
      exit(-1);
    }
    chillVD->arraysizes = as; // 'as' changed later!

    
    char *ptr = dupe;
    //fprintf(stderr, "dupe '%s'\n", ptr);
    while (ptr = index(ptr, '[')) { 
      ptr++;
      //fprintf(stderr, "tmp '%s'\n", ptr);
      int dim;
      sscanf(ptr, "%d", &dim);
      //fprintf(stderr, "dim %d\n", dim);
      *as++ = dim; 
      
      ptr =  index(ptr, ']');
      //fprintf(stderr, "bottom of loop, ptr = '%s'\n", ptr); 
    }
    free(dupe);
    //for (int i=0; i<numdim; i++) { 
    //  fprintf(stderr, "dimension %d = %d\n", i,  chillVD->arraysizes[i]); 
    //} 
    
    //fprintf(stderr, "need to handle [] array to determine num dimensions\n");
    //exit(-1); 
  }
  
  Expr *Init = D->getInit();
  if (Init) {
    fprintf(stderr," = VARDECL HAS INIT.  (TODO) (RIGHT NOW)"); exit(-1); 
  }
  //fprintf(stderr, ";\n");
  

  //fprintf(stderr, "calling chillVD->print()\n"); 
  //chillVD->print();  // debugging only

  free (vartype);
  free (varname);

  // store this away for declrefexpr that references it! 
  VariableDeclarations.push_back(chillVD);
  return chillVD;
}



chillAST_node * ConvertRecordDecl( clang::RecordDecl *RD, chillAST_node *p ) { // for structs and unions

  //fprintf(stderr, "ConvertRecordDecl(  )\n\nclang sees\n");
  //RD->dump();
  //fflush(stdout); 
  //fprintf(stderr, "\n"); 
  
  //fprintf(stderr, "%s with name %s\n", ((clang::Decl *)RD)->getDeclKindName(), RD->getNameAsString().c_str());  
  //const clang::ASTRecordLayout RL = RD->getASTContext().getASTRecordLayout( RD );
  //RD->getASTContext().DumpRecordLayout( RD , cout ); 

  int count = 0;
  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) count++; 
  //fprintf(stderr, "%d fields in this struct/union\n", count); 

  char blurb[128];
  sprintf(blurb, "struct %s", RD->getNameAsString().c_str()); 
  fprintf(stderr, "blurb is '%s'\n", blurb); 

  chillAST_TypedefDecl *astruct = new chillAST_TypedefDecl( blurb, "", p);
  astruct->setStruct( true ); 
  astruct->setStructName( RD->getNameAsString().c_str() );

  for (clang::RecordDecl::field_iterator fi = RD->field_begin(); fi != RD->field_end(); fi++) { 
    clang::FieldDecl *FD = (*fi);
    FD->dump(); printf(";\n"); fflush(stdout); 
    string TypeStr = FD->getType().getAsString(); 

    const char *typ  = TypeStr.c_str();
    const char *name = FD->getNameAsString().c_str();
    fprintf(stderr, "(typ) %s (name) %s\n", typ, name);

    chillAST_VarDecl *VD = NULL;
    // very clunky and incomplete
    VD = new chillAST_VarDecl( typ, name, "", astruct ); // can't handle arrays yet 
    
    astruct->subparts.push_back(VD); 
  }


  fprintf(stderr, "I just defined a struct\n"); 
  astruct->print(0, stderr); 

  return astruct; 
}







chillAST_node * ConvertTypeDefDecl( TypedefDecl *TDD, chillAST_node *p ) {
  //fprintf(stderr, "ConvertTypedefDecl(  ) \n");
  //fprintf(stderr, "TDD has type %s\n", TDD->getDeclKindName()); 
  //TDD->dump(); fprintf(stderr, "\n"); 

  char *under =  strdup( TDD->getUnderlyingType().getAsString().c_str());
  //fprintf(stderr, "under = '%s'\n", under); 
  char *arraypart = splitTypeInfo(under);
  //fprintf(stderr, "typedef %s %s%s;\n", under, TDD->getName().str().c_str(), arraypart);
  //  fprintf(stderr, "len arraypart = %d\n", strlen(arraypart)); 
  char *alias = strdup(TDD->getName().str().c_str());

  //fprintf(stderr, "underlying type %s  arraypart '%s'  name %s\n", under, arraypart, TDD->getName().str().c_str() ); 
  chillAST_TypedefDecl *CTDD = new chillAST_TypedefDecl( under, alias, arraypart, p ); 

  free(under);        
  free(arraypart);   

  return CTDD; 
}



chillAST_node * ConvertDeclStmt( DeclStmt *clangDS, chillAST_node *p ) {
  //fprintf(stderr, "ConvertDeclStmt()\n"); 

  chillAST_VarDecl *chillvardecl; // the thing we'll return if this is a single declaration
  
  bool multiples = !clangDS->isSingleDecl();
  if ( multiples) { 
    //fprintf(stderr, "ir_clang.cc  multiple declarations in a single CLANG DeclStmt  not really handled! (??)\n"); 
    // for now, try to make the multiple decls into a compoundstmt with them inside.
    // if we don't get scoping problems, this might work
  }
  
  DeclGroupRef dgr = clangDS->getDeclGroup();
  clang::DeclGroupRef::iterator DI = dgr.begin();
  clang::DeclGroupRef::iterator DE = dgr.end();
  
  for ( ; DI != DE; ++DI) {
    Decl *D = *DI;
    const char *decltype =  D->getDeclKindName();
    //fprintf(stderr, "a decl of type %s\n", D->getDeclKindName()); 
    
    if (!strcmp("Var", decltype)) { 
      VarDecl *V = dyn_cast<VarDecl>(D);
      // ValueDecl *VD = dyn_cast<ValueDecl>(D); // not needed? 
      std::string Name = V->getNameAsString();
      char *varname = strdup( Name.c_str()); 
      
      //fprintf(stderr, "variable named %s\n", Name.c_str()); 
      QualType T = V->getType();
      string TypeStr = T.getAsString();
      char *vartype =  strdup( TypeStr.c_str());
      
      //fprintf(stderr, "%s %s\n", td, varname); 
      char *arraypart = splitTypeInfo( vartype );
      
      chillvardecl = new chillAST_VarDecl(vartype, varname, arraypart, (void *)D, p );
      //fprintf(stderr, "DeclStmt (clang 0x%x) for %s %s%s\n", D, vartype,  varname, arraypart);

      // store this away for declrefexpr that references it! 
      VariableDeclarations.push_back(chillvardecl);
      
      if (multiples) p->addChild( chillvardecl ); 

      // TODO 
      if (V->hasInit()) { 
        fprintf(stderr, " ConvertDeclStmt()  UNHANDLED initialization\n");
        exit(-1); 
      }
    }
  }  // for each of possibly multiple decls 
  
  if (multiples) return NULL;    // multiple decls added themselves already
  return chillvardecl;  // OR a single decl
}



chillAST_node * ConvertCompoundStmt( CompoundStmt *clangCS, chillAST_node *p ) {
  //fprintf(stderr, "ConvertCompoundStmt(  )\n");
  int numchildren= clangCS->size();
  //fprintf(stderr, "clang CompoundStmt has %d children\n", numchildren);

  // make an empty CHILL compound statement 
  chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt; 
  chillCS->setParent( p );

  // for each clang child
  for (Stmt::child_range I = clangCS->children(); I; ++I)  { // ?? loop looks WRONG
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
  //fprintf(stderr, "\nConvertFunctionDecl(  )\n");
  QualType QT = D->getResultType();
  string ReturnTypeStr = QT.getAsString();

  // Function name
  DeclarationName DeclName = D->getNameInfo().getName();
  string FuncName = DeclName.getAsString();
  //fprintf(stderr, "function %s has type %s ", FuncName.c_str(),  ReturnTypeStr.c_str()); 
  //fprintf(stderr, "\n%s %s()\n", ReturnTypeStr.c_str(), FuncName.c_str());

  chillAST_FunctionDecl *chillFD = new chillAST_FunctionDecl( ReturnTypeStr.c_str(),  FuncName.c_str(), p, D);
  

  int numparams = D->getNumParams();

  //fprintf(stderr, "\nand %d parameters\n", numparams);
  for (int i=0; i<numparams; i++) {
    if (i) fprintf(stderr, ", ");
    VarDecl *clangvardecl = D->getParamDecl(i);  // the ith parameter  (CLANG)
    ParmVarDecl *pvd = D->getParamDecl(i); 
    QualType T = pvd->getOriginalType();
    fprintf(stderr, "OTYPE %s\n", T.getAsString().c_str()); 

    chillAST_VarDecl *chillPVD = (chillAST_VarDecl *)ConvertVarDecl( clangvardecl, chillFD ) ; 
    //chillPVD->print();  fflush(stdout); 

    //chillPVD->isAParameter = 1;
    VariableDeclarations.push_back(chillPVD); 
    
    chillFD->addParameter(chillPVD); 
    fprintf(stderr, "chillAST ParmVarDecl for %s from chill location 0x%x\n",chillPVD->varname, clangvardecl); 
  } // for each parameter 



  //fprintf(stderr, ")\n{\n"); // beginning of function body 
  //if (D->isExternC())    { chillFD->setExtern();  fprintf(stderr, "%s is extern\n", FuncName.c_str()); }; 
  if (D->getBuiltinID()) { chillFD->setExtern();  fprintf(stderr, "%s is builtin (extern)\n", FuncName.c_str()); }; 

  Stmt *clangbody = D->getBody();
  if (clangbody) { // may just be fwd decl or external, without an actual body 
    //fprintf(stderr, "body of type %s\n", clangbody->getStmtClassName()); 
    //chillAST_node *CB = ConvertCompoundStmt(  dyn_cast<CompoundStmt>(clangbody) ); // always a compound statement?
    chillAST_node *CB = ConvertGenericClangAST( clangbody, chillFD ); 
    //fprintf(stderr, "FunctionDecl body = 0x%x of type %s\n", CB, CB->getTypeString());
    chillFD->setBody ( CB ); 
  }

  //fprintf(stderr, "adding function %s  0x%x to FunctionDeclarations\n", chillFD->functionName, chillFD); 
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
  if (bod->asttype != CHILLAST_NODETYPE_COMPOUNDSTMT) { 
    //fprintf(stderr, "ForStmt body of type %s\n", bod->getTypeString()); 
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
  //fprintf(stderr, "clang base: "); clangbase->dump(); fprintf(stderr, "\n"); 

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
  //fprintf(stderr, "\n\n*** type %s ***\n\n", TypeStr.c_str()); 
  //fprintf(stderr, "kind %s\n", vd->getDeclKindName()); 

  DeclarationName DN = DNI.getName();
  const char *varname = DN.getAsString().c_str() ; 
  chillAST_DeclRefExpr * chillDRE = new chillAST_DeclRefExpr(TypeStr.c_str(),  varname, p ); 

  //fprintf(stderr, "clang DeclRefExpr refers to declaration of %s @ 0x%x\n", varname, vd);
  //fprintf(stderr, "clang DeclRefExpr refers to declaration of %s of kind %s\n", varname, vd->getDeclKindName()); 
  
  // find the definition (we hope)
  if ( (!strcmp("Var",  vd->getDeclKindName())) || (!strcmp("ParmVar",  vd->getDeclKindName()))) { 
    // it's a variable reference 
    int numvars = VariableDeclarations.size();
    chillAST_VarDecl *chillvd = NULL;
    for (int i=0; i<numvars; i++) { 
      if (VariableDeclarations[i]->uniquePtr == vd) {
        chillvd = VariableDeclarations[i];
        //fprintf(stderr, "found it at variabledeclaration %d of %d\n", i, numvars);
      }
    }
    if (!chillvd) { 
      fprintf(stderr, "\nWARNING, ir_clang.cc clang DeclRefExpr %s refers to a declaration I can't find! at ox%x\n", varname, vd); 
      fprintf(stderr, "variables I know of are:\n");
      for (int i=0; i<numvars; i++) { 
        chillAST_VarDecl *adecl = VariableDeclarations[i];
        if (adecl->isParmVarDecl()) fprintf(stderr, "(parameter) ");
        fprintf(stderr, "%s %s at location 0x%x\n", adecl->vartype, adecl->varname, adecl->uniquePtr); 
      }  
      fprintf(stderr, "\n"); 
    }
    
    if (chillvd == NULL) { fprintf(stderr, "chillDRE->decl = 0x%x\n", chillvd); exit(-1); }

    chillDRE->decl = (chillAST_node *)chillvd; // start of spaghetti pointers ...
  }
  else  if (!strcmp("Function",  vd->getDeclKindName())) { 
    //fprintf(stderr, "declrefexpr of type Function\n");
    int numfuncs = FunctionDeclarations.size();
    chillAST_FunctionDecl *chillfd = NULL;
    for (int i=0; i<numfuncs; i++) { 
      if (FunctionDeclarations[i]->uniquePtr == vd) {
        chillfd = FunctionDeclarations[i];
        //fprintf(stderr, "found it at functiondeclaration %d of %d\n", i, numfuncs);
      }
    }
    if (chillfd == NULL) { fprintf(stderr, "chillDRE->decl = 0x%x\n", chillfd); exit(-1); }

    chillDRE->decl = (chillAST_node *)chillfd; // start of spaghetti pointers ...
    
  }
  else { 
  fprintf(stderr, "clang DeclRefExpr refers to declaration of %s of kind %s\n", varname, vd->getDeclKindName()); 
    fprintf(stderr, "chillDRE->decl = UNDEFINED\n"); 
    exit(-1); 
  }

  //fprintf(stderr, "%s\n", DN.getAsString().c_str()); 
  return chillDRE; 
}



chillAST_node * ConvertIntegerLiteral( IntegerLiteral *clangIL, chillAST_node *p ) { 
  bool isSigned = clangIL->getType()->isSignedIntegerType();
  //int val = clangIL->getIntValue();
  const char *printable = clangIL->getValue().toString(10, isSigned).c_str(); 
  int val = atoi( printable ); 
  //fprintf(stderr, "int value %s  (%d)\n", printable, val); 
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val, p );
  return chillIL; 
}


chillAST_node * ConvertFloatingLiteral( FloatingLiteral *clangFL, chillAST_node *p ) { 
  //fprintf(stderr, "\nConvertFloatingLiteral()\n"); 
  float val = clangFL->getValueAsApproximateDouble(); // TODO approx is a bad idea!
  string WHAT; 
  SmallString<16> Str;
  clangFL->getValue().toString( Str );
  const char *printable = Str.c_str(); 
  //fprintf(stderr, "literal %s\n", printable); 

  SourceLocation sloc = clangFL->getLocStart();
  SourceLocation eloc = clangFL->getLocEnd();

  std::string start = sloc.printToString( *globalSRCMAN ); 
  std::string end   = eloc.printToString( *globalSRCMAN ); 
  //fprintf(stderr, "literal try2 start %s end %s\n", start.c_str(), end.c_str()); 
  //printlines( sloc, eloc, globalSRCMAN ); 
  unsigned int startlineno = globalSRCMAN->getPresumedLineNumber( sloc );
  unsigned int   endlineno = globalSRCMAN->getPresumedLineNumber( eloc ); ;
  const char     *filename = globalSRCMAN->getBufferName( sloc );

  std::string  fname = globalSRCMAN->getFilename( sloc );
  //fprintf(stderr, "fname %s\n", fname.c_str()); 

  if (filename && strlen(filename) > 0) {} // fprintf(stderr, "literal file '%s'\n", filename);
  else { 
    fprintf(stderr, "\nConvertFloatingLiteral() filename is NULL?\n"); 

    //sloc =  globalSRCMAN->getFileLoc( sloc );  // should get spelling loc? 
    sloc =  globalSRCMAN->getSpellingLoc( sloc );  // should get spelling loc? 
    //eloc =  globalSRCMAN->getFileLoc( eloc );

    start = sloc.printToString( *globalSRCMAN ); 
    //end   = eloc.printToString( *globalSRCMAN );  
    //fprintf(stderr, "literal try3 start %s end %s\n", start.c_str(), end.c_str()); 
   
    startlineno = globalSRCMAN->getPresumedLineNumber( sloc );
    //endlineno = globalSRCMAN->getPresumedLineNumber( eloc ); ;    
    //fprintf(stderr, "start, end line numbers %d %d\n", startlineno, endlineno); 
    
    filename = globalSRCMAN->getBufferName( sloc );

    //if (globalSRCMAN->isMacroBodyExpansion( sloc )) { 
    //  fprintf(stderr, "IS MACRO\n");
    //} 
  }
  
  unsigned int  offset = globalSRCMAN->getFileOffset( sloc );
  //fprintf(stderr, "literal file offset %d\n", offset); 

  FILE *fp = fopen (filename, "r");
  fseek(fp, offset, SEEK_SET); // go to the part of the file where the float is defined
  
  char buf[10240];
  fgets (buf, sizeof(buf), fp); // read a line starting where the float starts
  fclose(fp);

  // buf has the line we want   grab the float constant out of it
  //fprintf(stderr, "\nbuf '%s'\n", buf);
  char *ptr = buf;
  if (*ptr == '-') ptr++; // ignore possible minus sign
  int len = strspn(ptr, ".-0123456789f"); 
  buf[len] = '\0';
  //fprintf(stderr, "'%s'\n", buf);

  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val, buf, p );
  
  //chillFL->print(); printf("\n"); fflush(stdout); 
  return chillFL; 
}


chillAST_node * ConvertImplicitCastExpr( ImplicitCastExpr *clangICE, chillAST_node *p ) {
  //fprintf(stderr, "ConvertImplicitCastExpr()\n"); 
  CastExpr *CE = dyn_cast<ImplicitCastExpr>(clangICE);
  //fprintf(stderr, "implicit cast of type %s\n", CE->getCastKindName());
  chillAST_node * sub = ConvertGenericClangAST( clangICE->getSubExpr(), p );
  chillAST_ImplicitCastExpr *chillICE = new chillAST_ImplicitCastExpr( sub, p ); 
  
  //sub->setParent( chillICE ); // these 2 lines work
  //return chillICE; 

  //sub->setParent(p);         // ignore the ImplicitCastExpr !!  TODO (probably a bad idea) 
  return sub; 

}




chillAST_node * ConvertCStyleCastExpr( CStyleCastExpr *clangCSCE, chillAST_node *p ) {
  //fprintf(stderr, "ConvertCStyleCastExpr()\n"); 
  //fprintf(stderr, "C Style cast of kind ");
  CastExpr *CE = dyn_cast<CastExpr>(clangCSCE);
  //fprintf(stderr, "%s\n", CE->getCastKindName());
  
  //clangCSCE->getTypeAsWritten().getAsString(Policy)
  const char * towhat = strdup( clangCSCE->getTypeAsWritten().getAsString().c_str() );
  //fprintf(stderr, "before sub towhat (%s)\n", towhat);

  chillAST_node * sub = ConvertGenericClangAST( clangCSCE->getSubExprAsWritten(), NULL );
  //fprintf(stderr, "after sub towhat (%s)\n", towhat);
  chillAST_CStyleCastExpr *chillCSCE = new chillAST_CStyleCastExpr( towhat, sub, p ); 
  //fprintf(stderr, "after CSCE towhat (%s)\n", towhat);
  sub->setParent( chillCSCE );
  return chillCSCE; 
}




chillAST_node * ConvertReturnStmt( ReturnStmt *clangRS, chillAST_node *p ) {
  chillAST_node * retval = ConvertGenericClangAST( clangRS->getRetValue(), NULL ); // NULL is handled
  //if (retval == NULL) fprintf(stderr, "return stmt returns nothing\n");

  chillAST_ReturnStmt * chillRS = new chillAST_ReturnStmt( retval, p );
  if (retval) retval->setParent( chillRS );
  return chillRS; 
}


chillAST_node * ConvertCallExpr( CallExpr *clangCE, chillAST_node *p ) {
  //fprintf(stderr, "ConvertCallExpr()\n"); 

  chillAST_node *callee = ConvertGenericClangAST( clangCE->getCallee(), NULL ); 
  //fprintf(stderr, "callee is of type %s\n", callee->getTypeString()); 

  //chillAST_node *next = ((chillAST_ImplicitCastExpr *)callee)->subexpr;
  //fprintf(stderr, "callee is of type %s\n", next->getTypeString()); 

  chillAST_CallExpr *chillCE = new chillAST_CallExpr( callee, p ); 
  callee->setParent( chillCE );

  int numargs = clangCE->getNumArgs();
  //fprintf(stderr, "CallExpr has %d args\n", numargs);
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
  //fprintf(stderr, "ConvertTranslationUnit( filename %s )\n\n", filename); 
  // TUD derived from Decl and DeclContext
  static DeclContext *DC = TUD->castToDeclContext( TUD );


  // TODO this was to get the filename without having to pass it in
  //ASTContext CTX = TUD->getASTContext (); 
  //SourceManager  SM =  CTX.getSourceManager();
  //SourceLocation srcloc = ForStatement->getForLoc();
  //unsigned int lineno   = SRCMAN->getPresumedLineNumber( srcloc );
  //const char *filename  = SRCMAN->getBufferName( srcloc ); 


  chillAST_SourceFile * topnode = new chillAST_SourceFile( filename  ); 
  topnode->setFrontend("clang"); 
  topnode->chill_array_counter  = 1;
  topnode->chill_scalar_counter = 0;

  // now recursively build clang AST from the children of TUD
  //for (DeclContext::decl_iterator DI = DC->decls_begin(), DE = DC->decls_end(); DI != DE; ++DI)
  DeclContext::decl_iterator start = DC->decls_begin();
  DeclContext::decl_iterator end   = DC->decls_end();
  for (DeclContext::decl_iterator DI=start; DI != end; ++DI) { 
    Decl *D = *DI;
       
    if (isa<FunctionDecl>(D))  { //fprintf(stderr, "\nTUD FunctionDecl\n");
      topnode->addChild( ConvertFunctionDecl( dyn_cast<FunctionDecl>(D), topnode )); 
    }
    else if (isa<VarDecl>(D)) { //fprintf(stderr, "\nTUD VarDecl\n");  
      topnode->addChild( ConvertVarDecl( dyn_cast<VarDecl>(D), topnode ));
      //fflush(stdout);  fprintf(stderr, "\nTUD VarDecl DONE\n");  
    }
    else if (isa<TypedefDecl>(D)) { //fprintf(stderr, "\nTUD TypedefDecl\n");  
      topnode->addChild( ConvertTypeDefDecl( dyn_cast<TypedefDecl>(D), topnode )); 
    }
    else if (isa<RecordDecl>(D)) { fprintf(stderr, "\nTUD RecordDecl\n");  
      topnode->addChild( ConvertRecordDecl( dyn_cast<RecordDecl>(D), topnode )); 
    }
    else if (isa<TypeAliasDecl>(D)) { fprintf(stderr, "TUD TypeAliasDecl  TODO \n");   exit(-1); 
    }
    else { 
      fprintf(stderr, "\nTUD a declaration of type %s (%d) which I can't handle\n",   D->getDeclKindName(), D->getKind());
      exit(-1);
    }
  }
  //fflush(stdout);  fprintf(stderr, "leaving ConvertTranslationUnit()\n\n");

  //fprintf(stderr, "in ConvertTranslationUnit(), dumping the file\n");  
  //topnode->dump();
  return ( chillAST_node *)  topnode;
}



 chillAST_node * ConvertGenericClangAST( Stmt *s, chillAST_node *p ) {
   
   if (s == NULL) return NULL;
   //fprintf(stderr, "\nConvertGenericClangAST() Stmt of type %d (%s)\n", s->getStmtClass(),s->getStmtClassName()); 
   Decl *D = (Decl *) s;
   //if (isa<Decl>(D)) fprintf(stderr, "Decl of kind %d (%s)\n",  D->getKind(),D->getDeclKindName() );
   

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
     // more work to do 
     fprintf(stderr, "ir_clang.cc ConvertGenericClangAST() UNHANDLED ");
     //if (isa<Decl>(D)) fprintf(stderr, "Decl of kind %s\n",  D->getDeclKindName() );
     if (isa<Stmt>(s))fprintf(stderr, "Stmt of type %s\n", s->getStmtClassName()); 
     exit(-1); 
   }
   
   return ret; 
 }










// ----------------------------------------------------------------------------
// Class: IR_chillScalarSymbol
// ----------------------------------------------------------------------------
 
std::string IR_chillScalarSymbol::name() const {
  //return vd_->getNameAsString();   CLANG
  //fprintf(stderr, "IR_chillScalarSymbol::name() %s\n", chillvd->varname); 
  return std::string(chillvd->varname);  // CHILL 
}
 

// Return size in bytes
int IR_chillScalarSymbol::size() const {
  //return (vd_->getASTContext().getTypeSize(vd_->getType())) / 8;    // ??
  fprintf(stderr, "IR_chillScalarSymbol::size()  probably WRONG\n"); 
  return (8); // bytes?? 
}


bool IR_chillScalarSymbol::operator==(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_xxxxScalarSymbol::operator==  probably WRONG\n"); 
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillScalarSymbol *l_that = static_cast<const IR_chillScalarSymbol *>(&that);
  return this->chillvd == l_that->chillvd;
}

IR_Symbol *IR_chillScalarSymbol::clone() const {
  return new IR_chillScalarSymbol(ir_, chillvd );  // clone
}

// ----------------------------------------------------------------------------
// Class: IR_chillArraySymbol
// ----------------------------------------------------------------------------

std::string IR_chillArraySymbol::name() const {
  return std::string( strdup( chillvd ->varname)); 
}


int IR_chillArraySymbol::elem_size() const {
  fprintf(stderr, "IR_chillArraySymbol::elem_size()  TODO\n");  exit(-1); 
  return 8;  // TODO 
  //const ArrayType *at = dyn_cast<ArrayType>(vd_->getType()); 
  //if(at) {
  //  return (vd_->getASTContext().getTypeSize(at->getElementType())) / 8;
  //} else
  //  throw ir_error("Symbol is not an array!");
  //return 0;
}


int IR_chillArraySymbol::n_dim() const {
  //fprintf(stderr, "IR_chillArraySymbol::n_dim()\n");
  //fprintf(stderr, "variable %s %s %s\n", chillvd->vartype, chillvd->varname, chillvd->arraypart);
  //fprintf(stderr, "IR_chillArraySymbol::n_dim() %d\n", chillvd->numdimensions); 
  //fprintf(stderr, "IR_chillArraySymbol::n_dim()  TODO \n"); exit(-1); 
  return chillvd->numdimensions; 
}


// TODO
omega::CG_outputRepr *IR_chillArraySymbol::size(int dim) const {
  fprintf(stderr, "IR_chillArraySymbol::n_size()  TODO \n"); exit(-1); 
  return NULL;
}


bool IR_chillArraySymbol::operator!=(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_xxxxArraySymbol::operator!=   NOT EQUAL\n"); 
  //chillAST_VarDecl *chillvd;
  return chillvd != ((IR_chillArraySymbol*)&that)->chillvd ; 
}

bool IR_chillArraySymbol::operator==(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_xxxxArraySymbol::operator==   EQUAL\n"); 
  //chillAST_VarDecl *chillvd;
  return chillvd == ((IR_chillArraySymbol*)&that)->chillvd ; 
  /*
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillArraySymbol *l_that = static_cast<const IR_chillArraySymbol *>(&that);
  return this->vd_ == l_that->vd_ && this->offset_ == l_that->offset_;
  */
}


IR_Symbol *IR_chillArraySymbol::clone() const {
  return new IR_chillArraySymbol(ir_, chillvd, offset_);
}

// ----------------------------------------------------------------------------
// Class: IR_chillConstantRef
// ----------------------------------------------------------------------------

bool IR_chillConstantRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillConstantRef *l_that = static_cast<const IR_chillConstantRef *>(&that);
  
  if (this->type_ != l_that->type_)
    return false;
  
  if (this->type_ == IR_CONSTANT_INT)
    return this->i_ == l_that->i_;
  else
    return this->f_ == l_that->f_;
}


omega::CG_outputRepr *IR_chillConstantRef::convert() {
  //assert(astContext_ != NULL);
  if (type_ == IR_CONSTANT_INT) {

    fprintf(stderr, "IR_chillConstantRef::convert() unimplemented\n");  exit(-1); 
    
    // TODO 
    /*
    BuiltinType *bint = new BuiltinType(BuiltinType::Int);
    IntegerLiteral *ilit = new (astContext_)IntegerLiteral(*astContext_, llvm::APInt(32, i_), bint->desugar(), SourceLocation());
    omega::CG_chillRepr *result = new omega::CG_chillRepr(ilit);
    delete this;
    return result;
    */
  }
  else
    throw ir_error("constant type not supported");
}


IR_Ref *IR_chillConstantRef::clone() const {
  if (type_ == IR_CONSTANT_INT)
    return new IR_chillConstantRef(ir_, i_);
  else if (type_ == IR_CONSTANT_FLOAT)
    return new IR_chillConstantRef(ir_, f_);
  else
    throw ir_error("constant type not supported");
}

// ----------------------------------------------------------------------------
// Class: IR_chillScalarRef
// ----------------------------------------------------------------------------

bool IR_chillScalarRef::is_write() const {
  return op_pos_ ==  OP_DEST; // 2 other alternatives: OP_UNKNOWN, OP_SRC
}


IR_ScalarSymbol *IR_chillScalarRef::symbol() const {
  //VarDecl *vd = static_cast<VarDecl *>(vs_->getDecl());
  //fprintf(stderr, "ir_clang.cc  IR_chillScalarRef::symbol()\n"); //exit(-1);
  chillAST_VarDecl *vd = NULL;
  if (chillvd) vd = chillvd; 
  return new IR_chillScalarSymbol(ir_, vd); // IR_chillScalarRef::symbol()
}


bool IR_chillScalarRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillScalarRef *l_that = static_cast<const IR_chillScalarRef *>(&that);
  
  return this->chillvd == l_that->chillvd;
}


omega::CG_outputRepr *IR_chillScalarRef::convert() {
  //fprintf(stderr, "IR_chillScalarRef::convert() unimplemented\n"); exit(-1); 
  if (!dre) fprintf(stderr, "IR_chillScalarRef::convert()   CLANG SCALAR REF has no dre\n"); 
  omega::CG_chillRepr *result = new omega::CG_chillRepr(dre);
  delete this;
  return result;
}

IR_Ref * IR_chillScalarRef::clone() const {
  if (dre) return new IR_chillScalarRef(ir_, dre); // use declrefexpr if it exists
  return new IR_chillScalarRef(ir_, chillvd); // uses vardecl
}


// ----------------------------------------------------------------------------
// Class: IR_chillArrayRef
// ----------------------------------------------------------------------------

bool IR_chillArrayRef::is_write() const {
  
  return (iswrite); // TODO 
}


// TODO
omega::CG_outputRepr *IR_chillArrayRef::index(int dim) const {
  fprintf(stderr, "IR_xxxxArrayRef::index( %d )  \n", dim); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //chillASE->getIndex(dim)->print(); printf("\n"); fflush(stdout); 
  return new omega::CG_chillRepr( chillASE->getIndex(dim) );
}


IR_ArraySymbol *IR_chillArrayRef::symbol() const {
  //fprintf(stderr, "IR_chillArrayRef::symbol()\n"); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //fprintf(stderr, "base:  ");  chillASE->base->print();  printf("\n"); fflush(stdout); 

  
  chillAST_node *mb = chillASE->multibase(); 
  chillAST_VarDecl *vd = (chillAST_VarDecl*)mb;
  //fprintf(stderr, "symbol: '%s'\n", vd->varname);

  //fprintf(stderr, "IR_chillArrayRef symbol: '%s%s'\n", vd->varname, vd->arraypart); 
  //fprintf(stderr, "numdimensions %d\n", vd->numdimensions); 
  IR_ArraySymbol *AS =  new IR_chillArraySymbol(ir_, vd); 
  //fprintf(stderr, "ir_clang.cc returning IR_chillArraySymbol 0x%x\n", AS); 
  return  AS;
/*
  chillAST_node *b = chillASE->base;
  fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  //b->print(); printf("\n"); fflush(stdout); 
  if (b->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) {
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
    fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  }
  
  if (b->asttype == CHILLAST_NODETYPE_DECLREFEXPR)  {
    if (NULL == ((chillAST_DeclRefExpr*)b)->decl) { 
      fprintf(stderr, "IR_chillArrayRef::symbol()  var decl = 0x%x\n", ((chillAST_DeclRefExpr*)b)->decl); 
      exit(-1); 
    }
    return new IR_chillArraySymbol(ir_, ((chillAST_DeclRefExpr*)b)->decl); // -> decl?
  }
  if (b->asttype ==  CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR)  { // multidimensional array
    return (
  }
  fprintf(stderr, "IR_chillArrayRef::symbol() can't handle\n");
  fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  exit(-1); 
  return NULL; 
*/
}


bool IR_chillArrayRef::operator!=(const IR_Ref &that) const {
  //fprintf(stderr, "IR_xxxxArrayRef::operator!=\n"); 
  bool op = (*this) == that; // opposite
  return !op;
}
  
void IR_chillArrayRef::Dump() const { 
  //fprintf(stderr, "IR_chillArrayRef::Dump()  this 0x%x  chillASE 0x%x\n", this, chillASE); 
  chillASE->print(); printf("\n");fflush(stdout);
}


bool IR_chillArrayRef::operator==(const IR_Ref &that) const {
  //fprintf(stderr, "IR_xxxxArrayRef::operator==\n"); 
  //printf("I am\n"); chillASE->print(); printf("\n"); 
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  const chillAST_ArraySubscriptExpr* thatASE = l_that->chillASE;
  //printf("other is:\n");  thatASE->print(); printf("\n"); fflush(stdout);
  //fprintf(stderr, "addresses are 0x%x  0x%x\n", chillASE, thatASE ); 
  return (*chillASE) == (*thatASE);
  /*

  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  
  return this->as_ == l_that->as_;
  */
}


omega::CG_outputRepr *IR_chillArrayRef::convert() {
  //fprintf(stderr, "IR_chillArrayRef::convert()\n"); 
  CG_chillRepr *result = new  CG_chillRepr( chillASE->clone() ); 
//  omega::CG_chillRepr *temp = new omega::CG_chillRepr(static_cast<Expr*>(this->as_));
//  omega::CG_outputRepr *result = temp->clone();
  delete this;
  return result;
}


IR_Ref *IR_chillArrayRef::clone() const {
  return new IR_chillArrayRef(ir_, chillASE, iswrite);
}


// ----------------------------------------------------------------------------
// Class: IR_chillLoop
// ----------------------------------------------------------------------------
IR_chillLoop::IR_chillLoop(const IR_Code *ir, clang::ForStmt *tf) { fprintf(stderr, "IR_chillLoop::IR_chillLoop() you lose\n"); exit(-1); }; 

IR_chillLoop::IR_chillLoop(const IR_Code *ir, chillAST_ForStmt *achillforstmt) { 
  fprintf(stderr, "IR_xxxxLoop::IR_xxxxLoop()\n"); 
  fprintf(stderr, "loop is:\n"); 
  achillforstmt->print(); 

  ir_ = ir; 
  chillforstmt = achillforstmt;

  chillAST_BinaryOperator *init = (chillAST_BinaryOperator *)chillforstmt->getInit();
  chillAST_BinaryOperator *cond = (chillAST_BinaryOperator *)chillforstmt->getCond();
  // check to be sure  (assert) 
  if (!init->isAssignmentOp() || !cond->isComparisonOp() ) {
    fprintf(stderr, "ir_clang.cc, malformed loop init or cond:\n");
    achillforstmt->print(); 
    exit(-1); 
  }

  chilllowerbound = init->getRHS();
  chillupperbound = cond->getRHS();
  conditionoperator = achillforstmt->conditionoperator; 
  
  chillAST_node *inc  = chillforstmt->getInc();
  // check the increment
  //fprintf(stderr, "increment is of type %s\n", inc->getTypeString()); 
  //inc->print(); printf("\n"); fflush(stdout);

  if (inc->asttype == CHILLAST_NODETYPE_UNARYOPERATOR) { 
    if (!strcmp(((chillAST_UnaryOperator *) inc)->op, "++")) step_size_ = 1;
    else  step_size_ = -1;
  }
  else if (inc->asttype == CHILLAST_NODETYPE_BINARYOPERATOR) { 
    int beets = false;  // slang
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator *) inc;
    if (bop->isAssignmentOp()) {        // I=I+1   or similar
      chillAST_node *rhs = bop->getRHS();  // (I+1)
      // TODO looks like this will fail for I=1+I or I=J+1 etc. do more checking
      
      char *assop =  bop->getOp(); 
      //fprintf(stderr, "'%s' is an assignment op\n", bop->getOp()); 
      if (streq(assop, "+=") || streq(assop, "-=")) {
        chillAST_node *stride = rhs;
        //fprintf(stderr, "stride is of type %s\n", stride->getTypeString());
        if  (stride->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)stride)->value;
          if      (streq( assop, "+=")) step_size_ =  val;
          else if (streq( assop, "-=")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;  // += or -= but not constant stride
      }
      else if (rhs->isBinaryOperator()) { 
        chillAST_BinaryOperator *binoprhs = (chillAST_BinaryOperator *)rhs;
        chillAST_node *intlit =  binoprhs->getRHS();
        if (intlit->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)intlit)->value;
          if      (!strcmp( binoprhs->getOp(), "+")) step_size_ =  val;
          else if (!strcmp( binoprhs->getOp(), "-")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;
      }
      else beets = true;
    }
    else beets = true;

    if (beets) {
      fprintf(stderr, "malformed loop increment (or more likely unhandled case)\n");
      inc->print(); 
      exit(-1); 
    }
  } // binary operator 
  else { 
    fprintf(stderr, "IR_chillLoop constructor, unhandled loop increment\n");
      inc->print(); 
      exit(-1); 
  }
  //inc->print(0, stderr);fprintf(stderr, "\n"); 

  chillAST_DeclRefExpr *dre = (chillAST_DeclRefExpr *)init->getLHS();
  if (!dre->isDeclRefExpr()) { 
    fprintf(stderr, "malformed loop init.\n"); 
    init->print(); 
  }

  chillindex = dre; // the loop index variable

  //fprintf(stderr, "\n\nindex is ");  dre->print(0, stderr);  fprintf(stderr, "\n"); 
  //fprintf(stderr, "init is   "); 
  //chilllowerbound->print(0, stderr);  fprintf(stderr, "\n");
  //fprintf(stderr, "condition is  %s ", "<"); 
  //chillupperbound->print(0, stderr);  fprintf(stderr, "\n");
  //fprintf(stderr, "step size is %d\n\n", step_size_) ; 

  chillbody = achillforstmt->getBody(); 

  fprintf(stderr, "IR_xxxxLoop::IR_xxxxLoop() DONE\n"); 
}


omega::CG_outputRepr *IR_chillLoop::lower_bound() const {
  fprintf(stderr, "IR_xxxxLoop::lower_bound()\n"); 
  return new omega::CG_chillRepr(chilllowerbound);
}

omega::CG_outputRepr *IR_chillLoop::upper_bound() const {
  fprintf(stderr, "IR_xxxxLoop::upper_bound()\n"); 
  return new omega::CG_chillRepr(chillupperbound);
}

IR_Block *IR_chillLoop::body() const {
  fprintf(stderr, "IR_xxxxLoop::body()\n");
  //assert(isa<CompoundStmt>(tf_->getBody()));
  //fprintf(stderr, "returning a clangBLOCK corresponding to the body of the loop\n"); 
  //fprintf(stderr, "body type %s\n", chillbody->getTypeString()); 
  return new IR_chillBlock(ir_, chillbody ) ; // static_cast<CompoundStmt *>(tf_->getBody()));
}

IR_Control *IR_chillLoop::clone() const {
  fprintf(stderr, "IR_xxxxLoop::clone()\n"); 
  //chillforstmt->print(); fflush(stdout); 
  return new IR_chillLoop(ir_, chillforstmt);
}

IR_CONDITION_TYPE IR_chillLoop::stop_cond() const {
  chillAST_BinaryOperator *loopcondition = (chillAST_BinaryOperator*) chillupperbound;
  fprintf(stderr, "IR_xxxxLoop::stop_cond()\n"); 
  return conditionoperator; 
}

IR_Block *IR_chillLoop::convert() { // convert the loop to a block 
  fprintf(stderr, "IR_xxxxLoop::convert()   maybe \n"); 
  return new IR_chillBlock( ir_, chillbody ); // ?? 
  return NULL;
}

void IR_chillLoop::dump() const { 
  fprintf(stderr, "TODO:  IR_chillLoop::dump()\n"); exit(-1); 
}


// ----------------------------------------------------------------------------
// Class: IR_chillBlock
// ----------------------------------------------------------------------------
omega::CG_outputRepr *IR_chillBlock::original() const {
  fprintf(stderr, "IR_xxxxBlock::original()  TODO \n"); 
  exit(-1); 
  return NULL;
}



omega::CG_outputRepr *IR_chillBlock::extract() const {
  fflush(stdout); 
  fprintf(stderr, "IR_xxxxBlock::extract()\n"); 
  //omega::CG_chillRepr *tnl =  new omega::CG_chillRepr(getStmtList());

  // if the block refers to a compound statement, return the next level
  // of statements ;  otherwise just return a repr of the statements

  chillAST_node *code = chillAST;
  //if (chillAST != NULL) fprintf(stderr, "block has chillAST of type %s\n",code->getTypeString()); 
  //fprintf(stderr, "block has %d exploded statements\n", statements.size()); 

  omega::CG_chillRepr *OR; 
  if (0 == statements.size()) { 
    OR = new omega::CG_chillRepr(code); // presumably a compound statement ??
  }
  else { 
    fprintf(stderr, "adding a statement from IR_chillBlock::extract()\n"); 
    OR = new omega::CG_chillRepr(); // empty of statements
    for (int i=0; i<statements.size(); i++) OR->addStatement( statements[i] ); 
  }

  fflush(stdout); 
  fprintf(stderr, "IR_xxxxBlock::extract() LEAVING\n"); 
  return OR;
}

IR_Control *IR_chillBlock::clone() const {
  fprintf(stderr, "IR_xxxxBlock::clone()\n"); 
  //fprintf(stderr, "IR_xxxxBlock::clone()  %d statements\n", statements.size());
  return new IR_chillBlock( this );  // shallow copy ? 
}

void IR_chillBlock::dump() const { 
  fprintf(stderr, "IR_chillBlock::dump()  TODO\n");  return;
}

//StmtList 
vector<chillAST_node*> IR_chillBlock::getStmtList() const {
  fprintf(stderr, "IR_xxxxBlock::getStmtList()\n");
  return statements; // ?? 
}


void IR_chillBlock::addStatement( chillAST_node* s ) {
  statements.push_back( s );
}



void PrintTranslationUnit(  TranslationUnitDecl *TUD) { // ,  ASTContext &CTX ) {
  fprintf(stderr, "MY PrintTranslationUnit()\n"); 
  // TUD derived from Decl and DeclContext
  static DeclContext *DC = TUD->castToDeclContext( TUD );
  //SourceManager  SM =  CTX.getSourceManager();

  for (DeclContext::decl_iterator DI = DC->decls_begin(), DE = DC->decls_end(); DI != DE; ++DI) {
    Decl *D = *DI;
    fprintf(stderr, "D\n"); 
    if (isa<FunctionDecl>(D))  { fprintf(stderr, "FunctionDecl\n");
      //PrintFunctionDecl( dyn_cast<FunctionDecl>(D), CTX.getSourceManager(), 0);
    }
    else if (isa<VarDecl>(D)) { fprintf(stderr, "VarDecl\n"); 
      //PrintVarDecl( dyn_cast<VarDecl>(D), CTX.getSourceManager(), 0 );
    }
    else if (isa<TypedefDecl>(D)) { fprintf(stderr, "TypedefDecl\n"); 
      //PrintTypeDefDecl( dyn_cast<TypedefDecl>(D), CTX.getSourceManager(), 0 );
    }
    else if (isa<TypeAliasDecl>(D)) { fprintf(stderr, "TypeAliasDecl\n");
    }
    else fprintf(stderr, "\na declaration of type %s (%d)\n",   D->getDeclKindName(), D->getKind());
    //else if (isa<TypedefNameDecl>(D)) { fprintf(stderr, "TypedefNameDecl\n");}
  }
}

class NULLASTConsumer : public ASTConsumer
{
};



void findmanually( chillAST_node *node, char *procname, vector<chillAST_node*>& procs ) {
  //fprintf(stderr, "findmanually()                CHILL AST node of type %s\n", node->getTypeString()); 
  
  if (node->asttype == CHILLAST_NODETYPE_FUNCTIONDECL ) { 
    char *name = ((chillAST_FunctionDecl *) node)->functionName;
    //fprintf(stderr, "node name 0x%x  ", name);
    //fprintf(stderr, "%s     procname ", name); 
    //fprintf(stderr, "0x%x  ", procname);
    //fprintf(stderr, "%s\n", procname); 
    if (!strcmp( name, procname)) {
      //fprintf(stderr, "found procedure %s\n", procname ); 
      procs.push_back( node ); 
      // quit recursing. probably not correct in some horrible case
      return; 
    }
    //else fprintf(stderr, "this is not the function we're looking for\n"); 
  }


  // this is where the children can be used effectively. 
  // we don't really care what kind of node we're at. We just check the node itself
  // and then its children is needed. 

  int numc = node->children.size();  
  //fprintf(stderr, "%d children\n", numc);

  for (int i=0; i<numc; i++) {
    //fprintf(stderr, "node of type %s is recursing to child %d\n",  node->getTypeString(), i); 
    findmanually( node->children[i], procname, procs );
  }
  return; 
}

// ----------------------------------------------------------------------------
// Class: IR_clangCode_Global_Init
// ----------------------------------------------------------------------------

IR_clangCode_Global_Init *IR_clangCode_Global_Init::pinstance = 0;


IR_clangCode_Global_Init *IR_clangCode_Global_Init::Instance(char **argv) {
  fprintf(stderr, "in IR_clangCode_Global_Init::Instance(), "); 
  if (pinstance == 0) {  
    //fprintf(stderr, "\n\n***  making the one and only instance ***\n\n\n"); 
    // this is the only way to create an IR_clangCode_Global_Init
    pinstance = new IR_clangCode_Global_Init; 
    pinstance->ClangCompiler = new aClangCompiler( argv[1] );

  }
  //fprintf(stderr, "leaving  IR_clangCode_Global_Init::Instance()\n"); 
  return pinstance;
}


aClangCompiler::aClangCompiler( char *filename ) {
  
  //fprintf(stderr, "making a clang compiler for file %s\n", filename);
  SourceFileName = strdup( filename );

  // Arguments to pass to the clang frontend
  std::vector<const char *> args;
  args.push_back(strdup(filename)); 
  
  // The compiler invocation needs a DiagnosticsEngine so it can report problems
  //IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions(); // temp
  diagnosticOptions =  new DiagnosticOptions(); // private member of aClangCompiler
  
  pTextDiagnosticPrinter = new clang::TextDiagnosticPrinter(llvm::errs(), diagnosticOptions); // private member of aClangCompiler
  
  //llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(new clang::DiagnosticIDs());
  diagID =  new clang::DiagnosticIDs(); // private member of IR_clangCode_Global_Init
  
  //clang::DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  diagnosticsEngine = new clang::DiagnosticsEngine(diagID, diagnosticOptions, pTextDiagnosticPrinter);
  
  // Create the compiler invocation
  // This class is designed to represent an abstract "invocation" of the compiler, 
  // including data such as the include paths, the code generation options, 
  // the warning flags, and so on.   
  llvm::OwningPtr<clang::CompilerInvocation> CI(new clang::CompilerInvocation);
  //CI = new clang::CompilerInvocation;
  clang::CompilerInvocation::CreateFromArgs(*CI, &args[0], &args[0] + args.size(), *diagnosticsEngine);

  
  // Create the compiler instance
  Clang = new clang::CompilerInstance();  // TODO should have a better name ClangCompilerInstance


  // Get ready to report problems
  Clang->createDiagnostics(args.size(), &args[0]);
  //Clang.createDiagnostics(0, 0);
  
  
//#ifdef KIDDINGME
    //fprintf(stderr, "target\n"); 
  // Initialize target info with the default triple for our platform.
  //TargetOptions TO;
  //TO.Triple = llvm::sys::getDefaultTargetTriple();
  //TargetInfo *TI = TargetInfo::CreateTargetInfo( Clang->getDiagnostics(), TO);

  // the above causes core dumps, because clang is stupid and frees the target multiple times, corrupting memory
  IntrusiveRefCntPtr<TargetOptions>TOCRAP(new clang::TargetOptions); 
  (*TOCRAP).Triple = llvm::sys::getDefaultTargetTriple();
  TargetInfo *TI = TargetInfo::CreateTargetInfo( Clang->getDiagnostics(), *TOCRAP);
  

  Clang->setTarget(TI);
  
//#endif

  
  // ?? 
  //fprintf(stderr, "filemgr\n");
  Clang->createFileManager();
  FileManager &FileMgr = Clang->getFileManager();
  fileManager = &FileMgr;
  
  //fprintf(stderr, "sourcemgr\n");
  Clang->createSourceManager(FileMgr);
  SourceManager &SourceMgr = Clang->getSourceManager();
  sourceManager = &SourceMgr; // ?? aclangcompiler copy
  globalSRCMAN =  &SourceMgr; //  TODO   global bad 

  Clang->setInvocation(CI.take()); // Replace the current invocation



  //fprintf(stderr, "PP\n"); 
  Clang->createPreprocessor();                             // needs target 



  //clang::Preprocessor Pre = Clang->getPreprocessor();
  //preprocessor = &Pre;

  //fprintf(stderr, "CONTEXT\n");
  Clang->createASTContext();                              // needs preprocessor 
  astContext_ = &Clang->getASTContext(); 
  

  //fprintf(stderr, "filein\n"); 
  const FileEntry *FileIn = FileMgr.getFile(filename); // needs preprocessor 
  SourceMgr.createMainFileID(FileIn);
  //DiagnosticConsumer DiagConsumer = Clang->getDiagnosticClient(); 
  Clang->getDiagnosticClient().BeginSourceFile( Clang->getLangOpts(), &Clang->getPreprocessor());


  NULLASTConsumer TheConsumer; // must pass a consumer in to ParseAST(). This one does nothing

  //fprintf(stderr, "ready? Parse.\n");
  fprintf(stderr, "actually parsing file %s using clang\n", filename); 

  ParseAST( Clang->getPreprocessor(), &TheConsumer, Clang->getASTContext());

  // Translation Unit is contents of a file 
  TranslationUnitDecl* TUD = astContext_->getTranslationUnitDecl();
  // TUD->dump();  // print it out 

  // create another AST, very similar to the clang AST but not written by idiots
  fprintf(stderr, "converting entire clang AST into chill AST (ir_clang.cc)\n"); 
  chillAST_node * wholefile = ConvertTranslationUnit( TUD, filename); 

  fflush(stdout);
  //fprintf(stderr, "printing whole file\n"); 
  //fprintf(stdout, "\n\n" );   fflush(stdout);
  //wholefile->print();
  //wholefile->dump(); 
  //fflush(stdout);

  entire_file_AST = (chillAST_SourceFile *)wholefile; 


  astContext_ = &Clang->getASTContext(); 

  //#define DOUBLE
#ifdef DOUBLE
  fprintf(stderr, "DOUBLE\n"); 

  fprintf(stderr, "\n\nCLANG dump of the file I parsed:\n"); 
  llvm::OwningPtr<clang::FrontendAction> Act2(new clang::ASTDumpAction());
  // here it actually does the FrontEndAction ?? 
  if (!Clang->ExecuteAction(*Act2)) { // ast dump using builtin function
    exit(3);
  }
#endif
  fflush(stdout); fflush(stderr); 
  fflush(stdout); fflush(stderr); 
  fflush(stdout); fflush(stderr); 
  fflush(stdout); fflush(stderr); 









#ifdef DONTDOTHIS   
  
  // calling this Action seems to overwrite the astcontext and the AST. (!)
  // don't ever do this, or you lose contact with the original AST (?)

  // Create an action and make the compiler instance carry it out
  //llvm::OwningPtr<clang::CodeGenAction> Act(new clang::EmitLLVMOnlyAction());
  llvm::OwningPtr<clang::FrontendAction> Act(new clang::ASTDumpAction());
  
  fprintf(stderr, "\n\ndump of the file I parsed:\n"); 
  // here it actually does the FrontEndAction ?? 
  if (!Clang->ExecuteAction(*Act)) { // ast dump using builtin function
    exit(3);
  }
  fflush(stdout);
#endif

  
  //fprintf(stderr, "leaving aClangCompiler::aClangCompiler( filename )\n"); 
}




chillAST_FunctionDecl*  aClangCompiler::findprocedurebyname( char *procname ) {

  //fprintf(stderr, "searching through files in the clang AST\n\n");
  //fprintf(stderr, "astContext_  0x%x\n", astContext_);

  vector<chillAST_node*> procs;
  findmanually( entire_file_AST, procname, procs );

  //fprintf(stderr, "procs has %d members\n", procs.size());

  if ( procs.size() == 0 ) { 
    fprintf(stderr, "could not find function named '%s' in AST from file %s\n", procname, SourceFileName);
    exit(-1);
  }
  
  if ( procs.size() > 1 ) { 
    fprintf(stderr, "oddly, found %d functions named '%s' in AST from file %s\n", procs.size(), procname, SourceFileName);
    fprintf(stderr, "I am unsure what to do\n"); 
    exit(-1);
  }

  fprintf(stderr, "found the procedure named %s\n", procname); 
  return (chillAST_FunctionDecl *)procs[0];

}



#ifdef NOPE 
IR_clangCode_Global_Init::IR_clangCode_Global_Init(char *filename  , clang::FileSystemOptions fso ) :   
  fileManager(fso) // ,  headerSearch( headerSearchOptions, fileManager, diagengine,  languageOptions, pTargetInfo )
{ 
  /* CLANG Initialization */
  diagnosticsEngine = new clang::DiagnosticsEngine( const IntrusiveRefCntPtr<DiagnosticIDs> &Diags,
                                                    diagnosticOptionsnsa utah
                                                    ) ; // DiagnosticConsumer *client = 0,  bool ShouldOwnClient = true)
  pTextDiagnosticPrinter = new clang::TextDiagnosticPrinter(llvm::outs(), diagnosticOptions);
  diagnostic = new clang::Diagnostic(pTextDiagnosticPrinter);
  sourceManager = new clang::SourceManager(*diagnostic);
  
  // FIXME
  
  // <Warning!!> -- Platform Specific Code lives here
  // This depends on A) that you're running linux and
  // B) that you have the same GCC LIBs installed that
  // I do. 
  // Search through Clang itself for something like this,
  // go on, you won't find it. The reason why is Clang
  // has its own versions of std* which are installed under 
  // /usr/local/lib/clang/<version>/include/
  // See somewhere around Driver.cpp:77 to see Clang adding
  // its version of the headers to its include path.
  /*headerSearchOptions.AddPath("/usr/include/linux", clang::frontend::Angled, false, false, false);
    headerSearchOptions.AddPath("/usr/include/c++/4.3/tr1", clang::frontend::Angled, false, false, false);
    headerSearchOptions.AddPath("/usr/include/c++/4.3", clang::frontend::Angled, false, false, false);*/
  // </Warning!!> -- End of Platform Specific Code
  
  targetOptions.Triple = llvm::sys::getHostTriple();
  pTargetInfo = clang::TargetInfo::CreateTargetInfo(*diagnostic, targetOptions);
  clang::ApplyHeaderSearchOptions( headerSearch, headerSearchOptions, languageOptions, pTargetInfo->getTriple());
  preprocessor = new clang::Preprocessor(*diagnostic, languageOptions, *pTargetInfo, *sourceManager, headerSearch);
  clang::InitializePreprocessor(*preprocessor, preprocessorOptions, headerSearchOptions, frontendOptions); 
  const clang::FileEntry *pFile = fileManager.getFile(filename);
  sourceManager->createMainFileID(pFile);
  //preprocessor.EnterMainSourceFile();
  
  clang::TargetInfo &targetInfo = *pTargetInfo;
  
  idTable = new clang::IdentifierTable(languageOptions);
  
  builtinContext = new clang::Builtin::Context(targetInfo);
  astContext_ = new clang::ASTContext(languageOptions, *sourceManager, targetInfo, *idTable, selTable, *builtinContext, 0); 
  astConsumer_ = new Chill_ASTConsumer();
  clang::Sema sema(*preprocessor, *astContext_, *astConsumer_);
  sema.Initialize();
  clang::ParseAST(*preprocessor, astConsumer_, *astContext_);
}
#endif 


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

IR_clangCode::IR_clangCode(const char *fname, char *proc_name): IR_Code() {
  fprintf(stderr, "\nIR_xxxxCode::IR_xxxxCode()\n\n"); 
  //fprintf(stderr, "IR_clangCode::IR_clangCode( filename %s, procedure %s )\n", filename, proc_name);
  
  filename = strdup(fname); // filename is internal to IR_clangCode
  procedurename = strdup(proc_name);

  int argc = 2;
  char *argv[2];
  argv[0] = "chill";
  argv[1] = strdup(filename);
  
  // use clang to parse the input file  ?   (or is that already done?) 
  //fprintf(stderr, "IR_clangCode::IR_clangCode(), parsing input file %s\n", argv[1]);
  
  // this causes opening and parsing of the file.  
  // this is the only call to Instance that has an argument list or file name 
  IR_clangCode_Global_Init *pInstance = IR_clangCode_Global_Init::Instance(argv);
  
  if(pInstance) {
    
    aClangCompiler *Clang = pInstance->ClangCompiler;
    //fprintf(stderr, "Clang is 0x%x\n", Clang);

    //fprintf(stderr, "want to get pointer to clang ast for procedure %s\n", proc_name); 
    pInstance->setCurrentFunction( NULL );  // we have no function AST yet

    entire_file_AST = Clang->entire_file_AST;  // ugly that same name, different classes
    chillAST_FunctionDecl *localFD = Clang->findprocedurebyname( proc_name );   // stored locally
    //fprintf(stderr, "back from findprocedurebyname( %s )\n", proc_name ); 
    //localFD->print();

    pInstance->setCurrentFunction( localFD ); 

    chillAST_node *b = localFD->getBody();  // we do this just because it will get done next 
    fprintf(stderr, "in IR_xxxxCode::IR_xxxxCode(), new CG_xxxxBuilder\n"); 

    fprintf(stderr, "ir_clang.cc calling new CG_chillBuilder() umwut?\n"); 
    ocg_ = new omega::CG_chillBuilder();  // ocg == omega code gen 
    chillfunc =  localFD; 

  }

  fprintf(stderr, "IR_xxxxCode::IR_xxxxCode() returning after reading source file and finding function\n\n"); 

  //chillfunc->dump( 0, stderr); 

}


IR_clangCode::~IR_clangCode() {
  //func_->print(llvm::outs(), 4); // printing as part of the destructor !! 
  fprintf(stderr, "IR_xxxxCode::~IR_xxxxCode()\noutput happening as part of the destructor !!\n");
  //chillfunc->dump(); 
  //chillfunc->print(); 

  //fprintf(stderr, "Constant Folding before\n"); 
  //chillfunc->print(); 
  chillfunc->constantFold(); 
  //fprintf(stderr, "\nConstant Folding after\n"); 
  //chillfunc->print(); 

  chillfunc->cleanUpVarDecls(); 

  //chillfunc->dump(); 

  // TODO should output the entire file, not just the function we're working on
  chillAST_SourceFile *src = chillfunc->getSourceFile(); 
  //chillAST_node *p = chillfunc->parent; // should be translationDeclUnit
  if (src) { 
    //src->print(); // tmp
    if (src->isSourceFile()) src->printToFile(  ); 
  }
}





//TODO
IR_ScalarSymbol *IR_clangCode::CreateScalarSymbol(const IR_Symbol *sym, int i) {
  //fprintf(stderr, "IR_clangCode::CreateScalarSymbol()\n");  
  if (typeid(*sym) == typeid( IR_chillScalarSymbol ) ) {  // should be the case ??? 
    fprintf(stderr, "IR_xxxxCode::CreateScalarSymbol() from a scalar symbol\n"); 
    //fprintf(stderr, "(typeid(*sym) == typeid( IR_chillScalarSymbol )\n"); 
    const IR_chillScalarSymbol *CSS = (IR_chillScalarSymbol*) sym;
    chillAST_VarDecl *vd = CSS->chillvd;
    
    // do we have to check to see if it's already there? 
     VariableDeclarations.push_back(vd);
     chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
     bod->insertChild(0, vd);
     fprintf(stderr, "returning ... really\n"); 
    return new IR_chillScalarSymbol( this, CSS->chillvd); // CSS->clone(); 
  }

  // ?? 
  if (typeid(*sym) == typeid( IR_chillArraySymbol ) ) {  
    fprintf(stderr, "IR_xxxxCode::CreateScalarSymbol() from an array symbol?\n"); 
    const IR_chillArraySymbol *CAS = (IR_chillArraySymbol*) sym;
    //fprintf(stderr, "CAS 0x%x   chillvd = 0x%x\n", CAS, CAS->chillvd);
    //fprintf(stderr, "\nthis is the SYMBOL?: \n"); 
    //CAS->print();
    //CAS->dump();

    chillAST_VarDecl *vd = CAS->chillvd; 
    //fprintf(stderr, "\nthis is the var decl?: "); 
    //vd->print(); printf("\n"); 
    //vd->dump(); printf("\n\n");
    fflush(stdout);  
    
    // figure out the base type (probably float) of the array
    char *basetype = vd->underlyingtype;
    fprintf(stderr, "scalar will be of type SgType%s\n", basetype);   

    char tmpname[128];
    sprintf(tmpname, "newVariable%i\0", vd->chill_scalar_counter++); 
    chillAST_VarDecl * scalarvd = new chillAST_VarDecl( basetype, tmpname,  "",  NULL);  // TODO parent
    scalarvd->print(); printf("\n"); fflush(stdout); 

    fprintf(stderr, "VarDecl has parent that is a NULL\n"); 

    return (IR_ScalarSymbol *) (new IR_chillScalarSymbol( this, scalarvd)); // CSS->clone(); 
  }
  
  fprintf(stderr, "IR_clangCode::CreateScalarSymbol(), passed a sym that is not a clang scalar symbol OR an array symbol???\n"); 
  int *n = NULL;
  n[0] = 1;
  exit(-1); 
  return NULL;
}


IR_ArraySymbol *IR_clangCode::CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int i) {
  fprintf(stderr, "IR_xxxxCode::CreateArraySymbol()\n");  

  // build a new array name 
  char namestring[128];

  sprintf(namestring, "_P%d\0", entire_file_AST->chill_array_counter++);
  fprintf(stderr, "creating Array %s\n", namestring); 
    
  char arraypart[100];
  char *s = &arraypart[0];

  for (int i=0; i<size.size(); i++) { 
    omega::CG_outputRepr *OR = size[i];
    CG_chillRepr * CR = (CG_chillRepr * ) OR;
    //fprintf(stderr, "%d chillnodes\n", CR->chillnodes.size()); 
    
    // this SHOULD be 1 chillnode of type IntegerLiteral (per dimension)
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      fprintf(stderr, 
              "IR_clangCode::CreateArraySymbol() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      fprintf(stderr, "IR_clangCode::CreateArraySymbol() array dimension %d not an IntegerLiteral\n", i);
      exit(-1);
    }

    chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *)nodezero;
    int val = IL->value;
    sprintf(s, "[%d]\0", val); 
    s = &arraypart[ strlen(arraypart) ];
  }
  //fprintf(stderr, "arraypart '%s'\n", arraypart); 

  chillAST_VarDecl *vd = new chillAST_VarDecl( "float",  namestring, arraypart, NULL); // todo type from sym

  // put decl in some symbol table
  VariableDeclarations.push_back(vd);
  // insert decl in the IR_code body
  chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
  bod->insertChild(0, vd);

  return new IR_chillArraySymbol( this, vd); 
}

// TODO 
std::vector<IR_ScalarRef *> IR_clangCode::FindScalarRef(const omega::CG_outputRepr *repr) const {
  std::vector<IR_ScalarRef *> scalars;
  fprintf(stderr, "IR_clangCode::FindScalarRef() DIE\n");  exit(-1); 
  return scalars;
}



IR_ScalarRef *IR_clangCode::CreateScalarRef(const IR_ScalarSymbol *sym) {
  //fprintf(stderr, "\n***** ir_clang.cc IR_clangCode::CreateScalarRef( sym %s )\n", sym->name().c_str()); 
  //DeclRefExpr *de = new (vd->getASTContext())DeclRefExpr(static_cast<ValueDecl*>(vd), vd->getType(), SourceLocation());
  //fprintf(stderr, "sym 0x%x\n", sym); 

  IR_chillScalarRef *sr = new IR_chillScalarRef(this, buildDeclRefExpr(((IR_chillScalarSymbol*)sym)->chillvd)); // uses VarDecl to mak a declrefexpr
  //fprintf(stderr, "returning ScalarRef with dre 0x%x\n", sr->dre); 
  return sr; 
  //return (IR_ScalarRef *)NULL;
}



IR_ArrayRef *IR_clangCode::CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index) {
  fprintf(stderr, "IR_clangCode::CreateArrayRef()   ir_clang.cc\n"); 
  fprintf(stderr, "sym->n_dim() %d   index.size() %d\n", sym->n_dim(), index.size()); 

  int t;
  if(sym->n_dim() != index.size()) {
    throw std::invalid_argument("incorrect array symbol dimensionality   dim != size    ir_clang.cc L2359");
  }

  const IR_chillArraySymbol *c_sym = static_cast<const IR_chillArraySymbol *>(sym);
  chillAST_VarDecl *vd = c_sym->chillvd;
  std::vector<chillAST_node *> inds;

  //fprintf(stderr, "%d array indeces\n", sym->n_dim()); 
  for (int i=0; i< index.size(); i++) { 
    CG_chillRepr *CR = (CG_chillRepr *)index[i];
   
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      fprintf(stderr, 
              "IR_clangCode::CreateArrayRef() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    inds.push_back( CR->chillnodes[0] );

    /* 
       chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      fprintf(stderr,"IR_clangCode::CreateArrayRef() array dimension %d not an IntegerLiteral\n",i);
      fprintf(stderr, "it is a %s\n", nodezero->getTypeString()); 
      nodezero->print(); printf("\n"); fflush(stdout); 
      exit(-1);
    }

    chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *)nodezero;
    int val = IL->value;
    inds.push_back( val );
    */
  }

  // now we've got the vardecl AND the indeces to make a chillAST that represents the array reference
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( vd, inds );

  return new IR_chillArrayRef( this, ASE, 0 ); 
}

// find all array references ANYWHERE in this block of code  ?? 
std::vector<IR_ArrayRef *> IR_clangCode::FindArrayRef(const omega::CG_outputRepr *repr) const {
  //fprintf(stderr, "FindArrayRef()\n"); 
  std::vector<IR_ArrayRef *> arrays;
  const omega::CG_chillRepr *crepr = static_cast<const omega::CG_chillRepr *>(repr); 
  vector<chillAST_node*> chillstmts = crepr->getChillCode();

  //fprintf(stderr, "there are %d chill statements in this repr\n", chillstmts.size()); 

  std::vector<chillAST_ArraySubscriptExpr*> refs; 
  for (int i=0; i<chillstmts.size(); i++) { 
    //fprintf(stderr, "\nchillstatement %d = ", i); chillstmts[i]->print(0, stderr); fprintf(stderr, "\n"); 
    chillstmts[i]->findArrayRefs( &refs, false );
  }
  //fprintf(stderr, "%d total refs\n", refs.size());
  for (int i=0; i<refs.size(); i++) { 
    if (refs[i]->imreadfrom) { 
      //fprintf(stderr, "ref[%d] going to be put in TWICE, as both read and write\n", i); 
      arrays.push_back( new IR_chillArrayRef( this, refs[i], 0 ) );  // UGLY TODO dual usage of a ref in "+="
    }
    arrays.push_back( new IR_chillArrayRef( this, refs[i], refs[i]->imwrittento ) ); // this is wrong
    // we need to know whether this reference will be written, etc. 
  }

  /* 
  if(chillstmts.size() > 1) {
    for(int i=0; i<tnl->size(); ++i) {
      omega::CG_chillRepr *r = new omega::CG_chillRepr((*tnl)[i]);
      std::vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
    }
  } else if(chillstmts.size() == 1) {
    Stmt *s = (*tnl)[0];
    
    if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)) {
      for(CompoundStmt::body_iterator bi = cs->body_begin(); bi != cs->body_end(); ++bi) {
        omega::CG_chillRepr *r = new omega::CG_chillRepr(*bi);
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
      }
    } else if(ForStmt *fs = dyn_cast<ForStmt>(s)) {
      omega::CG_chillRepr *r = new omega::CG_chillRepr(fs->getBody());
      std::vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
    } else if(IfStmt *ifs = dyn_cast<IfStmt>(s)) {
      omega::CG_chillRepr *r = new omega::CG_chillRepr(ifs->getCond());
      std::vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
      r = new omega::CG_chillRepr(ifs->getThen());
      a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
      if(Stmt *s_else = ifs->getElse()) {
        r = new omega::CG_chillRepr(s_else);
        a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
      }
    } else if(Expr *e = dyn_cast<Expr>(s)) {
      omega::CG_chillRepr *r = new omega::CG_chillRepr(static_cast<Expr*>(s));
      std::vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      std::copy(a.begin(), a.end(), back_inserter(arrays));
    } else throw ir_error("control structure not supported");
  }
  */
/* 
  else { // We have an expression
    Expr *op = static_cast<const omega::CG_chillRepr *>(repr)->GetExpression();
    if(0) { // TODO: Handle pointer reference exp. here
    } else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(op)) {
      omega::CG_chillRepr *r1 = new omega::CG_chillRepr(bop->getLHS());
      std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;      
      std::copy(a1.begin(), a1.end(), back_inserter(arrays));
      omega::CG_chillRepr *r2 = new omega::CG_chillRepr(bop->getRHS());
      std::vector<IR_ArrayRef *> a2 = FindArrayRef(r2);
      delete r2;
      std::copy(a2.begin(), a2.end(), back_inserter(arrays));
    } else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(op)) {
      omega::CG_chillRepr *r1 = new omega::CG_chillRepr(uop->getSubExpr());
      std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;      
      std::copy(a1.begin(), a1.end(), back_inserter(arrays));
    } //else throw ir_error("Invalid expr. type passed to FindArrayRef");
  }
  */
  return arrays;
}


std::vector<IR_Control *> IR_clangCode::FindOneLevelControlStructure(const IR_Block *block) const {
  fprintf(stderr, "IR_xxxxCode::FindOneLevelControlStructure()\n"); 
  const IR_chillBlock *CB = (const IR_chillBlock *) block; 
  //fprintf(stderr, "block 0x%x\n", block); 

  std::vector<IR_Control *> controls;

  chillAST_node *blockast = CB->chillAST;
  //fprintf(stderr, "blockast 0x%x\n", blockast); 
  if (blockast == NULL) { 
    int numstmts = CB->statements.size(); 
    fprintf(stderr, "%d statements\n", numstmts); 

    if (numstmts == 0) return controls; 

    else if (numstmts == 1) blockast = CB->statements[0]; // a single statement

    else { 
      fprintf(stderr, "IR_xxxBlock is dumb, with multiple ways to hold statements\n");
      exit(-1); // TODO FIX 
    }
  }

  //fprintf(stderr, "block's AST is of type %s\n", blockast->getTypeString()); 
  //blockast->print(); printf("\n\n"); fflush(stdout);

  
  //vector<chillAST_node *> funcchildren = chillfunc->getChildren(); 
  //fprintf(stderr, "%d children of clangcode\n", funcchildren.size());  // includes parameters

  // build up a vector of "controls".
  // a run of straight-line code (statements that can't cause branching) will be 
  // bundled up into an IR_Block
  // ifs and loops will get their own entry

  vector<chillAST_node *> children;

  
  if (blockast->asttype == CHILLAST_NODETYPE_FORSTMT) { fflush(stdout); 
    fprintf(stderr, "found a top level For statement (Loop)\n"); 
    fprintf(stderr, "For Stmt (loop) is:\n");
    blockast->print(); 
    fprintf(stderr, "pushing the loop at TOP\n"); 

    controls.push_back( new IR_chillLoop( this, (chillAST_ForStmt *)blockast)); 
  }
  //else if (blockast->asttype == CHILLAST_NODETYPE_IFSTMT) { 
  //  controls.push_back( new IR_clangIf( this, (chillAST_IfStmt *)blockast)); 
  //} 
  else if (blockast->asttype == CHILLAST_NODETYPE_COMPOUNDSTMT || 
           blockast->asttype == CHILLAST_NODETYPE_FUNCTIONDECL) { 

    if (blockast->asttype == CHILLAST_NODETYPE_FUNCTIONDECL) { 
      //fprintf(stderr, "ir_clanc.cc blockast->asttype == CHILLAST_NODETYPE_FUNCTIONDECL\n"); 

      chillAST_FunctionDecl *FD =  (chillAST_FunctionDecl *)blockast;
      chillAST_node *bod = FD->getBody(); 
      //fprintf(stderr, "bod 0x%x\n", bod); 

      children = bod->getChildren(); 

      //fprintf(stderr, "FunctionDecl body is of type %s\n", bod->getTypeString()); 
      //fprintf(stderr, "found a top level FunctionDecl (Basic Block)\n"); 
      //fprintf(stderr, "basic block has %d statements\n", children.size() );
      //fprintf(stderr, "basic block is:\n");
      //bod->print(); 
    }
    else /* CompoundStmt */ {
      //fprintf(stderr, "found a top level Basic Block\n"); 
      children = blockast->getChildren();
    }
    
    int numchildren = children.size(); 
    //fprintf(stderr, "basic block has %d statements\n", numchildren);
    //fprintf(stderr, "basic block is:\n");
    //fprintf(stderr, "{\n");
    //blockast->print(); 
    //fprintf(stderr, "}\n");
    
    int ns; 
    IR_chillBlock *basicblock = new IR_chillBlock(this); // no statements
    for (int i=0; i<numchildren; i++) { 
      //fprintf(stderr, "child %d is of type %s\n", i, children[i]->getTypeString());
      CHILL_ASTNODE_TYPE typ = children[i]->asttype;
      if (typ == CHILLAST_NODETYPE_LOOP) {
        if (numchildren == 1) { 
          fprintf(stderr, "found a For statement (Loop)\n"); 
        }
        else { 
          fprintf(stderr, "found a For statement (Loop) at %d within a Basic Block\n", i); 
        }
        //children[i]->print(); printf("\n"); fflush(stdout);
        
        ns = basicblock->numstatements();
        if (ns) {
          fprintf(stderr, "pushing a run of statements %d to %d as a block\n", i-ns, i-1); 
          controls.push_back( basicblock );
          basicblock = new IR_chillBlock(this); // start a new one
        }
        
        //fprintf(stderr, "pushing the loop at %d\n", i); 
        controls.push_back( new IR_chillLoop(this, (chillAST_ForStmt *)children[i] )); 
        
      }
      //else if (typ == CHILLAST_NODETYPE_IFSTMT ) // TODO 
      else { // straight line code
        //fprintf(stderr, "straight line code\n"); 
        basicblock->addStatement( children[i] );
        //fprintf(stderr, "child %d = \n", i); children[i]->print(); printf("\n"); fflush(stdout); 
        //fprintf(stderr, "child %d is part of a basic block\n", i);
      }
    } // for each child
    ns = basicblock->numstatements();
    //fprintf(stderr, "ns %d\n", ns); 
    if (ns != 0 ) { 
      if (ns != numchildren) {
        //fprintf(stderr, "end of body ends the run of %d statements in the Basic Block\n", ns); 
        controls.push_back( basicblock );
      }
      else { 
        //fprintf(stderr, "NOT sending straightline run of statements, because it would be the entire block. There are no control statements in the block\n"); 
      }
    }
    //else fprintf(stderr, "NOT sending the last run of %d statements\n", ns); 
    
  }
  else { 
    fprintf(stderr, "IR_clangCode::FindOneLevelControlStructure(), block is a %s???\n", blockast->getTypeString());
    exit(-1);
  }
  
  fprintf(stderr, "returning vector of %d controls\n", controls.size() ); 
  return controls;
}


IR_Block *IR_clangCode::MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const {
  fprintf(stderr, "IR_xxxxCode::MergeNeighboringControlStructures  %d controls\n", controls.size());

  if (controls.size() == 0)
    return NULL;
  
  IR_chillBlock *CBlock =  new IR_chillBlock(controls[0]->ir_); // the thing we're building

  vector<chillAST_node*> statements;
  chillAST_node *parent = NULL; 
   for (int i = 0; i < controls.size(); i++) {
    switch (controls[i]->type()) {
    case IR_CONTROL_LOOP: {
      fprintf(stderr, "control %d is IR_CONTROL_LOOP\n", i); 
      chillAST_ForStmt *loop =  static_cast<IR_chillLoop *>(controls[i])->chillforstmt;
      if (parent == NULL) {
        parent = loop->parent;
      } else {
        if (parent != loop->parent) { 
          throw ir_error("controls to merge not at the same level");
        }
      }
      CBlock->addStatement( loop );
      break;
     }
    case IR_CONTROL_BLOCK: {
      fprintf(stderr, "control %d is IR_CONTROL_BLOCK\n", i); 
      IR_chillBlock *CB =  static_cast<IR_chillBlock*>(controls[i]);
      vector<chillAST_node*> blockstmts = CB->statements;
      if (statements.size() != 0) { 
        for (int j=0; j< blockstmts.size(); j++) {
          if (parent == NULL) {
            parent = blockstmts[j]->parent;
          }
          else { 
            if (parent !=  blockstmts[j]->parent) { 
              throw ir_error("ir_clang.cc  IR_clangCode::MergeNeighboringControlStructures  controls to merge not at the same level");
            }
          }
          CBlock->addStatement( blockstmts[j] );
        }
      }
      else {
        if (CB->chillAST)  CBlock->addStatement(CBlock->chillAST); // if this is a block, add theblock's statements? 
        else { // should never happen
          fprintf(stderr, "WARNING: ir_clang.cc  IR_clangCode::MergeNeighboringControlStructures");
          fprintf(stderr, "    empty IR_CONTROL_BLOCK \n");
        }
      }
      break;
    }
    default:
      throw ir_error("unrecognized control to merge");
    }
   } // for each control

   return CBlock; 
}


IR_Block *IR_clangCode::GetCode() const {    // return IR_Block corresponding to current function?
  //fprintf(stderr, "IR_clangCode::GetCode()\n"); 
  //Stmt *s = func_->getBody();  // clang statement, and clang getBody
  //fprintf(stderr, "chillfunc 0x%x\n", chillfunc);

  //chillAST_node *bod = chillfunc->getBody();  // chillAST 
  //fprintf(stderr, "printing the function getBody()\n"); 
  //fprintf(stderr, "sourceManager 0x%x\n", sourceManager); 
  //bod->print(); 

  return new IR_chillBlock(this, chillfunc ) ; 
}


void IR_clangCode::ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr) {
  fflush(stdout); 
  fprintf(stderr, "IR_xxxxCode::ReplaceCode( old, *repr)\n"); 

  CG_chillRepr *chillrepr = (CG_chillRepr *) repr;
  vector<chillAST_node*>  newcode = chillrepr->getChillCode();
  int numnew = newcode.size();

  //fprintf(stderr, "new code (%d) is\n", numnew); 
  //for (int i=0; i<numnew; i++) { 
  //  newcode[i]->print(0, stderr);
  //  fprintf(stderr, "\n"); 
  //} 

  struct IR_chillLoop* cloop;

  vector<chillAST_VarDecl*> olddecls;
  chillfunc->gatherVarDecls( olddecls );
  //fprintf(stderr, "\n%d old decls   they are:\n", olddecls.size()); 
  //for (int i=0; i<olddecls.size(); i++) {
  //  fprintf(stderr, "olddecl[%d]  ox%x  ",i, olddecls[i]); 
  //  olddecls[i]->print(); printf("\n"); fflush(stdout); 
  //} 


  //fprintf(stderr, "num new stmts %d\n", numnew); 
  //fprintf(stderr, "new code we're look for decls in:\n"); 
  vector<chillAST_VarDecl*> decls;
  for (int i=0; i<numnew; i++)  {
    //newcode[i]->print(0,stderr);
    //fprintf(stderr, "\n"); 
    newcode[i]->gatherVarUsage( decls );
  }

  //fprintf(stderr, "\n%d new vars used  they are:\n", decls.size()); 
  //for (int i=0; i<decls.size(); i++) {
  //  fprintf(stderr, "decl[%d]  ox%x  ",i, decls[i]); 
  //  decls[i]->print(); printf("\n"); fflush(stdout); 
  //} 


  for (int i=0; i<decls.size(); i++) {
    //fprintf(stderr, "\nchecking "); decls[i]->print(); printf("\n"); fflush(stdout); 
    int inthere = 0; 
    for (int j=0; j<VariableDeclarations.size(); j++) { 
      if (VariableDeclarations[j] == decls[i]) { 
        //fprintf(stderr, "it's in the Variable Declarations()\n");
      }
    }
    for (int j=0; j<olddecls.size(); j++) { 
      if (decls[i] == olddecls[j]) { 
        //fprintf(stderr, "it's in the olddecls (exactly)\n");
        inthere = 1;
      }
      if (streq(decls[i]->varname, olddecls[j]->varname)) { 
        if (streq(decls[i]->arraypart, olddecls[j]->arraypart)) { 
          //fprintf(stderr, "it's in the olddecls (INEXACTLY)\n");
          inthere = 1;
        }
      }
    }
    if (!inthere) {
      //fprintf(stderr, "inserting decl[%d] for ",i); decls[i]->print(); printf("\n");fflush(stdout); 
      chillfunc->getBody()->insertChild(0, decls[i]); 
      olddecls.push_back( decls[i] ); 
    }
  }
  
  chillAST_node *par;
  switch (old->type()) {
  case IR_CONTROL_LOOP: 
    {
      //fprintf(stderr, "old is IR_CONTROL_LOOP\n"); 
      cloop = (struct IR_chillLoop* )old;
      chillAST_ForStmt *forstmt = cloop->chillforstmt;

      fprintf(stderr, "old was\n");
      forstmt->print(); printf("\n"); fflush(stdout);

      //fprintf(stderr, "\nnew code is\n");
      //for (int i=0; i<numnew; i++) { newcode[i]->print(); printf("\n"); } 
      //fflush(stdout);
      

      par = forstmt->parent;
      if (!par) {
        fprintf(stderr, "old parent was NULL\n"); 
        fprintf(stderr, "ir_clang.cc that will not work very well.\n");
        exit(-1); 
      }

      

      fprintf(stderr, "\nold parent was\n\n{\n"); 
      par->print(); printf("\n"); fflush(stdout);
      fprintf(stderr, "\n}\n"); 

      vector<chillAST_node*>  oldparentcode = par->getChildren(); // probably only works for compoundstmts
      //fprintf(stderr, "ir_clang.cc oldparentcode\n"); 

      // find loop in the parent
      int index = -1;
      int numstatements = oldparentcode.size();
      for (int i=0; i<numstatements; i++) if (oldparentcode[i] == forstmt) { index = i; }
      if (index == -1) { 
        fprintf(stderr, "ir_clang.cc can't find the loop in its parent\n"); 
        exit(-1); 
      }
      //fprintf(stderr, "loop is index %d\n", index); 

      // insert the new code
      par->setChild(index, newcode[0]);    // overwrite old stmt
      //fprintf(stderr, "inserting %s 0x%x as index %d of 0x%x\n", newcode[0]->getTypeString(), newcode[0], index, par); 
      // do we need to update the IR_cloop? 
      cloop->chillforstmt = (chillAST_ForStmt*) newcode[0]; // ?? DFL 



      //printf("inserting "); newcode[0]->print(); printf("\n"); 
      if (numnew > 1){ 
        //oldparentcode.insert( oldparentcode.begin()+index+1, numnew-1, NULL); // allocate in bulk
        
        // add the rest of the new statements
        for (int i=1; i<numnew; i++) {
          printf("inserting "); newcode[i]->print(); printf("\n"); 
          par->insertChild( index+i, newcode[i] );  // sets parent
        }
      }

      // TODO add in (insert) variable declarations that go with the new loops
      

      fflush(stdout); 
    }
    break; 
  case IR_CONTROL_BLOCK:
    fprintf(stderr, "old is IR_CONTROL_BLOCK\n"); 
    fprintf(stderr, "IR_clangCode::ReplaceCode() stubbed out\n"); 
    exit(-1); 
    //tf_old = static_cast<IR_chillBlock *>(old)->getStmtList()[0];
    break; 
  default:
    throw ir_error("control structure to be replaced not supported");
    break;    
  }
  
  fflush(stdout); 
  //fprintf(stderr, "\nafter inserting %d statements into the Clang IR,", numnew);
  fprintf(stderr, "\nnew parent2 is\n\n{\n");
  vector<chillAST_node*>  newparentcode = par->getChildren();
  for (int i=0; i<newparentcode.size(); i++) { 
    fflush(stdout); 
    //fprintf(stderr, "%d ", i); 
    newparentcode[i]->print(); printf(";\n"); fflush(stdout); 
  }



  fprintf(stderr, "}\n"); 

}




void IR_clangCode::ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr) {
  fprintf(stderr, "IR_xxxxCode::ReplaceExpression()\n");

  if (typeid(*old) == typeid(IR_chillArrayRef)) {
    //fprintf(stderr, "expressions is IR_chillArrayRef\n"); 
    IR_chillArrayRef *CAR = (IR_chillArrayRef *)old;
    chillAST_ArraySubscriptExpr* CASE = CAR->chillASE;
    printf("\nreplacing old "); CASE->print(); printf("\n"); fflush(stdout);

    omega::CG_chillRepr *crepr = (omega::CG_chillRepr *)repr;
    if (crepr->chillnodes.size() != 1) { 
      fprintf(stderr, "IR_clangCode::ReplaceExpression(), replacing with %d chillnodes???\n"); 
      //exit(-1);
    }
    
    chillAST_node *newthing = crepr->chillnodes[0]; 
    fprintf(stderr, "with new "); newthing->print(); printf("\n\n"); fflush(stdout);

    if (!CASE->parent) { 
      fprintf(stderr, "IR_clangCode::ReplaceExpression()  old has no parent ??\n"); 
      exit(-1); 
    }

    fprintf(stderr, "OLD parent = "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n"); fflush(stdout); 

    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 

    CASE->parent->replaceChild( CASE, newthing ); 

    fprintf(stderr, "after replace parent is "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n\n"); fflush(stdout); 



    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 


  }
  else  if (typeid(*old) == typeid(IR_chillScalarRef)) {
    fprintf(stderr, "IR_clangCode::ReplaceExpression()  IR_chillScalarRef unhandled\n"); 
  }
  else { 
    fprintf(stderr, "UNKNOWN KIND OF REF\n"); exit(-1); 
  }

  delete old;
}


// TODO 
IR_CONDITION_TYPE IR_clangCode::QueryBooleanExpOperation(const omega::CG_outputRepr *repr) const {
  return IR_COND_UNKNOWN;
}



IR_OPERATION_TYPE IR_clangCode::QueryExpOperation(const omega::CG_outputRepr *repr) const {
  //fprintf(stderr, "IR_clangCode::QueryExpOperation()\n");

  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *node = crepr->chillnodes[0];
  //fprintf(stderr, "chillAST node type %s\n", node->getTypeString());

  // really need to be more rigorous than this hack  // TODO 
  if (node->isImplicitCastExpr()) node = ((chillAST_ImplicitCastExpr*)node)->subexpr;
  if (node->isCStyleCastExpr())   node = ((chillAST_CStyleCastExpr*)  node)->subexpr;
  if (node->isParenExpr())        node = ((chillAST_ParenExpr*)       node)->subexpr;

  if (node->isIntegerLiteral() || node->isFloatingLiteral())  return IR_OP_CONSTANT;
  else if (node->isBinaryOperator() || node->isUnaryOperator()) {
    char *opstring;
    if (node->isBinaryOperator()) 
      opstring= ((chillAST_BinaryOperator*)node)->op; // TODO enum
    else
      opstring= ((chillAST_UnaryOperator*)node)->op; // TODO enum
      
    if (!strcmp(opstring, "+")) return IR_OP_PLUS;
    if (!strcmp(opstring, "-")) return IR_OP_MINUS;
    if (!strcmp(opstring, "*")) return IR_OP_MULTIPLY;
    if (!strcmp(opstring, "/")) return IR_OP_DIVIDE;
    if (!strcmp(opstring, "=")) return IR_OP_ASSIGNMENT;

    fprintf(stderr, "ir_clang.cc  IR_clangCode::QueryExpOperation() UNHANDLED Binary(or Unary)Operator op type (%s)\n", opstring); 
    exit(-1);
  }
  else if (node->isDeclRefExpr() ) return  IR_OP_VARIABLE; // ?? 
  //else if (node->is ) return  something;
  else { 
    fprintf(stderr, "IR_clangCode::QueryExpOperation()  UNHANDLED NODE TYPE %s\n", node->getTypeString());
    exit(-1); 
  }

  /* CLANG 
  Expr *e = static_cast<const omega::CG_chillRepr *>(repr)->GetExpression();
  if(isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e)) return IR_OP_CONSTANT;
  else if(isa<DeclRefExpr>(e)) return IR_OP_VARIABLE;
  else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(e)) {
    switch(bop->getOpcode()) {
    case BO_Assign: return IR_OP_ASSIGNMENT;
    case BO_Add: return IR_OP_PLUS;
    case BO_Sub: return IR_OP_MINUS;
    case BO_Mul: return IR_OP_MULTIPLY;
    case BO_Div: return IR_OP_DIVIDE;
    default: return IR_OP_UNKNOWN;
    }
  } else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(e)) {
    switch(uop->getOpcode()) {
    case UO_Minus: return IR_OP_NEGATIVE;
    case UO_Plus: return IR_OP_POSITIVE;
    default: return IR_OP_UNKNOWN;
    }
  } else if(ConditionalOperator *cop = dyn_cast<ConditionalOperator>(e)) {
    BinaryOperator *bop;
    if(bop = dyn_cast<BinaryOperator>(cop->getCond())) {
      if(bop->getOpcode() == BO_GT) return IR_OP_MAX;
      else if(bop->getOpcode() == BO_LT) return IR_OP_MIN;
    } else return IR_OP_UNKNOWN;
    
  } 
  
  else if(e == NULL) return IR_OP_NULL;
  else return IR_OP_UNKNOWN;
  }
   END CLANG */
}


std::vector<omega::CG_outputRepr *> IR_clangCode::QueryExpOperand(const omega::CG_outputRepr *repr) const { 
  //fprintf(stderr, "IR_clangCode::QueryExpOperand()\n"); 
  std::vector<omega::CG_outputRepr *> v;
  
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  //Expr *e = static_cast<const omega::CG_chillRepr *>(repr)->GetExpression(); wrong.. CLANG
  chillAST_node *e = crepr->chillnodes[0]; // ?? 
  //e->print(); printf("\n"); fflush(stdout); 

  // really need to be more rigorous than this hack  // TODO 
  if (e->isImplicitCastExpr()) e = ((chillAST_ImplicitCastExpr*)e)->subexpr;
  if (e->isCStyleCastExpr())   e = ((chillAST_CStyleCastExpr*)  e)->subexpr;
  if (e->isParenExpr())        e = ((chillAST_ParenExpr*)       e)->subexpr;


  //if(isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e) || isa<DeclRefExpr>(e)) {
  if (e->isIntegerLiteral() || e->isFloatingLiteral() || e->isDeclRefExpr() ) { 
    //fprintf(stderr, "it's a constant\n"); 
    omega::CG_chillRepr *repr = new omega::CG_chillRepr(e);
    v.push_back(repr);
    //} else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(e)) {
  } else if (e->isBinaryOperator()) { 
    //fprintf(stderr, "ir_clang.cc BOP TODO\n"); exit(-1); // 
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator*)e;
    char *op = bop->op;  // TODO enum for operator types
    if (streq(op, "=")) { 
      v.push_back(new omega::CG_chillRepr( bop->rhs ));  // for assign, return RHS
    }
    else if (streq(op, "+") || streq(op, "-") || streq(op, "*") || streq(op, "/") ) {
      v.push_back(new omega::CG_chillRepr( bop->lhs ));  // for +*-/ return both lhs and rhs
      v.push_back(new omega::CG_chillRepr( bop->rhs )); 
    }
    else { 
      fprintf(stderr, "ir_clang.cc  IR_clangCode::QueryExpOperand() Binary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // BinaryOperator
  else if  (e->isUnaryOperator()) { 
    omega::CG_chillRepr *repr;
    chillAST_UnaryOperator *uop = (chillAST_UnaryOperator*)e;
    char *op = uop->op; // TODO enum
    if (streq(op, "+") || streq(op, "-")) {
      v.push_back( new omega::CG_chillRepr( uop->subexpr ));
    }
    else { 
      fprintf(stderr, "ir_clang.cc  IR_clangCode::QueryExpOperand() Unary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // unaryoperator
  else { 
    fprintf(stderr, "ir_clang.cc  IR_clangCode::QueryExpOperand() UNHANDLED node type %s\n", e->getTypeString()); 
    exit(-1); 
  }
    

    /*   
  Expr *op1, *op2;
    switch(bop->getOpcode()) {
    case BO_Assign:
      op2 = bop->getRHS();
      repr = new omega::CG_chillRepr(op2);
      v.push_back(repr);
      break;
    case BO_Add:
    case BO_Sub:
    case BO_Mul:
    case BO_Div:
      op1 = bop->getLHS();
      repr = new omega::CG_chillRepr(op1);
      v.push_back(repr);
      op2 = bop->getRHS();
      repr = new omega::CG_chillRepr(op2);
      v.push_back(repr);
      break;
    default:
      throw ir_error("operation not supported");
    }
    */
    //} else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(e)) {
    //} else if(e->isUnaryOperator()) { 
    /* 
    omega::CG_chillRepr *repr;
    
    switch(uop->getOpcode()) {
    case UO_Minus:
    case UO_Plus:
      op1 = uop->getSubExpr();
      repr = new omega::CG_chillRepr(op1);
      v.push_back(repr);
      break;
    default:
      throw ir_error("operation not supported");
    }
    */
    //} else if(ConditionalOperator *cop = dyn_cast<ConditionalOperator>(e)) {
    //omega::CG_chillRepr *repr;
    
    // TODO: Handle conditional operator here
    //} else  throw ir_error("operand type UNsupported");
  
  return v;
}

IR_Ref *IR_clangCode::Repr2Ref(const omega::CG_outputRepr *repr) const {
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *node = crepr->chillnodes[0]; 
  
  //Expr *e = static_cast<const omega::CG_chillRep *>(repr)->GetExpression();

  if(node->isIntegerLiteral()) { 
    // FIXME: Not sure if it'll work in all cases (long?)
    int val = ((chillAST_IntegerLiteral*)node)->value; 
    return new IR_chillConstantRef(this, static_cast<omega::coef_t>(val) ); 
  } else if(node->isFloatingLiteral()) { 
    float val = ((chillAST_FloatingLiteral*)node)->value; 
    return new IR_chillConstantRef(this, val );
  } else if(node->isDeclRefExpr()) { 
    //fprintf(stderr, "ir_clang.cc  IR_clangCode::Repr2Ref()  declrefexpr TODO\n"); exit(-1); 
    return new IR_chillScalarRef(this, (chillAST_DeclRefExpr*)node);  // uses DRE
  } else  { 
    fprintf(stderr, "ir_clang.cc IR_clangCode::Repr2Ref() UNHANDLED node type %s\n", node->getTypeString()); 
    exit(-1); 
    //assert(0);
  }
}

chillAST_node * ConvertMemberExpr( clang::MemberExpr *clangME , chillAST_node *) { 
  fprintf(stderr, "ConvertMemberExpr()\n"); 
  
  clang::Expr *E = clangME->getBase(); 
  E->dump();

  chillAST_node *base = ConvertGenericClangAST( clangME->getBase(), NULL );

  DeclarationNameInfo memnameinfo = clangME->getMemberNameInfo(); 
  DeclarationName DN = memnameinfo.getName();
  const char *member = DN.getAsString().c_str();
  //fprintf(stderr, "%s\n", DN.getAsString().c_str());  

  chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, NULL, clangME ); 

  fprintf(stderr, "this is the Member Expresion\n"); 
  ME->print(); 
  fprintf(stderr, "\n"); 

  return ME; 
  
} 
