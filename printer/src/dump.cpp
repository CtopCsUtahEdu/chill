//
// Created by ztuowen on 9/24/16.
//

#include "printer/dump.h"

using namespace chill::printer;
using namespace std;

template<typename T>
void dumpVector(GenericPrinter *p, string ident, vector<T *> *n, ostream &o) {
  for (int i = 0; i < n->size(); ++i)
    p->print(ident, (*n)[i], o);
}

void Dump::print(string ident, chillAST_node *n, ostream &o) {
  if (!n) return;
  o << "(" << n->getTypeString() << " ";
  if (n->getSymbolTable()) {
    o << "(VarScope: ";
    dumpVector(this, ident, n->getSymbolTable(), o);
    o << ") ";
  }
  o << ": ";
  // Recurse
  GenericPrinter::print(ident, n, o);
  o << ") ";
}

void Dump::printS(std::string ident, chillAST_ArraySubscriptExpr *n, std::ostream &o) {
  if (n->basedecl)
    o << "(" << n->basedecl->varname << ") ";
  if (n->basedecl && n->basedecl->vartype)
    o << n->basedecl->vartype;
  if (n->imwrittento) {
    if (n->imreadfrom)
      o << "lvalue AND rvalue ";
    else
      o << "lvalue ";
  } else o << "rvalue ";
  print(ident, n->base, o);
  print(ident, n->index, o);
}

void Dump::printS(std::string ident, chillAST_BinaryOperator *n, std::ostream &o) {
  o << n->op << " ";
  if (n->getLHS()) print(ident, n->getLHS(), o);
  else o << "(NULL) ";
  if (n->getRHS()) print(ident, n->getRHS(), o);
  else o << "(NULL) ";
}

void Dump::printS(std::string ident, chillAST_CallExpr *n, std::ostream &o) {
  dumpVector(this, ident, &(n->getChildren()), o);
}

void Dump::printS(std::string ident, chillAST_CompoundStmt *n, std::ostream &o) {
  dumpVector(this, ident, &(n->getChildren()), o);
}

void Dump::printS(std::string ident, chillAST_CStyleAddressOf *n, std::ostream &o) {
  print(ident, n->subexpr, o);
}

void Dump::printS(std::string ident, chillAST_CStyleCastExpr *n, std::ostream &o) {
  o << n->towhat << " ";
  print(ident, n->subexpr, o);
}

void Dump::printS(std::string ident, chillAST_CudaFree *n, std::ostream &o) {
  print(ident, n->variable, o);
}

void Dump::printS(std::string ident, chillAST_CudaKernelCall *n, std::ostream &o) {
  chill_error_printf("Not implemented");
}

void Dump::printS(std::string ident, chillAST_CudaMalloc *n, std::ostream &o) {
  print(ident, n->devPtr, o);
  print(ident, n->sizeinbytes, o);
}

void Dump::printS(std::string ident, chillAST_CudaMemcpy *n, std::ostream &o) {
  o << n->cudaMemcpyKind << " ";
  print(ident, n->dest, o);
  print(ident, n->src, o);
  print(ident, n->size, o);
}

void Dump::printS(std::string ident, chillAST_CudaSyncthreads *n, std::ostream &o) {}

void Dump::printS(std::string ident, chillAST_DeclRefExpr *n, std::ostream &o) {
  chillAST_VarDecl *vd = n->getVarDecl();
  if (vd)
    if (vd->isAParameter) o << "ParmVar "; else o << "Var ";
  o << n->declarationName << " ";
  chillAST_FunctionDecl *fd = n->getFunctionDecl();
  if (fd) dumpVector(this, ident, fd->getParameterSymbolTable(), o);
}

void Dump::printS(std::string ident, chillAST_FloatingLiteral *n, std::ostream &o) {
  if (n->precision == 1) o << "float ";
  else o << "double ";
  o << n->value;
}

void Dump::printS(std::string ident, chillAST_ForStmt *n, std::ostream &o) {
  print(ident, n->init, o);
  print(ident, n->cond, o);
  print(ident, n->incr, o);
  print(ident, n->body, o);
}

void Dump::printS(std::string ident, chillAST_Free *n, std::ostream &o) {}

void Dump::printS(std::string ident, chillAST_FunctionDecl *n, std::ostream &o) {
  if (n->filename) o << n->filename << " ";
  if (n->isFromSourceFile) o << "FromSourceFile" << " ";
  o << n->returnType << " " << n->functionName << " ";
  if (n->getBody()) print(ident, n->getBody(), o);
}

void Dump::printS(std::string ident, chillAST_IfStmt *n, std::ostream &o) {
  print(ident, n->getCond(), o);
  print(ident, n->getThen(), o);
  if (n->getElse())
    print(ident, n->getElse(), o);
}

void Dump::printS(std::string ident, chillAST_IntegerLiteral *n, std::ostream &o) {
  o << n->value << " ";
}

void Dump::printS(std::string ident, chillAST_ImplicitCastExpr *n, std::ostream &o) {
  print(ident, n->subexpr, o);
}

void Dump::printS(std::string ident, chillAST_MacroDefinition *n, std::ostream &o) {
  o << n->macroName << " ";
  dumpVector(this, ident, &(n->parameters), o);
  print(ident, n->getBody(), o);
}

void Dump::printS(std::string ident, chillAST_Malloc *n, std::ostream &o) {
  print(ident, n->sizeexpr, o);
}

void Dump::printS(std::string ident, chillAST_MemberExpr *n, std::ostream &o) {
  print(ident, n->base, o);
  if (n->exptype == CHILL_MEMBER_EXP_ARROW) o << "-> ";
  else o << ". ";
  o << n->member << " ";
}

void Dump::printS(std::string ident, chillAST_NULL *n, std::ostream &o) {
  o << "(NULL) ";
}

void Dump::printS(std::string ident, chillAST_NoOp *n, std::ostream &o) {}

void Dump::printS(std::string ident, chillAST_ParenExpr *n, std::ostream &o) {
  print(ident, n->subexpr, o);
}

void Dump::printS(std::string ident, chillAST_Preprocessing *n, std::ostream &o) {
  o << "(PreProc " << n->pptype << " " << n->position << " " << n->blurb << " )";
}

void Dump::printS(std::string ident, chillAST_RecordDecl *n, std::ostream &o) {
  // TODO access control
  o << n->getName() << " ";
  o << n->isAStruct() << " ";
  o << n->isAUnion() << " ";
}

void Dump::printS(std::string ident, chillAST_ReturnStmt *n, std::ostream &o) {
  if (n->returnvalue) print(ident, n->returnvalue, o);
}

void Dump::printS(std::string ident, chillAST_Sizeof *n, std::ostream &o) {
  o << n->thing << " ";
}

void Dump::printS(std::string ident, chillAST_SourceFile *n, std::ostream &o) {
  dumpVector(this, ident, &(n->getChildren()), o);
}

void Dump::printS(std::string ident, chillAST_TypedefDecl *n, std::ostream &o) {
  o << n->underlyingtype << " " << n->newtype << " " << n->arraypart << " ";
}

void Dump::printS(std::string ident, chillAST_TernaryOperator *n, std::ostream &o) {
  o << n->op << " ";
  print(ident, n->condition, o);
  print(ident, n->lhs, o);
  print(ident, n->rhs, o);
}

void Dump::printS(std::string ident, chillAST_UnaryOperator *n, std::ostream &o) {
  if (n->prefix) o << "prefix ";
  else o << "postfix ";
  print(ident, n->subexpr, o);
}

void Dump::printS(std::string ident, chillAST_VarDecl *n, std::ostream &o) {
  o << "\"'" << n->vartype << "' '" << n->varname << "' '" << n->arraypart << "'\" dim " << n->numdimensions << " ";
}

