//
// Created by ztuowen on 9/24/16.
//

#include "printer/dump.h"

using namespace chill::printer;
using namespace std;

template<typename T>
void dumpVector(GenericPrinter *p, vector<T *> *n, std::string indent, std::ostream &o) {
  for (int i = 0; i < n->size(); ++i)
    p->run((*n)[i], indent, o);
}

void Dump::run(chillAST_node *n, std::string indent, std::ostream &o) {
  if (!n) return;
  o << "(" << n->getTypeString() << " ";
  if (n->getSymbolTable()) {
    o << "(VarScope: ";
    dumpVector(this, n->getSymbolTable(), indent, o);
    o << ") ";
  }
  o << ": ";
  // Recurse
  GenericPrinter::run(n, indent, o);
  o << ") ";
}

void Dump::runS(chillAST_ArraySubscriptExpr *n, std::string indent, std::ostream &o) {
  if (n->multibase())
    o << "(" << n->multibase()->varname << ") ";
  if (n->multibase() && n->multibase()->vartype)
    o << n->multibase()->vartype;
  if (n->imwrittento) {
    if (n->imreadfrom)
      o << "lvalue AND rvalue ";
    else
      o << "lvalue ";
  } else o << "rvalue ";
  run(n->base, indent, o);
  run(n->index, indent, o);
}

void Dump::runS(chillAST_BinaryOperator *n, std::string indent, std::ostream &o) {
  o << n->op << " ";
  if (n->getLHS()) run(n->getLHS(), indent, o);
  else o << "(NULL) ";
  if (n->getRHS()) run(n->getRHS(), indent, o);
  else o << "(NULL) ";
}

void Dump::runS(chillAST_CallExpr *n, std::string indent, std::ostream &o) {
  dumpVector(this, &(n->getChildren()), indent, o);
}

void Dump::runS(chillAST_CompoundStmt *n, std::string indent, std::ostream &o) {
  dumpVector(this, &(n->getChildren()), indent, o);
}

void Dump::runS(chillAST_CStyleAddressOf *n, std::string indent, std::ostream &o) {
  run(n->subexpr, indent, o);
}

void Dump::runS(chillAST_CStyleCastExpr *n, std::string indent, std::ostream &o) {
  o << n->towhat << " ";
  run(n->subexpr, indent, o);
}

void Dump::runS(chillAST_CudaFree *n, std::string indent, std::ostream &o) {
  run(n->variable, indent, o);
}

void Dump::runS(chillAST_CudaKernelCall *n, std::string indent, std::ostream &o) {
  chill_error_printf("Not implemented");
}

void Dump::runS(chillAST_CudaMalloc *n, std::string indent, std::ostream &o) {
  run(n->devPtr, indent, o);
  run(n->sizeinbytes, indent, o);
}

void Dump::runS(chillAST_CudaMemcpy *n, std::string indent, std::ostream &o) {
  o << n->cudaMemcpyKind << " ";
  run(n->dest, indent, o);
  run(n->src, indent, o);
  run(n->size, indent, o);
}

void Dump::runS(chillAST_CudaSyncthreads *n, std::string indent, std::ostream &o) {}

void Dump::runS(chillAST_DeclRefExpr *n, std::string indent, std::ostream &o) {
  chillAST_VarDecl *vd = n->getVarDecl();
  if (vd)
    if (vd->isAParameter) o << "ParmVar "; else o << "Var ";
  o << n->declarationName << " ";
  chillAST_FunctionDecl *fd = n->getFunctionDecl();
  if (fd) dumpVector(this, fd->getParameterSymbolTable(), indent, o);
}

void Dump::runS(chillAST_FloatingLiteral *n, std::string indent, std::ostream &o) {
  if (n->precision == 1) o << "float ";
  else o << "double ";
  o << n->value;
}

void Dump::runS(chillAST_ForStmt *n, std::string indent, std::ostream &o) {
  run(n->init, indent, o);
  run(n->cond, indent, o);
  run(n->incr, indent, o);
  run(n->body, indent, o);
}

void Dump::runS(chillAST_WhileStmt *n, std::string indent, std::ostream &o) {
  run(n->cond, indent, o);
  run(n->body, indent, o);
}

void Dump::runS(chillAST_Free *n, std::string indent, std::ostream &o) {}

void Dump::runS(chillAST_FunctionDecl *n, std::string indent, std::ostream &o) {
  if (n->filename) o << n->filename << " ";
  if (n->isFromSourceFile) o << "FromSourceFile" << " ";
  o << n->returnType << " " << n->functionName << " ";
  if (n->getBody()) run(n->getBody(), indent, o);
}

void Dump::runS(chillAST_IfStmt *n, std::string indent, std::ostream &o) {
  run(n->getCond(), indent, o);
  run(n->getThen(), indent, o);
  if (n->getElse())
    run(n->getElse(), indent, o);
}

void Dump::runS(chillAST_IntegerLiteral *n, std::string indent, std::ostream &o) {
  o << n->value << " ";
}

void Dump::runS(chillAST_ImplicitCastExpr *n, std::string indent, std::ostream &o) {
  run(n->subexpr, indent, o);
}

void Dump::runS(chillAST_MacroDefinition *n, std::string indent, std::ostream &o) {
  o << n->macroName << " ";
  dumpVector(this, &(n->parameters), indent, o);
  run(n->getBody(), indent, o);
}

void Dump::runS(chillAST_Malloc *n, std::string indent, std::ostream &o) {
  run(n->sizeexpr, indent, o);
}

void Dump::runS(chillAST_MemberExpr *n, std::string indent, std::ostream &o) {
  run(n->base, indent, o);
  if (n->exptype == CHILL_MEMBER_EXP_ARROW) o << "-> ";
  else o << ". ";
  o << n->member << " ";
}

void Dump::runS(chillAST_NULL *n, std::string indent, std::ostream &o) {
  o << "(NULL) ";
}

void Dump::runS(chillAST_NoOp *n, std::string indent, std::ostream &o) {}

void Dump::runS(chillAST_ParenExpr *n, std::string indent, std::ostream &o) {
  run(n->subexpr, indent, o);
}

void Dump::runS(chillAST_Preprocessing *n, std::string indent, std::ostream &o) {
  o << "(PreProc " << n->pptype << " " << n->position << " " << n->blurb << " )";
}

void Dump::runS(chillAST_RecordDecl *n, std::string indent, std::ostream &o) {
  // TODO access control
  o << n->getName() << " ";
  o << n->isAStruct() << " ";
  o << n->isAUnion() << " ";
}

void Dump::runS(chillAST_ReturnStmt *n, std::string indent, std::ostream &o) {
  if (n->returnvalue) run(n->returnvalue, indent, o);
}

void Dump::runS(chillAST_Sizeof *n, std::string indent, std::ostream &o) {
  o << n->thing << " ";
}

void Dump::runS(chillAST_SourceFile *n, std::string indent, std::ostream &o) {
  dumpVector(this, &(n->getChildren()), indent, o);
}

void Dump::runS(chillAST_TypedefDecl *n, std::string indent, std::ostream &o) {
  o << n->underlyingtype << " " << n->newtype << " " << n->arraypart << " ";
}

void Dump::runS(chillAST_TernaryOperator *n, std::string indent, std::ostream &o) {
  o << n->op << " ";
  run(n->condition, indent, o);
  run(n->lhs, indent, o);
  run(n->rhs, indent, o);
}

void Dump::runS(chillAST_UnaryOperator *n, std::string indent, std::ostream &o) {
  if (n->prefix) o << "prefix ";
  else o << "postfix ";
  run(n->subexpr, indent, o);
}

void Dump::runS(chillAST_VarDecl *n, std::string indent, std::ostream &o) {
  o << "\"'" << n->vartype << " " << n->arraypointerpart <<"' '" << n->varname << "' '" <<std::flush;
  dumpVector(this, &(n->getChildren()), indent, o);
  o << "'\" dim " << n->numdimensions << " ";
}

