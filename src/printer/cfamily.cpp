//
// Created by ztuowen on 9/24/16.
//

#include "printer/cfamily.h"
#include "printer/data.h"
#include <iomanip>
#include <limits>


using namespace std;
using namespace chill::printer;

void
printPreProcPOS(GenericPrinter *p, vector<chillAST_Preprocessing *> &n, CHILL_PREPROCESSING_POSITION pos,
                std::string indent, std::ostream &o) {
  for (int i = 0; i < n.size(); ++i)
    if (n[i]->position == pos)
      p->run(n[i], indent, o);
}

bool opInSet(const char *set, char *op) {
  string tmp = op;
  tmp = " " + tmp + " ";
  return strstr(set, tmp.c_str()) != NULL;
}

bool ifSemicolonFree(CHILL_ASTNODE_TYPE t) {
  return t == CHILLAST_NODETYPE_FUNCTIONDECL || t == CHILLAST_NODETYPE_IFSTMT ||
         t == CHILLAST_NODETYPE_FORSTMT || t == CHILLAST_NODETYPE_MACRODEFINITION ||
         t == CHILLAST_NODETYPE_COMPOUNDSTMT;
}

const char *binaryPrec[] = {
    " :: ",
    " . -> ",
    "",
    " .* ->* ",
    " * / % ",
    " + - ",
    " << >> ",
    " < <= > >= ",
    " == != ",
    " & ",
    " ^ ",
    " | ",
    " && ",
    " || ",
    " = += -= *= /= %= <<= >>= &= ^= |= ",
    " , "
};

void CPrec::runS(chillAST_BinaryOperator *n, int &p) {
  errorRun(n, p);
  for (int i = 0; i < 16; ++i)
    if (opInSet(binaryPrec[i], n->op)) {
      p += i + 1;
      return;
    }
  chill_error_printf("Unrecognized binary operator: %s\n", n->op);
}

void CPrec::runS(chillAST_CallExpr *n, int &p) {
  errorRun(n, p);
  p += 2;
}

void CPrec::runS(chillAST_CStyleAddressOf *n, int &p) {
  errorRun(n, p);
  p += 3;
}

void CPrec::runS(chillAST_CStyleCastExpr *n, int &p) {
  errorRun(n, p);
  p += 3;
}

void CPrec::runS(chillAST_TernaryOperator *n, int &p) {
  errorRun(n, p);
  p += 15;
}

const char *unaryPrec[] = {
    "",
    " -- ++ ",
    " -- ++ + - ! ~ * & ",
};

void CPrec::runS(chillAST_UnaryOperator *n, int &p) {
  errorRun(n, p);
  if (n->prefix) {
    for (int i = 2; i >= 0; --i)
      if (opInSet(unaryPrec[i], n->op)) {
        p += i + 1;
        return;
      }
  } else
    for (int i = 1; i < 3; ++i)
      if (opInSet(unaryPrec[i], n->op)) {
        p += i + 1;
        return;
      }
}

void CFamily::runS(chillAST_ArraySubscriptExpr *n, std::string indent, std::ostream &o) {
  run(n->base, indent, o);
  o << "[";
  run(n->index, indent, o);
  o << "]";
}

void CFamily::runS(chillAST_BinaryOperator *n, std::string indent, std::ostream &o) {
  int prec = getPrec(n);
  if (n->getLHS()) printPrec(n->getLHS(), indent, o, prec);
  else o << "(NULL)";
  o << " " << n->op << " ";
  if (n->getRHS()) printPrec(n->getRHS(), indent, o, prec - 1);
  else o << "(NULL)";
}

void CFamily::runS(chillAST_CallExpr *n, std::string indent, std::ostream &o) {
  chillAST_FunctionDecl *FD = NULL;
  chillAST_MacroDefinition *MD = NULL;
  if (n->callee->isDeclRefExpr()) {
    chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *) (n->callee);
    if (DRE->decl)
      if (DRE->decl->isFunctionDecl()) FD = (chillAST_FunctionDecl *) (DRE->decl);
      else
        chill_error_printf("Function DRE of type %s\n", DRE->decl->getTypeString());
  } else if (n->callee->isFunctionDecl())
    FD = (chillAST_FunctionDecl *) n->callee;
  else if (n->callee->isMacroDefinition())
    MD = (chillAST_MacroDefinition *) n->callee;
  run(n->callee, indent, o);
  if (MD && n->getNumChildren() - 1)
    o << "(";
  else {
    if (n->grid && n->block)
      o << "<<<" << n->grid->varname << "," << n->block->varname << ">>>";
    o << "(";
  }
  for (int i = 0; i < n->args.size(); ++i) {
    if (i != 0) o << ", ";
    run(n->args.at(i), indent, o);
  }
  if (!MD || n->getNumChildren() - 1)
    o << ")";
}

void CFamily::runS(chillAST_CompoundStmt *n, std::string indent, std::ostream &o) {
  vector<chillAST_node *> *c = &(n->getChildren());
  string nid = indent + identSpace;
  if (!n->getParent() || c->size() > 1 || n->getParent()->isFunctionDecl() || n->getParent()->isCompoundStmt())
    o << "{";
  for (int i = 0; i < c->size(); ++i) {
    o << "\n" << nid;
    printPreProcPOS(this, c->at(i)->preprocessinginfo, CHILL_PREPROCESSING_LINEBEFORE, nid, o);
    printPreProcPOS(this, c->at(i)->preprocessinginfo, CHILL_PREPROCESSING_IMMEDIATELYBEFORE, nid, o);
    print(c->at(i), nid, o);
    if (!ifSemicolonFree(c->at(i)->getType())) o << ";";
    printPreProcPOS(this, c->at(i)->preprocessinginfo, CHILL_PREPROCESSING_TOTHERIGHT, nid, o);
    printPreProcPOS(this, c->at(i)->preprocessinginfo, CHILL_PREPROCESSING_LINEAFTER, nid, o);
  }
  if (!n->getParent() || c->size() > 1 || n->getParent()->isFunctionDecl() || n->getParent()->isCompoundStmt())
    o << "\n" << indent << "}";
}


void CFamily::runS(chillAST_CStyleAddressOf *n, std::string indent, std::ostream &o) {
  o << "&";
  printPrec(n->subexpr, indent, o, getPrec(n));
}


void CFamily::runS(chillAST_CStyleCastExpr *n, std::string indent, std::ostream &o) {
  o << "(" << n->towhat << ")";
  printPrec(n->subexpr, indent, o, getPrec(n));
}

void CFamily::runS(chillAST_CudaFree *n, std::string indent, std::ostream &o) {
  o << "cudaFree(";
  run(new chillAST_DeclRefExpr(n->variable), indent, o);
  o << ")";
}

void CFamily::runS(chillAST_CudaKernelCall *n, std::string indent, std::ostream &o) {
  chill_error_printf("Not implemented");
}

void CFamily::runS(chillAST_CudaMalloc *n, std::string indent, std::ostream &o) {
  o << "cudaMalloc(";
  run(n->devPtr, indent, o);
  o << ", ";
  run(n->sizeinbytes, indent, o);
  o << ")";
}

void CFamily::runS(chillAST_CudaMemcpy *n, std::string indent, std::ostream &o) {
  o << "cudaMemcpy(";
  run(new chillAST_DeclRefExpr(n->dest), indent, o);
  o << ", ";
  run(new chillAST_DeclRefExpr(n->src), indent, o);
  o << ", ";
  run(n->size, indent, o);
  o << ", " << n->cudaMemcpyKind << ")";
}

void CFamily::runS(chillAST_CudaSyncthreads *n, std::string indent, std::ostream &o) {
  o << "__syncthreads()";
}

void CFamily::runS(chillAST_DeclRefExpr *n, std::string indent, std::ostream &o) {
  o << n->declarationName;
}

void CFamily::runS(chillAST_FloatingLiteral *n, std::string indent, std::ostream &o) {
  if (n->allthedigits)
    o << n->allthedigits;
  else {
    // C++11 only constants See http://en.cppreference.com/w/cpp/types/numeric_limits/max_digits10
    if (n->getPrecision() == 2)
      o << setprecision(std::numeric_limits<double>::max_digits10) << n->value;
    else {
      ostringstream st;
      st << setprecision(std::numeric_limits<float>::max_digits10) << n->value;
      o << st.str();
      if (st.str().find('.') == string::npos)
        o << ".0";
      o << "f";
    }
  }
}

void CFamily::runS(chillAST_ForStmt *n, std::string indent, std::ostream &o) {
  if (n->metacomment)
    o << "// " << n->metacomment << "\n" << indent;
  if (n->pragma)
    o << "#pragma " << n->pragma << "\n" << indent;
  o << "for (";
  run(n->getInit(), indent, o);
  o << "; ";
  run(n->getCond(), indent, o);
  o << "; ";
  run(n->getInc(), indent, o);
  o << ") ";
  if (n->getBody()->isCompoundStmt()) {
    run(n->getBody(), indent, o);
  } else {
    chill_error_printf("Body of for loop not COMPOUNDSTMT\n");
    run(n->getBody(), indent, o);
  }
}

void CFamily::runS(chillAST_WhileStmt *n, std::string indent, std::ostream &o) {
  o << "while (";
  run(n->cond, indent, o);
  o << ") ";
  if (n->body->isCompoundStmt()) {
    run(n->body, indent, o);
  } else {
    chill_error_printf("Body of while loop not COMPOUNDSTMT\n");
    run(n->body, indent, o);
  }
}

void CFamily::runS(chillAST_Free *n, std::string indent, std::ostream &o) {}

void CFamily::runS(chillAST_FunctionDecl *n, std::string indent, std::ostream &o) {
  if (n->isExtern()) o << "extern ";
  if (n->isFunctionGPU()) o << "__global__ ";
  o << n->returnType << " " << n->functionName << "(";

  chillAST_SymbolTable *pars = &(n->parameters);
  for (int i = 0; i < pars->size(); ++i) {
    if (i != 0)
      o << ", ";
    run(pars->at(i), indent, o);
  }
  o << ")";
  if (!(n->isExtern() || n->isForward())) {
    o << " ";
    if (n->getBody())
      run(n->getBody(), indent, o);
    else {
      chill_error_printf("Non-extern or forward function decl doesn't have a body");
      o << "{}";
    }
  } else {
    o << ";";
  }
}

void CFamily::runS(chillAST_IfStmt *n, std::string indent, std::ostream &o) {
  o << "if (";
  run(n->getCond(), indent, o);
  o << ") ";
  if (!n->getThen()) {
    chill_error_printf("If statement is without then part!");
    exit(-1);
  }
  run(n->getThen(), indent, o);
  if (!(n->getThen()->isCompoundStmt())) chill_error_printf("Then part is not a CompoundStmt!\n");
  if (n->getElse()) {
    if (n->getThen()->getNumChildren() == 1)
      o << std::endl << indent;
    else o << " ";
    o << "else ";
    run(n->getElse(), indent, o);
  }
}

void CFamily::runS(chillAST_IntegerLiteral *n, std::string indent, std::ostream &o) {
  o << n->value;
}

void CFamily::runS(chillAST_ImplicitCastExpr *n, std::string indent, std::ostream &o) {
  run(n->subexpr, indent, o);
}

void CFamily::runS(chillAST_MacroDefinition *n, std::string indent, std::ostream &o) {
  o << "#define " << n->macroName;
  int np = n->parameters.size();
  if (np) {
    o << "(" << n->parameters.at(0)->varname;
    for (int i = 1; i < np; ++i)
      o << ", " << n->parameters.at(i)->varname;
    o << ")";
  }
  // TODO newline for multiline macro
  o << " ";
  run(n->getBody(), indent, o);
}

void CFamily::runS(chillAST_Malloc *n, std::string indent, std::ostream &o) {
  o << "malloc(";
  run(n->sizeexpr, indent, o);
  o << ")";
}

void CFamily::runS(chillAST_MemberExpr *n, std::string indent, std::ostream &o) {
  int prec = getPrec(n);
  if (n->base) printPrec(n->base, indent, o, prec);
  else o << "(NULL)";
  if (n->exptype == CHILL_MEMBER_EXP_ARROW) o << "->";
  else o << ".";
  if (n->member) o << n->member;
  else o << "(NULL)";
}

void CFamily::runS(chillAST_NULL *n, std::string indent, std::ostream &o) {
  o << "/* (NULL statement) */";
}

void CFamily::runS(chillAST_NoOp *n, std::string indent, std::ostream &o) {}

void CFamily::runS(chillAST_ParenExpr *n, std::string indent, std::ostream &o) {
  o << "(";
  run(n->subexpr, indent, o);
  o << ")";
}

void CFamily::runS(chillAST_Preprocessing *n, std::string indent, std::ostream &o) {
  switch (n->position) {
    // This will need to setup for the next one
    case CHILL_PREPROCESSING_LINEBEFORE:
      o << n->blurb << "\n" << indent;
      break;
      // This will setup for itself
    case CHILL_PREPROCESSING_LINEAFTER:
      o << "\n" << indent << n->blurb << "\n" << indent;
      break;
      // These will not(reside in the same line)
    case CHILL_PREPROCESSING_IMMEDIATELYBEFORE:
    case CHILL_PREPROCESSING_TOTHERIGHT:
      o << n->blurb;
      break;
    default:
      break;
  }
}

void CFamily::runS(chillAST_RecordDecl *n, std::string indent, std::ostream &o) {
  if (n->isAStruct()) {
    string nid = indent + identSpace;
    o << "struct ";
    if (strncmp(n->getName(), "unnamed", 7)) o << n->getName() << " ";
    o << "{";
    chillAST_SymbolTable *sp = &(n->getSubparts());
    for (int i = 0; i < sp->size(); ++i) {
      o << "\n" << nid;
      print(sp->at(i), nid, o);
      o << ";";
    }
    o << "\n" << indent << "}";
  } else {
    chill_error_printf("Encountered Unkown record type");
    exit(-1);
  }
}

void CFamily::runS(chillAST_ReturnStmt *n, std::string indent, std::ostream &o) {
  o << "return";
  if (n->returnvalue) {
    o << " ";
    run(n->returnvalue, indent, o);
  }
}

void CFamily::runS(chillAST_Sizeof *n, std::string indent, std::ostream &o) {
  o << "sizeof(" << n->thing << ")";
}

void CFamily::runS(chillAST_SourceFile *n, std::string indent, std::ostream &o) {
  o << "// this source is derived from CHILL AST originally from file '"
    << n->SourceFileName << "' as parsed by frontend compiler " << n->frontend << "\n\n";
  int nchild = n->getChildren().size();
  for (int i = 0; i < nchild; ++i) {
    if (n->getChild(i)->isFromSourceFile) {
      o << indent;
      printPreProcPOS(this, n->getChild(i)->preprocessinginfo, CHILL_PREPROCESSING_LINEBEFORE, indent, o);
      printPreProcPOS(this, n->getChild(i)->preprocessinginfo, CHILL_PREPROCESSING_IMMEDIATELYBEFORE, indent, o);
      run(n->getChild(i), indent, o);
      if (!ifSemicolonFree(n->getChild(i)->getType())) o << ";";
      printPreProcPOS(this, n->getChild(i)->preprocessinginfo, CHILL_PREPROCESSING_TOTHERIGHT, indent, o);
      printPreProcPOS(this, n->getChild(i)->preprocessinginfo, CHILL_PREPROCESSING_LINEAFTER, indent, o);
      o << "\n";
    }
  }
}

void CFamily::runS(chillAST_TypedefDecl *n, std::string indent, std::ostream &o) {
  if (n->isAStruct())
    o << "/* A typedef STRUCT */\n";
  o << indent << "typedef ";
  if (n->rd) run(n->rd, indent, o);
  else if (n->isAStruct()) {
    o << "/* no rd */ ";
    // the struct subparts
  } else
    o << "/* Not A STRUCT */ " << n->getUnderlyingType() << " " << n->newtype << " " << n->arraypart;
  o << n->newtype;
}


void CFamily::runS(chillAST_TernaryOperator *n, std::string indent, std::ostream &o) {
  int prec = getPrec(n);
  printPrec(n->getCond(), indent, o, prec);
  o << " " << n->op << " ";
  printPrec(n->getLHS(), indent, o, prec);
  o << " : ";
  printPrec(n->getRHS(), indent, o, prec);
}

void CFamily::runS(chillAST_UnaryOperator *n, std::string indent, std::ostream &o) {
  int prec = getPrec(n);
  if (n->prefix) o << n->op;
  printPrec(n->subexpr, indent, o, prec);
  if (!n->prefix) o << n->op;
}

void CFamily::runS(chillAST_VarDecl *n, std::string indent, std::ostream &o) {
  if (n->isDevice) o << "__device__ ";
  if (n->isShared) o << "__shared__ ";
  //if (n->isRestrict) o << "__restrict__ ";

  if ((!n->isAParameter) && n->isAStruct() && n->vardef) {
    run(n->vardef, indent, o);
    o << " " << n->varname;
  }

  if (n->typedefinition && n->typedefinition->isAStruct())
    o << "struct ";
  o << n->vartype << " ";
  if (n->arraypointerpart)
    o << n->arraypointerpart;
  if (n->byreference)
    o << "&";

  if (n->isRestrict) o << " __restrict__ ";

  string def = n->varname;
  bool paren = false;
  for (int i = 0; i < (n->getNumChildren()); ++i) {
    if (n->getChild(i)->isNull()) {
      def = "*" + def;
      paren = true;
    } else {
      if (paren)
        def = "(" + def + ")";
      paren = false;
      def += "[" + print(n->getChild(i), indent) + "]";
    }
  }
  o << def;

  if (n->init) {
    o << " = ";
    run(n->init, indent, o);
  }
}

void CFamily::printPrec(chillAST_node *n, std::string indent, std::ostream &o, int prec) {
  if (getPrec(n) > prec) o << "(";
  run(n, indent, o);
  if (getPrec(n) > prec) o << ")";
}
