//
// Created by ztuowen on 9/24/16.
//

#include "printer/generic.h"

using namespace chill::printer;

void GenericPrinter::print(std::string ident, chillAST_node *n, std::ostream &o) {
  if (!n) return;
  switch (n->getType()) {
    case CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR:
      printS(ident, dynamic_cast<chillAST_ArraySubscriptExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_BINARYOPERATOR:
      printS(ident, dynamic_cast<chillAST_BinaryOperator *>(n), o);
      break;
    case CHILLAST_NODETYPE_CALLEXPR:
      printS(ident, dynamic_cast<chillAST_CallExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_COMPOUNDSTMT:
      printS(ident, dynamic_cast<chillAST_CompoundStmt *>(n), o);
      break;
    case CHILLAST_NODETYPE_CSTYLEADDRESSOF:
      printS(ident, dynamic_cast<chillAST_CStyleAddressOf *>(n), o);
      break;
    case CHILLAST_NODETYPE_CSTYLECASTEXPR:
      printS(ident, dynamic_cast<chillAST_CStyleCastExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_CUDAFREE:
      printS(ident, dynamic_cast<chillAST_CudaFree *>(n), o);
      break;
//    case CHILLAST_NODETYPE_CUDAKERNELCALL:
//      printS(ident, dynamic_cast<chillAST_CudaKernelCall *>(n), o);
//      break;
    case CHILLAST_NODETYPE_CUDAMALLOC:
      printS(ident, dynamic_cast<chillAST_CudaMalloc *>(n), o);
      break;
    case CHILLAST_NODETYPE_CUDAMEMCPY:
      printS(ident, dynamic_cast<chillAST_CudaMemcpy *>(n), o);
      break;
    case CHILLAST_NODETYPE_CUDASYNCTHREADS:
      printS(ident, dynamic_cast<chillAST_CudaSyncthreads *>(n), o);
      break;
    case CHILLAST_NODETYPE_DECLREFEXPR:
      printS(ident, dynamic_cast<chillAST_DeclRefExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_FLOATINGLITERAL:
      printS(ident, dynamic_cast<chillAST_FloatingLiteral *>(n), o);
      break;
    case CHILLAST_NODETYPE_LOOP:
//    case CHILLAST_NODETYPE_FORSTMT:
      printS(ident, dynamic_cast<chillAST_ForStmt *>(n), o);
      break;
    case CHILLAST_NODETYPE_FREE:
      printS(ident, dynamic_cast<chillAST_Free *>(n), o);
      break;
    case CHILLAST_NODETYPE_FUNCTIONDECL:
      printS(ident, dynamic_cast<chillAST_FunctionDecl *>(n), o);
      break;
    case CHILLAST_NODETYPE_IFSTMT:
      printS(ident, dynamic_cast<chillAST_IfStmt *>(n), o);
      break;
    case CHILLAST_NODETYPE_IMPLICITCASTEXPR:
      printS(ident, dynamic_cast<chillAST_ImplicitCastExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_INTEGERLITERAL:
      printS(ident, dynamic_cast<chillAST_IntegerLiteral *>(n), o);
      break;
    case CHILLAST_NODETYPE_MACRODEFINITION:
      printS(ident, dynamic_cast<chillAST_MacroDefinition *>(n), o);
      break;
    case CHILLAST_NODETYPE_MALLOC:
      printS(ident, dynamic_cast<chillAST_Malloc *>(n), o);
      break;
    case CHILLAST_NODETYPE_MEMBEREXPR:
      printS(ident, dynamic_cast<chillAST_MemberExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_NOOP:
      printS(ident, dynamic_cast<chillAST_NoOp *>(n), o);
      break;
    case CHILLAST_NODETYPE_NULL:
      printS(ident, dynamic_cast<chillAST_NULL *>(n), o);
      break;
    case CHILLAST_NODETYPE_PARENEXPR:
      printS(ident, dynamic_cast<chillAST_ParenExpr *>(n), o);
      break;
    case CHILLAST_NODETYPE_PREPROCESSING:
      printS(ident, dynamic_cast<chillAST_Preprocessing *>(n), o);
      break;
    case CHILLAST_NODETYPE_RECORDDECL:
      printS(ident, dynamic_cast<chillAST_RecordDecl *>(n), o);
      break;
    case CHILLAST_NODETYPE_RETURNSTMT:
      printS(ident, dynamic_cast<chillAST_ReturnStmt *>(n), o);
      break;
    case CHILLAST_NODETYPE_SIZEOF:
      printS(ident, dynamic_cast<chillAST_Sizeof *>(n), o);
      break;
    case CHILLAST_NODETYPE_TRANSLATIONUNIT:
//    case CHILLAST_NODETYPE_SOURCEFILE:
      printS(ident, dynamic_cast<chillAST_SourceFile *>(n), o);
      break;
    case CHILLAST_NODETYPE_TERNARYOPERATOR:
      printS(ident, dynamic_cast<chillAST_TernaryOperator *>(n), o);
      break;
    case CHILLAST_NODETYPE_TYPEDEFDECL:
      printS(ident, dynamic_cast<chillAST_TypedefDecl *>(n), o);
      break;
    case CHILLAST_NODETYPE_UNARYOPERATOR:
      printS(ident, dynamic_cast<chillAST_UnaryOperator *>(n), o);
      break;
    case CHILLAST_NODETYPE_VARDECL:
      printS(ident, dynamic_cast<chillAST_VarDecl *>(n), o);
      break;
    case CHILLAST_NODETYPE_UNKNOWN:
    default: chill_error_printf("Printing an unknown type of Node: %s\n", n->getTypeString());
  }
  o.flush();
}

int GenericPrinter::getPrec(chillAST_node *n) {
  if (!n) return defGetPrecS();
  switch (n->getType()) {
    case CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR:
      return getPrecS(dynamic_cast<chillAST_ArraySubscriptExpr *>(n));
    case CHILLAST_NODETYPE_BINARYOPERATOR:
      return getPrecS(dynamic_cast<chillAST_BinaryOperator *>(n));
    case CHILLAST_NODETYPE_CALLEXPR:
      return getPrecS(dynamic_cast<chillAST_CallExpr *>(n));
    case CHILLAST_NODETYPE_COMPOUNDSTMT:
      return getPrecS(dynamic_cast<chillAST_CompoundStmt *>(n));
    case CHILLAST_NODETYPE_CSTYLEADDRESSOF:
      return getPrecS(dynamic_cast<chillAST_CStyleAddressOf *>(n));
    case CHILLAST_NODETYPE_CSTYLECASTEXPR:
      return getPrecS(dynamic_cast<chillAST_CStyleCastExpr *>(n));
    case CHILLAST_NODETYPE_CUDAFREE:
      return getPrecS(dynamic_cast<chillAST_CudaFree *>(n));
//    case CHILLAST_NODETYPE_CUDAKERNELCALL:
//      return getPrecS(dynamic_cast<chillAST_CudaKernelCall *>(n));
    case CHILLAST_NODETYPE_CUDAMALLOC:
      return getPrecS(dynamic_cast<chillAST_CudaMalloc *>(n));
    case CHILLAST_NODETYPE_CUDAMEMCPY:
      return getPrecS(dynamic_cast<chillAST_CudaMemcpy *>(n));
    case CHILLAST_NODETYPE_CUDASYNCTHREADS:
      return getPrecS(dynamic_cast<chillAST_CudaSyncthreads *>(n));
    case CHILLAST_NODETYPE_DECLREFEXPR:
      return getPrecS(dynamic_cast<chillAST_DeclRefExpr *>(n));
    case CHILLAST_NODETYPE_FLOATINGLITERAL:
      return getPrecS(dynamic_cast<chillAST_FloatingLiteral *>(n));
    case CHILLAST_NODETYPE_LOOP:
//    case CHILLAST_NODETYPE_FORSTMT:
      return getPrecS(dynamic_cast<chillAST_ForStmt *>(n));
    case CHILLAST_NODETYPE_FREE:
      return getPrecS(dynamic_cast<chillAST_Free *>(n));
    case CHILLAST_NODETYPE_FUNCTIONDECL:
      return getPrecS(dynamic_cast<chillAST_FunctionDecl *>(n));
    case CHILLAST_NODETYPE_IFSTMT:
      return getPrecS(dynamic_cast<chillAST_IfStmt *>(n));
    case CHILLAST_NODETYPE_IMPLICITCASTEXPR:
      return getPrecS(dynamic_cast<chillAST_ImplicitCastExpr *>(n));
    case CHILLAST_NODETYPE_INTEGERLITERAL:
      return getPrecS(dynamic_cast<chillAST_IntegerLiteral *>(n));
    case CHILLAST_NODETYPE_MACRODEFINITION:
      return getPrecS(dynamic_cast<chillAST_MacroDefinition *>(n));
    case CHILLAST_NODETYPE_MALLOC:
      return getPrecS(dynamic_cast<chillAST_Malloc *>(n));
    case CHILLAST_NODETYPE_MEMBEREXPR:
      return getPrecS(dynamic_cast<chillAST_MemberExpr *>(n));
    case CHILLAST_NODETYPE_NOOP:
      return getPrecS(dynamic_cast<chillAST_NoOp *>(n));
    case CHILLAST_NODETYPE_NULL:
      return getPrecS(dynamic_cast<chillAST_NULL *>(n));
    case CHILLAST_NODETYPE_PARENEXPR:
      return getPrecS(dynamic_cast<chillAST_ParenExpr *>(n));
    case CHILLAST_NODETYPE_PREPROCESSING:
      return getPrecS(dynamic_cast<chillAST_Preprocessing *>(n));
    case CHILLAST_NODETYPE_RECORDDECL:
      return getPrecS(dynamic_cast<chillAST_RecordDecl *>(n));
    case CHILLAST_NODETYPE_RETURNSTMT:
      return getPrecS(dynamic_cast<chillAST_ReturnStmt *>(n));
    case CHILLAST_NODETYPE_SIZEOF:
      return getPrecS(dynamic_cast<chillAST_Sizeof *>(n));
    case CHILLAST_NODETYPE_TRANSLATIONUNIT:
//    case CHILLAST_NODETYPE_SOURCEFILE:
      return getPrecS(dynamic_cast<chillAST_SourceFile *>(n));
    case CHILLAST_NODETYPE_TERNARYOPERATOR:
      return getPrecS(dynamic_cast<chillAST_TernaryOperator *>(n));
    case CHILLAST_NODETYPE_TYPEDEFDECL:
      return getPrecS(dynamic_cast<chillAST_TypedefDecl *>(n));
    case CHILLAST_NODETYPE_UNARYOPERATOR:
      return getPrecS(dynamic_cast<chillAST_UnaryOperator *>(n));
    case CHILLAST_NODETYPE_VARDECL:
      return getPrecS(dynamic_cast<chillAST_VarDecl *>(n));
    case CHILLAST_NODETYPE_UNKNOWN:
    default: chill_error_printf("Getting precedence an unknown type of Node: %s\n", n->getTypeString());
      return 255;
  }
}

void GenericPrinter::errorPrintS(std::string ident, chillAST_node *n, std::ostream &o) {
  chill_error_printf("Unhandled case in printer: %s\n", n->getTypeString());
}
