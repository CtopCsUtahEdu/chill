//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_PRINTER_H_H
#define CHILL_PRINTER_H_H

#include "chill_ast.hh"
#include <string>
#include <sstream>

namespace chill {
  /*!
   * \brief this is a generic AST scanner that walk the the AST and collect or transform the content
   */
  template<typename OBJ>
  class Scanner {
  protected:
    virtual void errorRun(chillAST_node *n, OBJ p) {
      debug_printf("Unhandled case in scanner: %s\n", n->getTypeString());
      // This is generic
      for (int i = 0; i < n->getNumChildren(); ++i)
        run(n->getChild(i), p);
    }

    virtual void runS(chillAST_ArraySubscriptExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_BinaryOperator *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CallExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CompoundStmt *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CStyleAddressOf *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CStyleCastExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CudaFree *n, OBJ o) { errorRun(n, o); }

//      virtual void runS(chillAST_CudaKernelCall *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CudaMalloc *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CudaMemcpy *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_CudaSyncthreads *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_DeclRefExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_FloatingLiteral *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_ForStmt *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_Free *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_FunctionDecl *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_IfStmt *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_IntegerLiteral *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_ImplicitCastExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_MacroDefinition *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_Malloc *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_MemberExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_NULL *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_NoOp *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_ParenExpr *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_Preprocessing *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_RecordDecl *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_ReturnStmt *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_Sizeof *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_SourceFile *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_TypedefDecl *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_TernaryOperator *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_UnaryOperator *n, OBJ o) { errorRun(n, o); }

    virtual void runS(chillAST_VarDecl *n, OBJ o) { errorRun(n, o); }

  public:
    //! Scanner entry, multiplexer
    /*!
     * @param n the chillAST_Node
     * @param o the parameters
     */
    virtual void run(chillAST_node *n, OBJ o) {
      if (!n) return;
      switch (n->getType()) {
        case CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR:
          return runS(dynamic_cast<chillAST_ArraySubscriptExpr *>(n), o);
        case CHILLAST_NODETYPE_BINARYOPERATOR:
          return runS(dynamic_cast<chillAST_BinaryOperator *>(n), o);
        case CHILLAST_NODETYPE_CALLEXPR:
          return runS(dynamic_cast<chillAST_CallExpr *>(n), o);
        case CHILLAST_NODETYPE_COMPOUNDSTMT:
          return runS(dynamic_cast<chillAST_CompoundStmt *>(n), o);
        case CHILLAST_NODETYPE_CSTYLEADDRESSOF:
          return runS(dynamic_cast<chillAST_CStyleAddressOf *>(n), o);
        case CHILLAST_NODETYPE_CSTYLECASTEXPR:
          return runS(dynamic_cast<chillAST_CStyleCastExpr *>(n), o);
        case CHILLAST_NODETYPE_CUDAFREE:
          return runS(dynamic_cast<chillAST_CudaFree *>(n), o);
//      case CHILLAST_NODETYPE_CUDAKERNELCALL:
//        return runS(dynamic_cast<chillAST_CudaKernelCall *>(n), o);
//        break;
        case CHILLAST_NODETYPE_CUDAMALLOC:
          return runS(dynamic_cast<chillAST_CudaMalloc *>(n), o);
        case CHILLAST_NODETYPE_CUDAMEMCPY:
          return runS(dynamic_cast<chillAST_CudaMemcpy *>(n), o);
        case CHILLAST_NODETYPE_CUDASYNCTHREADS:
          return runS(dynamic_cast<chillAST_CudaSyncthreads *>(n), o);
        case CHILLAST_NODETYPE_DECLREFEXPR:
          return runS(dynamic_cast<chillAST_DeclRefExpr *>(n), o);
        case CHILLAST_NODETYPE_FLOATINGLITERAL:
          return runS(dynamic_cast<chillAST_FloatingLiteral *>(n), o);
        case CHILLAST_NODETYPE_LOOP:
//      case CHILLAST_NODETYPE_FORSTMT:
          return runS(dynamic_cast<chillAST_ForStmt *>(n), o);
        case CHILLAST_NODETYPE_FREE:
          return runS(dynamic_cast<chillAST_Free *>(n), o);
        case CHILLAST_NODETYPE_FUNCTIONDECL:
          return runS(dynamic_cast<chillAST_FunctionDecl *>(n), o);
        case CHILLAST_NODETYPE_IFSTMT:
          return runS(dynamic_cast<chillAST_IfStmt *>(n), o);
        case CHILLAST_NODETYPE_IMPLICITCASTEXPR:
          return runS(dynamic_cast<chillAST_ImplicitCastExpr *>(n), o);
        case CHILLAST_NODETYPE_INTEGERLITERAL:
          return runS(dynamic_cast<chillAST_IntegerLiteral *>(n), o);
        case CHILLAST_NODETYPE_MACRODEFINITION:
          return runS(dynamic_cast<chillAST_MacroDefinition *>(n), o);
        case CHILLAST_NODETYPE_MALLOC:
          return runS(dynamic_cast<chillAST_Malloc *>(n), o);
        case CHILLAST_NODETYPE_MEMBEREXPR:
          return runS(dynamic_cast<chillAST_MemberExpr *>(n), o);
        case CHILLAST_NODETYPE_NOOP:
          return runS(dynamic_cast<chillAST_NoOp *>(n), o);
        case CHILLAST_NODETYPE_NULL:
          return runS(dynamic_cast<chillAST_NULL *>(n), o);
        case CHILLAST_NODETYPE_PARENEXPR:
          return runS(dynamic_cast<chillAST_ParenExpr *>(n), o);
        case CHILLAST_NODETYPE_PREPROCESSING:
          return runS(dynamic_cast<chillAST_Preprocessing *>(n), o);
        case CHILLAST_NODETYPE_RECORDDECL:
          return runS(dynamic_cast<chillAST_RecordDecl *>(n), o);
        case CHILLAST_NODETYPE_RETURNSTMT:
          return runS(dynamic_cast<chillAST_ReturnStmt *>(n), o);
        case CHILLAST_NODETYPE_SIZEOF:
          return runS(dynamic_cast<chillAST_Sizeof *>(n), o);
        case CHILLAST_NODETYPE_TRANSLATIONUNIT:
//    case CHILLAST_NODETYPE_SOURCEFILE:
          return runS(dynamic_cast<chillAST_SourceFile *>(n), o);
        case CHILLAST_NODETYPE_TERNARYOPERATOR:
          return runS(dynamic_cast<chillAST_TernaryOperator *>(n), o);
        case CHILLAST_NODETYPE_TYPEDEFDECL:
          return runS(dynamic_cast<chillAST_TypedefDecl *>(n), o);
        case CHILLAST_NODETYPE_UNARYOPERATOR:
          return runS(dynamic_cast<chillAST_UnaryOperator *>(n), o);
        case CHILLAST_NODETYPE_VARDECL:
          return runS(dynamic_cast<chillAST_VarDecl *>(n), o);
        case CHILLAST_NODETYPE_UNKNOWN:
        default:
          return errorRun(n, o);
      }
    }
  };
}

#endif //CHILL_PRINTER_H_H
