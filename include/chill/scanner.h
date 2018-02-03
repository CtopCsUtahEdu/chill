//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_SCANNER_H
#define CHILL_SCANNER_H

#include "chill_ast.hh"
#include <string>
#include <sstream>

namespace chill {
  /*!
   * \brief this is a generic AST scanner that walk the the AST and collect or transform the content
   */
  template<typename... Ts>
  class Scanner {
  protected:
    virtual void errorRun(chillAST_node *n, Ts... args) {
      debug_printf("Unhandled case in scanner: %s\n", n->getTypeString());
      // This is generic
      for (int i = 0; i < n->getNumChildren(); ++i)
        run(n->getChild(i), args...);
    }

    virtual void runS(chillAST_ArraySubscriptExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_BinaryOperator *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CallExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CompoundStmt *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CStyleAddressOf *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CStyleCastExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CudaFree *n, Ts... args) { errorRun(n, args...); }

//      virtual void runS(chillAST_CudaKernelCall *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CudaMalloc *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CudaMemcpy *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_CudaSyncthreads *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_DeclRefExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_FloatingLiteral *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_ForStmt *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_WhileStmt *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_Free *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_FunctionDecl *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_IfStmt *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_IntegerLiteral *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_ImplicitCastExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_MacroDefinition *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_Malloc *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_MemberExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_NULL *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_NoOp *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_ParenExpr *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_Preprocessing *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_RecordDecl *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_ReturnStmt *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_Sizeof *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_SourceFile *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_TypedefDecl *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_TernaryOperator *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_UnaryOperator *n, Ts... args) { errorRun(n, args...); }

    virtual void runS(chillAST_VarDecl *n, Ts... args) { errorRun(n, args...); }

  public:

    virtual ~Scanner() = default;

    //! Scanner entry, multiplexer
    /*!
     * @param n the chillAST_Node
     * @param o the parameters
     */
    virtual void run(chillAST_node *n, Ts... args) {
      if (!n) return;
      switch (n->getType()) {
        case CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR:
          return runS(dynamic_cast<chillAST_ArraySubscriptExpr *>(n), args...);
        case CHILLAST_NODETYPE_BINARYOPERATOR:
          return runS(dynamic_cast<chillAST_BinaryOperator *>(n), args...);
        case CHILLAST_NODETYPE_CALLEXPR:
          return runS(dynamic_cast<chillAST_CallExpr *>(n), args...);
        case CHILLAST_NODETYPE_COMPOUNDSTMT:
          return runS(dynamic_cast<chillAST_CompoundStmt *>(n), args...);
        case CHILLAST_NODETYPE_CSTYLEADDRESSOF:
          return runS(dynamic_cast<chillAST_CStyleAddressOf *>(n), args...);
        case CHILLAST_NODETYPE_CSTYLECASTEXPR:
          return runS(dynamic_cast<chillAST_CStyleCastExpr *>(n), args...);
        case CHILLAST_NODETYPE_CUDAFREE:
          return runS(dynamic_cast<chillAST_CudaFree *>(n), args...);
//      case CHILLAST_NODETYPE_CUDAKERNELCALL:
//        return runS(dynamic_cast<chillAST_CudaKernelCall *>(n), args...);
//        break;
        case CHILLAST_NODETYPE_CUDAMALLOC:
          return runS(dynamic_cast<chillAST_CudaMalloc *>(n), args...);
        case CHILLAST_NODETYPE_CUDAMEMCPY:
          return runS(dynamic_cast<chillAST_CudaMemcpy *>(n), args...);
        case CHILLAST_NODETYPE_CUDASYNCTHREADS:
          return runS(dynamic_cast<chillAST_CudaSyncthreads *>(n), args...);
        case CHILLAST_NODETYPE_DECLREFEXPR:
          return runS(dynamic_cast<chillAST_DeclRefExpr *>(n), args...);
        case CHILLAST_NODETYPE_FLOATINGLITERAL:
          return runS(dynamic_cast<chillAST_FloatingLiteral *>(n), args...);
        case CHILLAST_NODETYPE_LOOP:
//      case CHILLAST_NODETYPE_FORSTMT:
          return runS(dynamic_cast<chillAST_ForStmt *>(n), args...);
        case CHILLAST_NODETYPE_WHILESTMT:
          return runS(dynamic_cast<chillAST_WhileStmt *>(n), args...);
        case CHILLAST_NODETYPE_FREE:
          return runS(dynamic_cast<chillAST_Free *>(n), args...);
        case CHILLAST_NODETYPE_FUNCTIONDECL:
          return runS(dynamic_cast<chillAST_FunctionDecl *>(n), args...);
        case CHILLAST_NODETYPE_IFSTMT:
          return runS(dynamic_cast<chillAST_IfStmt *>(n), args...);
        case CHILLAST_NODETYPE_IMPLICITCASTEXPR:
          return runS(dynamic_cast<chillAST_ImplicitCastExpr *>(n), args...);
        case CHILLAST_NODETYPE_INTEGERLITERAL:
          return runS(dynamic_cast<chillAST_IntegerLiteral *>(n), args...);
        case CHILLAST_NODETYPE_MACRODEFINITION:
          return runS(dynamic_cast<chillAST_MacroDefinition *>(n), args...);
        case CHILLAST_NODETYPE_MALLOC:
          return runS(dynamic_cast<chillAST_Malloc *>(n), args...);
        case CHILLAST_NODETYPE_MEMBEREXPR:
          return runS(dynamic_cast<chillAST_MemberExpr *>(n), args...);
        case CHILLAST_NODETYPE_NOOP:
          return runS(dynamic_cast<chillAST_NoOp *>(n), args...);
        case CHILLAST_NODETYPE_NULL:
          return runS(dynamic_cast<chillAST_NULL *>(n), args...);
        case CHILLAST_NODETYPE_PARENEXPR:
          return runS(dynamic_cast<chillAST_ParenExpr *>(n), args...);
        case CHILLAST_NODETYPE_PREPROCESSING:
          return runS(dynamic_cast<chillAST_Preprocessing *>(n), args...);
        case CHILLAST_NODETYPE_RECORDDECL:
          return runS(dynamic_cast<chillAST_RecordDecl *>(n), args...);
        case CHILLAST_NODETYPE_RETURNSTMT:
          return runS(dynamic_cast<chillAST_ReturnStmt *>(n), args...);
        case CHILLAST_NODETYPE_SIZEOF:
          return runS(dynamic_cast<chillAST_Sizeof *>(n), args...);
        case CHILLAST_NODETYPE_TRANSLATIONUNIT:
//    case CHILLAST_NODETYPE_SOURCEFILE:
          return runS(dynamic_cast<chillAST_SourceFile *>(n), args...);
        case CHILLAST_NODETYPE_TERNARYOPERATOR:
          return runS(dynamic_cast<chillAST_TernaryOperator *>(n), args...);
        case CHILLAST_NODETYPE_TYPEDEFDECL:
          return runS(dynamic_cast<chillAST_TypedefDecl *>(n), args...);
        case CHILLAST_NODETYPE_UNARYOPERATOR:
          return runS(dynamic_cast<chillAST_UnaryOperator *>(n), args...);
        case CHILLAST_NODETYPE_VARDECL:
          return runS(dynamic_cast<chillAST_VarDecl *>(n), args...);
        case CHILLAST_NODETYPE_UNKNOWN:
        default:
          return errorRun(n, args...);
      }
    }
  };
}

#endif //CHILL_SCANNER_H
