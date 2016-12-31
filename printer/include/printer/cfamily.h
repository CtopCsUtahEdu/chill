//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_CFAMILY_H
#define CHILL_CFAMILY_H

#include "printer/generic.h"

namespace chill {
  namespace printer {
    /*!
     * \brief Print the AST in a C-like syntax.
     *
     * This replace the old print function.
     * Custom multiplexer should not be needed. This version should calculate the correct precedence for expressions.
     * Expression should be encapsulated in {} or () or ended with ; with heuristics at the parent node
     *
     * All precedence calculation taken from http://en.cppreference.com/w/cpp/language/operator_precedence
     */
    class CFamily : public GenericPrinter {
    protected:
      virtual int getPrecS(chillAST_BinaryOperator *n);

      virtual int getPrecS(chillAST_CallExpr *n);

      virtual int getPrecS(chillAST_CStyleAddressOf *n);

      virtual int getPrecS(chillAST_CStyleCastExpr *n);

      virtual int getPrecS(chillAST_TernaryOperator *n);

      virtual int getPrecS(chillAST_UnaryOperator *n);

      virtual void printS(std::string ident, chillAST_ArraySubscriptExpr *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_BinaryOperator *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CallExpr *n, std::ostream &o);

      //! Compound statement is responsible to break a new line if necessary
      virtual void printS(std::string ident, chillAST_CompoundStmt *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CStyleAddressOf *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CStyleCastExpr *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CudaFree *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CudaKernelCall *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CudaMalloc *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CudaMemcpy *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_CudaSyncthreads *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_DeclRefExpr *n, std::ostream &o);

      /*!
       * Prints the floatpoint literal, only the showpoint flag is currently set
       * @param ident
       * @param n
       * @param o
       */
      virtual void printS(std::string ident, chillAST_FloatingLiteral *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_ForStmt *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_Free *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_FunctionDecl *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_IfStmt *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_IntegerLiteral *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_ImplicitCastExpr *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_MacroDefinition *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_Malloc *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_MemberExpr *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_NULL *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_NoOp *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_ParenExpr *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_Preprocessing *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_RecordDecl *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_ReturnStmt *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_Sizeof *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_SourceFile *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_TypedefDecl *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_TernaryOperator *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_UnaryOperator *n, std::ostream &o);

      virtual void printS(std::string ident, chillAST_VarDecl *n, std::ostream &o);

    public:
      CFamily() {}
    };
  }
}

#endif //CHILL_CFAMILY_H
