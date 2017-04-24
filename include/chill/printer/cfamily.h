//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_CFAMILY_H
#define CHILL_CFAMILY_H

#include "generic.h"

namespace chill {
  namespace printer {
    /**
     * @brief Calculate the operator precedence in C-like syntax
     *
     * All precedence calculation taken from http://en.cppreference.com/w/cpp/language/operator_precedence
     */
    class CPrec : public Scanner<int &> {
      virtual void errorRun(chillAST_node *n, int &p) {
        p = 255;
      }
      virtual void runS(chillAST_BinaryOperator *n, int &p);

      virtual void runS(chillAST_CallExpr *n, int &p);

      virtual void runS(chillAST_CStyleAddressOf *n, int &p);

      virtual void runS(chillAST_CStyleCastExpr *n, int &p);

      virtual void runS(chillAST_TernaryOperator *n, int &p);

      virtual void runS(chillAST_UnaryOperator *n, int &p);
    };
    /*!
     * \brief Print the AST in a C-like syntax.
     *
     * This replace the old print function.
     * Custom multiplexer should not be needed. This version should calculate the correct precedence for expressions.
     * Expression should be encapsulated in {} or () or ended with ; with heuristics at the parent node
     */
    class CFamily : public GenericPrinter {
    protected:

      virtual void runS(chillAST_ArraySubscriptExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_BinaryOperator *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CallExpr *n, std::string indent, std::ostream &o);

      //! Compound statement is responsible to break a new line if necessary
      virtual void runS(chillAST_CompoundStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CStyleAddressOf *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CStyleCastExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaFree *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaKernelCall *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaMalloc *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaMemcpy *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaSyncthreads *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_DeclRefExpr *n, std::string indent, std::ostream &o);

      /*!
       * Prints the floatpoint literal, only the showpoint flag is currently set
       * @param ident
       * @param n
       * @param o
       */
      virtual void runS(chillAST_FloatingLiteral *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_ForStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_WhileStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_Free *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_FunctionDecl *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_IfStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_IntegerLiteral *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_ImplicitCastExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_MacroDefinition *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_Malloc *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_MemberExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_NULL *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_NoOp *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_ParenExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_Preprocessing *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_RecordDecl *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_ReturnStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_Sizeof *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_SourceFile *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_TypedefDecl *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_TernaryOperator *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_UnaryOperator *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_VarDecl *n, std::string indent, std::ostream &o);

    public:
      CFamily() {}
      int getPrec(chillAST_node *n) {
        int p;
        CPrec cp;
        cp.run(n, p);
        return p;
      }
      //! Print the subexpression with precedence
      virtual void printPrec(chillAST_node *n, std::string indent, std::ostream &o, int prec);
    };
  }
}

#endif //CHILL_CFAMILY_H
