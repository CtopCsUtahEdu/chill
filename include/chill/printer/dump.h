//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_DUMP_H
#define CHILL_DUMP_H

#include "generic.h"

namespace chill {
  namespace printer {
    /*!
     * \brief Dump the whole AST in Prefix format
     *
     * This replace the old dump function in the chillAST.
     * Everthing is written in a Tree-like structure: (<NodeName> <Params>). No precedence calculation is needed.
     */
    class Dump : public GenericPrinter {
    protected:
      virtual void runS(chillAST_ArraySubscriptExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_BinaryOperator *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CallExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CompoundStmt *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CStyleAddressOf *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CStyleCastExpr *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaFree *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaKernelCall *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaMalloc *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaMemcpy *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_CudaSyncthreads *n, std::string indent, std::ostream &o);

      virtual void runS(chillAST_DeclRefExpr *n, std::string indent, std::ostream &o);

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
      Dump() {}
      virtual ~Dump() = default;

      /*!
       * Just prints everything. Indent is igored due to need to limit the number of output
       * @param n
       * @param o
       */
      virtual void run(chillAST_node *n, std::string indent, std::ostream &o);
    };
  }
}

#endif //CHILL_DUMP_H
