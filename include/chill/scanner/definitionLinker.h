//
// Created by ztuowen on 2/19/17.
//

#ifndef CHILL_DEFINITIONLINKER_H
#define CHILL_DEFINITIONLINKER_H

#include "scanner.h"

namespace chill {
  namespace scanner {
    /**
     * @brief A definition linker that will link missing references(in-tree)
     */
    class DefinitionLinker : public Scanner<void *> {
    protected:
      //! Add type definition to scope
      virtual void runS(chillAST_TypedefDecl *n, void *o);

      //! Add type definition to scope
      virtual void runS(chillAST_RecordDecl *n, void *o);

      //! Extract function parameters add function decl
      virtual void runS(chillAST_FunctionDecl *n, void *o);

      //! Scoping the decls inside
      virtual void runS(chillAST_CompoundStmt *n, void *o);

      //! Adding one variable decl
      virtual void runS(chillAST_VarDecl *n, void *o);

      //! Finding the referenced decl
      virtual void runS(chillAST_DeclRefExpr *n, void *o);

    public:
      /**
       * @brief Main entry point
       * @param n the node and children to link definitions
       */
      void exec(chillAST_node *n);
    };
  }
}

#endif //CHILL_DEFINITIONLINKER_H
