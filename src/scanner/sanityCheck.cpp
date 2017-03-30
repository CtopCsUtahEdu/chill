//
// Created by ztuowen on 2/19/17.
//

#include "scanner/sanityCheck.h"

using namespace chill::scanner;

void SanityCheck::runS(chillAST_DeclRefExpr *n, std::ostream &o) {
  if (n->decl == NULL)
    debug_begin
      o << "DeclRef to " << n->declarationName << " of type " << n->declarationType << " is not linked" << std::endl;
    debug_end
}
