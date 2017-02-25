//
// Created by ztuowen on 2/19/17.
//

#ifndef CHILL_SANITYCHECK_H
#define CHILL_SANITYCHECK_H

#include "scanner.h"
#include <ostream>

namespace chill {
  namespace scanner {
    /**
     * @brief A sanity checker that will print diagnostics to ostream
     */
    class SanityCheck : public Scanner<std::ostream &> {
    protected:
      virtual void runS(chillAST_DeclRefExpr *n, std::ostream &o);
    };
  }
}

#endif //CHILL_SANITYCHECK_H
