//
// Created by joe on 7/2/17.
//

#ifndef CHILL_CLANG_H
#define CHILL_CLANG_H

#include "parser.h"

namespace chill {
  namespace parser {
    class Clang : public Parser {
    public:
      void parse(std::string filename, std::string procname);
    };
  }
}

#endif //CHILL_CLANG_H
