//
// Created by joe on 7/2/17.
//

#ifndef CHILL_ROSE_H
#define CHILL_ROSE_H

#include "parser.h"

namespace chill {
  namespace parser {
    class Rose : public Parser {
    public:
      void parse(std::string filename, std::string procname);
    };
  }
}

#endif //CHILL_ROSE_H
