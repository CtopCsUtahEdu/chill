#ifndef _CODEGEN_ERROR_H
#define _CODEGEN_ERROR_H

#include <stdexcept>

namespace omega {

struct codegen_error: public std::runtime_error {
  codegen_error(const std::string &msg): std::runtime_error("codegen error: " + msg) {}
};


}
#endif

