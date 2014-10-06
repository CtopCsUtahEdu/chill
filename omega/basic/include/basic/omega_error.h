#ifndef OMEGA_ERROR_H
#define OMEGA_ERROR_H

namespace omega {

struct presburger_error: public std::runtime_error {
  presburger_error(const std::string &msg): std::runtime_error("presburger error: " + msg) {}
};



}
#endif

