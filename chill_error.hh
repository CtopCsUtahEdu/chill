#ifndef CHILL_ERROR_HH
#define CHILL_ERROR_HH

/*!
 * \file
 * \brief CHiLL runtime exceptions
 */

#include <stdexcept>

//! for loop transformation problem
struct loop_error: public std::runtime_error {
  loop_error(const std::string &msg): std::runtime_error(msg){}
};

//! for generic compiler intermediate code handling problem
struct ir_error: public std::runtime_error {
  ir_error(const std::string &msg): std::runtime_error(msg){}
};

//! specific for expression to presburger math translation problem
struct ir_exp_error: public ir_error {
  ir_exp_error(const std::string &msg): ir_error(msg){}
};

struct omega_error: public std::runtime_error {
  omega_error(const std::string &msg): std::runtime_error(msg){}
};


#define __throw_at(t, f, l, msg)                throw t(std::string(#f ", " #l ": ") + msg)
#define __throw_loop_error_at(f, l, msg)        __throw_at(loop_error, f, l, msg)
#define __throw_ir_error_at(f, l, msg)          __throw_at(ir_error, f, l, msg)
#define __throw_ir_exp_error_at(f, l, msg)      __throw_at(ir_exp_error, f, l, msg)
#define __throw_omega_error_at(f, l, msg)       __throw_at(omega_error, f, l, msg)
#define __throw_runtime_error_at(f, l, msg)     __throw_at(std::runtime_error, f, l, msg)

#endif
