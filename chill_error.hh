#ifndef CHILL_ERROR_HH
#define CHILL_ERROR_HH

/*!
 * \file
 * \brief CHiLL runtime exceptions
 */

#include <stdexcept>

// Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch
namespace chill {
  namespace error {
    struct build : public std::runtime_error {
      build(const std::string &msg) : std::runtime_error(msg) {}
    };
  }
}


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


#define __stringify(id)                #id
#define __tostring(id)                 __stringify(id)
#define __throw(t, msg)                throw t(std::string(__FILE__ ", " __tostring(__LINE__) ", ") + __func__ + ": " + msg)
#define __throw_loop_error(msg)        __throw(loop_error, msg)
#define __throw_ir_errort(msg)         __throw(ir_error, msg)
#define __throw_ir_exp_error(msg)      __throw(ir_exp_error, msg)
#define __throw_omega_error(msg)       __throw(omega_error, msg)
#define __throw_runtime_error(msg)     __throw(std::runtime_error, msg)

#endif
