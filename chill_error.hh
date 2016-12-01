#ifndef CHILL_ERROR_HH
#define CHILL_ERROR_HH

/*!
 * \file
 * \brief CHiLL runtime exceptions
 */

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


#endif
