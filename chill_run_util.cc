#include <stdio.h>
#include <string.h>
#include "chill_io.hh"
#include "chill_run_util.hh"

static std::string to_string(int ival) {
  char buffer[4];
  sprintf(buffer, "%d", ival);
  return std::string(buffer);
}

simap_vec_t* make_prog(simap_vec_t* cond) {
  return cond;
}

simap_vec_t* make_cond_gt(simap_t* lhs, simap_t* rhs) {
  simap_vec_t* nvec = new simap_vec_t();
  for(simap_t::iterator it = rhs->begin(); it != rhs->end(); it++)
    (*lhs)[it->first] -= it->second;
  (*lhs)[to_string(0)] -= 1;
  nvec->push_back(*lhs);
  delete rhs;
  delete lhs;
  return nvec;
}

simap_vec_t* make_cond_lt(simap_t* lhs, simap_t* rhs) {
  return make_cond_gt(rhs, lhs);
}

simap_vec_t* make_cond_ge(simap_t* lhs, simap_t* rhs) {
  simap_vec_t* nvec = new simap_vec_t();
  for(simap_t::iterator it = rhs->begin(); it != rhs->end(); it++)
    (*lhs)[it->first] -= it->second;
  nvec->push_back(*lhs);
  delete rhs;
  delete lhs;
  return nvec;
}

simap_vec_t* make_cond_le(simap_t* lhs, simap_t* rhs) {
  return make_cond_ge(rhs, lhs);
}

simap_vec_t* make_cond_eq(simap_t* lhs, simap_t* rhs) {
  simap_vec_t* nvec = new simap_vec_t();
  for(simap_t::iterator it = lhs->begin(); it != lhs->end(); it++)
    (*rhs)[it->first] -= it->second;
  nvec->push_back(*rhs);
  for(simap_t::iterator it = rhs->begin(); it != rhs->end(); it++)
    it->second = -it->second;
  nvec->push_back(*rhs);
  delete rhs;
  delete lhs;
  return nvec;
}

simap_t* make_cond_item_add(simap_t* lhs, simap_t* rhs) {
  for(simap_t::iterator it = lhs->begin(); it != lhs->end(); it++)
    (*rhs)[it->first] += it->second;
  delete lhs;
  return rhs;
}

simap_t* make_cond_item_sub(simap_t* lhs, simap_t* rhs) {
  for(simap_t::iterator it = rhs->begin(); it != rhs->end(); it++)
    it->second = -it->second;
  return make_cond_item_add(lhs, rhs);
}

simap_t* make_cond_item_mul(simap_t* lhs, simap_t* rhs) {
  (*lhs)[to_string(0)] += 0;
  (*rhs)[to_string(0)] += 0;
  if(rhs->size() == 1) {
    int t = (*rhs)[to_string(0)];
    for(simap_t::iterator it = lhs->begin(); it != lhs->end(); it++)
      it->second *= t;
    delete rhs;
    return lhs;
  }
  else if(rhs->size() == 1) {
    int t = (*lhs)[to_string(0)];
    for(simap_t::iterator it = rhs->begin(); it != rhs->end(); it++)
      it->second *= t;
    delete lhs;
    return rhs;
  }
  else {
    chill_error_fprintf(stderr, "require Presburger formula");
    delete lhs;
    delete rhs;
    // exit(2); <-- this may be a boost feature
  }
}

simap_t* make_cond_item_neg(simap_t* expr) {
  for (simap_t::iterator it = expr->begin(); it != expr->end(); it++) {
    it->second = -(it->second);
  }
  return expr;
}

simap_t* make_cond_item_number(int n) {
  simap_t* nmap = new simap_t();
  (*nmap)[to_string(0)] = n;
  return nmap;
}

simap_t* make_cond_item_variable(const char* var) {
  simap_t* nmap = new simap_t();
  (*nmap)[std::string(var)] = 1;
  return nmap;
}

simap_t* make_cond_item_level(int n) {
  simap_t* nmap = new simap_t();
  (*nmap)[to_string(n)] = 1;
  return nmap;
}

/*simap_t* make_cond_item_variable(const char* varname) {
  simap_t* nmap = new simap_t();
#ifdef PYTHON
  PyObject* globals = PyEval_GetGlobals();
  PyObject* itemval = PyDict_GetItemString(globals, varname);
  
#elif LUA
#endif
}*/
