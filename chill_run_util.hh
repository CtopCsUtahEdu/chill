#ifndef CHILL_RUN_UTIL_HH
#define CHILL_RUN_UTIL_HH

#include <vector>
#include <map>
#include <string>

typedef std::map<std::string, int>               simap_t;
typedef std::vector<std::map<std::string, int> > simap_vec_t;

simap_vec_t* make_prog(simap_vec_t* cond);
simap_vec_t* make_cond_gt(simap_t* lhs, simap_t* rhs);
simap_vec_t* make_cond_lt(simap_t* lhs, simap_t* rhs);
simap_vec_t* make_cond_ge(simap_t* lhs, simap_t* rhs);
simap_vec_t* make_cond_le(simap_t* lhs, simap_t* rhs);
simap_vec_t* make_cond_eq(simap_t* lhs, simap_t* rhs);
simap_t* make_cond_item_add(simap_t* lhs, simap_t* rhs);
simap_t* make_cond_item_sub(simap_t* lhs, simap_t* rhs);
simap_t* make_cond_item_mul(simap_t* lhs, simap_t* rhs);
simap_t* make_cond_item_neg(simap_t* expr);
simap_t* make_cond_item_number(int n);
simap_t* make_cond_item_variable(const char* var);
simap_t* make_cond_item_level(int n);
simap_vec_t* parse_relation_vector(const char* expr);

#endif
