#ifndef IR_SUIF_UTILS_HH
#define IR_SUIF_UTILS_HH
#include <vector>
#include <suif1.h>
// #include "cctools.hh"
#include "omegatools.hh"
// #include "loop.hh"

// c++ stuff:

// template <class T> const T& min(const T &a, const T &b) {
//   if ( a < b)
//     return a;
//   else
//     return b;
// }

// template <class T> T abs(const T &v) {
//   if (v < static_cast<T>(0))
//     return -v;
//   else
//     return v;
// }

// class CG_suifArray: public CG_inputArray {
// protected:
//   in_array *ia;
// public:
//   CG_suifArray(in_array *ia_);
//   virtual bool is_write();
// };


// class SUIF_IR {
// public:
//   file_set_entry *_fse;
//   proc_sym *_psym;
//   SUIF_IR(char *filename, int proc_num);
//   ~SUIF_IR();

//   tree_for *get_loop(int loop_num);
//   void commit(Loop *lp, int loop_num);
// };

// extern SUIF_IR *ir;

// suif stuff:

// tree_for *init_loop(char *filename, int proc_num, int loop_num);
// void finalize_loop();


operand find_array_index(in_array *ia, int n, int dim, bool is_fortran);
bool is_null_statement(tree_node *tn);
std::vector<tree_for *> find_deepest_loops(tree_node *tn);
std::vector<tree_for *> find_deepest_loops(tree_node_list *tnl);
std::vector<tree_for *> find_loops(tree_node_list *tnl);
std::vector<tree_for*> find_outer_loops(tree_node *tn);
std::vector<tree_for *> find_common_loops(tree_node *tn1, tree_node *tn2);
LexicalOrderType lexical_order(tree_node *tn1, tree_node *tn2);
std::vector<in_array *> find_arrays(instruction *ins);
std::vector<in_array *> find_arrays(tree_node_list *tnl);

//protonu--adding a few functions used it cuda-chil
//these are defined in ir_cuda_suif_uitls.cc
tree_node_list* loop_body_at_level(tree_node_list* tnl, int level);
tree_node_list* loop_body_at_level(tree_for* loop, int level);
tree_node_list* swap_node_for_node_list(tree_node* tn, tree_node_list* new_tnl);
// std::vector<CG_suifArray *> find_arrays_access(instruction *ins);
// std::vector<CG_suifArray *> find_arrays_access(tree_node_list *tnl);

#endif
