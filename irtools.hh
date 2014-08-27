#ifndef IRTOOLS_HH
#define IRTOOLS_HH

#include <vector>
#include <omega.h>
#include <code_gen/CG_outputRepr.h>
#include "ir_code.hh"
#include "dep.hh"

// IR tree is used to initialize a loop. For a loop node, payload is
// its mapped iteration space dimension. For a simple block node,
// payload is its mapped statement number. Normal if-else is splitted
// into two nodes where the one with odd payload represents then-part and
// the one with even payload represents else-part.
struct ir_tree_node {
  IR_Control *content;
  ir_tree_node *parent;
  std::vector<ir_tree_node *> children;
  int payload;
  
  ~ir_tree_node() {
    for (int i = 0; i < children.size(); i++)
      delete children[i];
    delete content;
  }
};

std::vector<ir_tree_node *> build_ir_tree(IR_Control *control,
                                          ir_tree_node *parent = NULL);
std::vector<ir_tree_node *> extract_ir_stmts(
  const std::vector<ir_tree_node *> &ir_tree);
bool is_dependence_valid(ir_tree_node *src_node, ir_tree_node *dst_node,
                         const DependenceVector &dv, bool before);
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > test_data_dependences(
  IR_Code *ir, const omega::CG_outputRepr *repr1,
  const omega::Relation &IS1, const omega::CG_outputRepr *repr2,
  const omega::Relation &IS2, std::vector<omega::Free_Var_Decl*> &freevar,
  std::vector<std::string> index, int i, int j);

#endif
