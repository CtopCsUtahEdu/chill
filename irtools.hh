#ifndef IRTOOLS_HH
#define IRTOOLS_HH

#include <vector>
#include <omega.h>
#include <code_gen/CG_outputRepr.h>
#include "ir_code.hh"
#include "dep.hh"
#define DEP_DEBUG 0

// IR tree is used to initialize a loop. For a loop node, payload is
// its mapped iteration space dimension. For a simple block node,
// payload is its mapped statement number. Normal if-else is split
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
  std::vector<std::string> index, int i, int j, 
  std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols,
  std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols_stringrepr);

std::vector<omega::CG_outputRepr *> collect_loop_inductive_and_conditionals(ir_tree_node * stmt_node);

// Manu
typedef std::map<int, std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > > tempResultMap;
typedef std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > DVPair;

// Manu:: this function is required for reduction operation
//omega::CG_outputRepr * from_same_statement(IR_Code *ir, IR_ArrayRef *a, IR_ArrayRef *b);
bool from_same_statement(IR_Code *ir, IR_ArrayRef *a, IR_ArrayRef *b);
int stmtType(IR_Code *ir, const omega::CG_outputRepr *repr);
IR_OPERATION_TYPE getReductionOperator(IR_Code *ir, const omega::CG_outputRepr *repr);
void mapRefstoStatements(IR_Code *ir,std::vector<IR_ArrayRef *> access, int ref2Stmt[], std::map<int,std::set<int> >& rMap, std::set<int>& tnrStmts, std::set<int>& nrStmts);
void checkReductionDependence(int i, int j, int nestLeveli, omega::coef_t lbound[], omega::coef_t ubound[], int ref2Stmt[], std::map<int,std::set<int> >& rMap, DVPair& dv, tempResultMap& trMap, std::set<int> nrStmts );

#endif
