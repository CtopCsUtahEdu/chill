#ifndef IRTOOLS_HH
#define IRTOOLS_HH

#include <vector>
#include <omega.h>
#include <code_gen/CG_outputRepr.h>
#include "ir_code.hh"
#include "dep.hh"
#define DEP_DEBUG 0

/*!
 * \file
 * \brief Useful tools to analyze code in compiler IR format.
 */

//! IR tree is used to initialize a loop.
struct ir_tree_node {
  IR_Control *content;
  ir_tree_node *parent;
  std::vector<ir_tree_node *> children;
/*!
 * * For a loop node, payload is its mapped iteration space dimension.
 * * For a simple block node, payload is its mapped statement number.
 * * Normal if-else is split into two nodes where
 *   * odd payload represents then-part
 *   * even payload represents else-part.
 */
  int payload;
  
  ~ir_tree_node() {
    for (int i = 0; i < children.size(); i++)
      delete children[i];
    delete content;
  }
};

class Loop; // Forward definition for test_data_dependences
/*!
 * @brief Build IR tree from the source code
 *
 * Block type node can only be leaf, i.e., there is no further stuctures inside a block allowed
 *
 * @param control
 * @param parent
 * @return
 */
std::vector<ir_tree_node *> build_ir_tree(IR_Control *control,
                                          ir_tree_node *parent = NULL);
/*!
 * @brief Extract statements from IR tree
 *
 * Statements returned are ordered in lexical order in the source code
 *
 * @param ir_tree
 * @return
 */
std::vector<ir_tree_node *> extract_ir_stmts(
  const std::vector<ir_tree_node *> &ir_tree);

bool is_dependence_valid(ir_tree_node *src_node, ir_tree_node *dst_node,
                         const DependenceVector &dv, bool before);
/*!
 * @brief test data dependeces between two statements
 *
 * The first statement in parameter must be lexically before the second statement in parameter.
 * Returned dependences are all lexicographically positive
 *
 * @param loop
 * @param ir
 * @param repr1
 * @param IS1
 * @param repr2
 * @param IS2
 * @param freevar
 * @param index
 * @param nestLeveli
 * @param nestLevelj
 * @param uninterpreted_symbols
 * @param uninterpreted_symbols_stringrepr
 * @param unin_rel
 * @param dep_relation
 * @return
 */
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > test_data_dependences(Loop *loop,
                                                                                               IR_Code *ir, const omega::CG_outputRepr *repr1, const omega::Relation &IS1,
                                                                                               const omega::CG_outputRepr *repr2, const omega::Relation &IS2,
                                                                                               std::vector<omega::Free_Var_Decl*> &freevar, std::vector<std::string> index,
                                                                                               int nestLeveli, int nestLevelj, std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols,
                                                                                               std::map<std::string, std::vector<omega::CG_outputRepr * > > &uninterpreted_symbols_stringrepr,
                                                                                               std::map<std::string, std::vector<omega::Relation > > &unin_rel,
                                                                                               std::vector<omega::Relation> &dep_relation);

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
