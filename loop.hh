#ifndef LOOP_HH
#define LOOP_HH


#include <omega.h>
#include <omega/code_gen/include/codegen.h>
#include <code_gen/CG.h>
#include <vector>
#include <map>
#include <set>
#include "dep.hh"
#include "ir_code.hh"
#include "irtools.hh"
#include <code_gen/CG_stringBuilder.h>

#include "stencil.hh"

/*!
 * \file
 * \brief Core loop transformation functionality.
 *
 * "level" (starting from 1) means loop level and it corresponds to "dim"
 * (starting from 0) in transformed iteration space [c_1,l_1,c_2,l_2,....,
 * c_n,l_n,c_(n+1)], e.g., l_2 is loop level 2 in generated code, dim 3
 * in transformed iteration space, and variable 4 in Omega relation.
 * All c's are constant numbers only and they will not show up as actual loops.
 *
 * Formula:
 *
 * ~~~
 *   dim = 2*level - 1
 *   var = dim + 1
 * ~~~
 */

class IR_Code;

enum TilingMethodType { StridedTile, CountedTile };
enum LoopLevelType { LoopLevelOriginal, LoopLevelTile, LoopLevelUnknown };
enum BarrierType { Barrier, P2P, DOACROSS };

//! Describes properties of each loop level of a statement.
struct LoopLevel {
  LoopLevelType type;
  /*!
   * For LoopLevelOriginal means iteration space dimension
   * For LoopLevelTile means tiled loop level. Special value -1 for
   * LoopLevelTile means purely derived loop. For dependence dimension
   * payloads, the values must be in an increasing order.
   */
  int payload;
  /*!
   * Used by code generation to support
   * multi-level parallelization (default 0 means sequential loop under
   * the current parallelization level).
   */
  int parallel_level;
  bool segreducible;
  std::string segment_descriptor;
};


struct Statement {
  omega::CG_outputRepr *code;
  omega::Relation IS;
  omega::Relation xform;
  std::vector<LoopLevel> loop_level;
  ir_tree_node *ir_stmt_node;
  bool has_inspector;
  /*!
   * @brief Whether reduction is possible
   *
   * 0 == reduction not possible, 1 == reduction possible, 2 == reduction with some processing
   */
  int reduction;
  IR_OPERATION_TYPE reductionOp; // Manu

  class stencilInfo *statementStencil;

  //protonu--temporarily putting this back here
  //omega::Tuple<int> nonSplitLevels;
  //end--protonu.
};

/*!
 * @brief Info for pragmas during code generation
 */
struct PragmaInfo {
public:

  int           stmt;
  int           loop_level;
  std::string   name;

  inline PragmaInfo(int stmt, int loop_level, std::string name) noexcept
          : stmt(stmt), loop_level(loop_level), name(name) {
      // do nothiing
  }

};

/*!
 * @brief Info for omp pragma during code generation
 */
struct OMPPragmaInfo {
public:

  int                       stmt;
  int                       loop_level;
  std::vector<std::string>  privitized_vars;
  std::vector<std::string>  shared_vars;

  inline OMPPragmaInfo(int stmt, int loop_level, const std::vector<std::string>& privitized_vars, const std::vector<std::string>& shared_vars) noexcept
          : stmt(stmt), loop_level(loop_level), privitized_vars(privitized_vars), shared_vars(shared_vars) {
      // do nothing
  }
};

class Loop {
protected:
  int tmp_loop_var_name_counter;
  static const std::string tmp_loop_var_name_prefix;
  int overflow_var_name_counter;
  static const std::string overflow_var_name_prefix;
  std::vector<int> stmt_nesting_level_; // UNDERLINE 
  std::vector<std::string> index;
  std::map<int, omega::CG_outputRepr *> replace;
  std::map<int, std::pair<int, std::string> > reduced_statements;

public:
  void debugRelations() const;
  IR_Code *ir;
  std::vector<IR_PointerSymbol *> ptr_variables;
  std::vector<omega::Free_Var_Decl*> freevar;
  std::vector<Statement> stmt;
  std::vector<omega::CG_outputRepr*> actual_code;  // ????? 
  std::vector<ir_tree_node *> ir_stmt;
  std::vector<ir_tree_node *> ir_tree;
  std::set<std::string> reduced_write_refs;
  std::map<std::string, int> array_dims;
  DependenceGraph dep;
  std::vector<omega::Relation> dep_relation; // TODO What is this for: Anand's
  int num_dep_dim;
  omega::Relation known;
  omega::CG_outputRepr *init_code;
  omega::CG_outputRepr *cleanup_code;
  std::map<int, std::vector<omega::Free_Var_Decl *> > overflow;
  std::vector<std::map<std::string, std::vector<omega::CG_outputRepr * > > > uninterpreted_symbols;
  std::vector<std::map<std::string, std::vector<omega::CG_outputRepr * > > > uninterpreted_symbols_stringrepr;

  // Need for sparse
  std::vector<std::map<std::string, std::vector<omega::Relation > > >unin_rel;
  std::map<std::string, std::set<std::string > > unin_symbol_args;
  std::map<std::string, std::string > unin_symbol_for_iegen;
  std::vector<std::pair<std::string, std::string > > dep_rel_for_iegen;

  // Need for OMP parallel regions
  std::vector<PragmaInfo>               general_pragma_info;
  std::vector<OMPPragmaInfo>            omp_pragma_info;

protected:
  mutable omega::CodeGen *last_compute_cg_;
  mutable omega::CG_result *last_compute_cgr_;
  mutable int last_compute_effort_;
  
protected:
  // Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch init_loop
  // is renamed as buildIS, actualy commented in Tuowen's branch, it is related to current way of 
  // generating iteration space that Tuowen may want to keep, so I am leaving it in there for now 
  bool init_loop(std::vector<ir_tree_node *> &ir_tree, std::vector<ir_tree_node *> &ir_stmt);
  // Mahdi: Following two functions are added for above reason
  void buildIS(std::vector<ir_tree_node*> &ir_tree,std::vector<int> &lexicalOrder,std::vector<ir_tree_node*> &ctrls, int level);
  void align_loops(std::vector<ir_tree_node*> &ir_tree, std::vector<std::string> &vars_to_be_replaced, std::vector<omega::CG_outputRepr*> &vars_replacement,int level);

  int get_dep_dim_of(int stmt, int level) const;
  int get_last_dep_dim_before(int stmt, int level) const;
  std::vector<omega::Relation> getNewIS() const;
  omega::Relation getNewIS(int stmt_num) const;
  /**
   * @brief Get the lexical order of a statment as a vector
   * @return a 2*level+1 vector with real Loop set to 0
   */
  std::vector<int> getLexicalOrder(int stmt_num) const;
  /**
   * @brief Get the lexical ordering of the statement at level
   * @param level loop level starting with 1
   * @return
   */
  int getLexicalOrder(int stmt_num, int level) const;
  std::set<int> getStatements(const std::vector<int> &lex, int dim) const;
  /**
   * @brief Shift the Lexical order of the statements
   *
   * Shift only when <dim have the same lexical order and when amount >= 0, all the statment after lex or when amount
   * <= 0, all the statement before lex
   */
  void shiftLexicalOrder(const std::vector<int> &lex, int dim, int amount);
  /**
   * @brief Assign the lexical order of statements according to dependences
   *
   * @param dim The dimension to set starting with 0
   * @param active Set of statements to set order
   * @param starting_order
   * @param idxNamesopp_
   */
  void setLexicalOrder(int dim, const std::set<int> &active, int starting_order = 0, std::vector< std::vector<std::string> >idxNames= std::vector< std::vector<std::string> >());
  void apply_xform(int stmt_num);
  void apply_xform(std::set<int> &active);
  void apply_xform();
  std::set<int> getSubLoopNest(int stmt_num, int level) const;
  int  getMinLexValue(std::set<int> stmts, int level);
  omega::Relation parseExpWithWhileToRel(omega::CG_outputRepr *repr, omega::Relation &R, int loc);
 
  //
  // OMP operations
  //

  void                  omp_apply_pragmas() const;

public:

  //
  // OMP Interface
  //

  void                  omp_mark_pragma(int, int, std::string);
  void                  omp_mark_parallel_for(int, int, const std::vector<std::string>&, const std::vector<std::string>&);

private:

  //omega::CG_outputRepr* omp_add_pragma(omega::CG_outputRepr* repr, int, int, std::string) const;
  //omega::CG_outputRepr* omp_add_omp_thread_info(omega::CG_outputRepr* repr) const;
  //omega::CG_outputRepr* omp_add_omp_for_recursive(omega::CG_outputRepr* repr, int, int, int num_threads = 0, std::vector<std::string> prv = std::vector<std::string>()) const;
  
public:
  Loop() { ir = NULL; tmp_loop_var_name_counter = 1; init_code = NULL; replaceCode_ind = 1;}
  Loop(const IR_Control *control);
  ~Loop();
  
  omega::CG_outputRepr *getCode(int effort = 3) const; // TODO was 1
  //chillAST_node* LoopCuda::getCode(int effort, std::set<int> stmts) const;

  void stencilASEPadded(int stmt_num);
  /**
   * @brief invalidate saved codegen computation
   *
   * Must be called whenever changes are made to the IS, even with auxiliary loop indices.
   */
  void invalidateCodeGen() {
    if(last_compute_cgr_ != NULL) {
      delete last_compute_cgr_;
      last_compute_cgr_ = NULL;
    }
    if(last_compute_cg_ != NULL) {
      delete last_compute_cg_;
      last_compute_cg_ = NULL;
    }
  }
  
  void printCode(int effort = 3) const;
  void addKnown(const omega::Relation &cond);
  void print_internal_loop_structure() const;
  bool isInitialized() const;
  int num_statement() const { return stmt.size(); }
  void printIterationSpace() const;
  void printDependenceGraph() const;
  /*!
   * Mahdi: This functions extarcts and returns the data dependence relations 
   * that are needed for generating inspectors for wavefront paralleization of a 
   * specific loop level
   * Loop levels start with 0 (being outer most loop), outer most loop is the default
   * Input:  loop level for parallelization
   * Output: dependence relations in teh form of strings that are in ISL (IEGenLib) syntax  
   */
  std::vector<std::pair<std::string, std::string >> 
    depRelsForParallelization(std::string privatizable_arrays, 
                              std::string reduction_operations, int parallelLoopLevel = 0);
  // Mahdi: a temporary hack for getting dependence extraction changes integrated
  // Reason: Transformed code that is suppose to be printed out when Chill finishes everything,
  //         is not correct for our examples!
  int replaceCode_ind;
  int replaceCode(){ return replaceCode_ind;}

  void removeDependence(int stmt_num_from, int stmt_num_to);
  void dump() const;
  
  std::vector<std::set <int > > sort_by_same_loops(std::set<int > active, int level);
  //! legacy unimodular transformations for perfectly nested loops
  /*!
   * e.g. \f$M*(i,j)^T = (i',j')^T or M*(i,j,1)^T = (i',j')^T\f$
   */
  bool nonsingular(const std::vector<std::vector<int> > &M);
  
  //
  // high-level loop transformations
  //
  void permute(const std::set<int> &active, const std::vector<int> &pi);
  void permute(int stmt_num, int level, const std::vector<int> &pi);
  void permute(const std::vector<int> &pi);
  // TODO doc and usage needed
  void original();
  
  void tile(int stmt_num, int level, int tile_size, int outer_level = 1, TilingMethodType method = StridedTile, int alignment_offset = 0, int alignment_multiple = 1);
  std::set<int> split(int stmt_num, int level, const omega::Relation &cond);
  std::set<int> unroll(int stmt_num, int level, int unroll_amount, std::vector< std::vector<std::string> >idxNames= std::vector< std::vector<std::string> >(), int cleanup_split_level = 0);

  //! Datacopy function by reffering arrays by numbers
  /*!
   * for example
   * ~~~
   * A[i] = A[i-1] + B[i];
   * ~~~
   * parameter array_ref_num=[0,2] means to copy data touched by A[i-1] and A[i]
   *
   * @param array_ref_nums
   * @param level
   * @param allow_extra_read
   * @param fastest_changing_dimension
   * @param padding_stride
   * @param padding_alignment
   * @param memory_type
   * @return
   */
  bool datacopy(const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums, int level, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, int memory_type = 0);
  //! Datacopy function by reffering arrays by name
  /*!
   * parameter array_name=A means to copy data touched by A[i-1] and A[i]
   * @param stmt_num
   * @param level
   * @param array_name
   * @param allow_extra_read
   * @param fastest_changing_dimension
   * @param padding_stride
   * @param padding_alignment
   * @param memory_type
   * @return
   */
  bool datacopy(int stmt_num, int level, const std::string &array_name, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, int memory_type = 0);
  bool datacopy_privatized(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, int memory_type = 0);
  bool datacopy_privatized(const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums, int level, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, int memory_type = 0);
  bool datacopy_privatized(const std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > &stmt_refs, int level, const std::vector<int> &privatized_levels, bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, int memory_type = 0);
  //std::set<int> scalar_replacement_inner(int stmt_num);
  bool find_stencil_shape( int stmt_num ); 
  
  
  Graph<std::set<int>, bool> construct_induced_graph_at_level(std::vector<std::set<int> > s, DependenceGraph dep, int dep_dim);
  std::vector<std::set<int> > typed_fusion(Graph<std::set<int>, bool> g, std::vector<bool> &types);

  void fuse(const std::set<int> &stmt_nums, int level);
  void distribute(const std::set<int> &stmt_nums, int level);
  void skew(const std::set<int> &stmt_nums, int level, const std::vector<int> &skew_amount);
  void shift(const std::set<int> &stmt_nums, int level, int shift_amount);
  void scale(const std::set<int> &stmt_nums, int level, int scale_amount);
  void reverse(const std::set<int> &stmt_nums, int level);
  void peel(int stmt_num, int level, int peel_amount = 1);
  //
  // more fancy loop transformations
  //
  void modular_shift(int stmt_num, int level, int shift_amount) {}
  void diagonal_map(int stmt_num, const std::pair<int, int> &levels, int offset) {}
  void modular_partition(int stmt_num, int level, int stride) {}
  void flatten(int stmt_num, std::string index_name, std::vector<int> &loop_levels, std::string inspector_name);
  void normalize(int stmt_num,  int loop_level);

  void generate_ghostcells_v2(std::vector<int> stmt, int loop_num, int ghost_value, int hold_inner_loop_constant=0 );
  


  //
  // derived loop transformations
  //
  void shift_to(int stmt_num, int level, int absolute_position);
  std::set<int> unroll_extra(int stmt_num, int level, int unroll_amount, int cleanup_split_level = 0);
  bool is_dependence_valid_based_on_lex_order(int i, int j,
                                              const DependenceVector &dv, bool before);
  void split_with_alignment(int stmt_num, int level, int alignment,
      int direction=0);

  // Manu:: reduction operation
  void reduce(int stmt_num, std::vector<int> &level, int param, std::string func_name, std::vector<int> &seq_levels, std::vector<int> cudaized_levels = std::vector<int>(), int bound_level = -1);
  void scalar_expand(int stmt_num, const std::vector<int> &levels, std::string arrName, int memory_type =0, int padding_alignment=0, int assign_then_accumulate = 1, int padding_stride = 0);
  void ELLify(int stmt_num, std::vector<std::string> arrays_to_pad, int pad_to, bool dense_pad = false, std::string dense_pad_pos_array = "");
  void compact(int stmt_num, int level, std::string new_array, int zero,
      std::string data_array);
  void make_dense(int stmt_num, int loop_level, std::string new_loop_index);
  void set_array_size(std::string name, int size );
  omega::CG_outputRepr * iegen_parser(std::string &str, std::vector<std::string> &index_names);

  //
  // other public operations
  //
  void pragma(int stmt_num, int level, const std::string &pragmaText);
  void prefetch(int stmt_num, int level, const std::string &arrName, int hint);
  //void prefetch(int stmt_num, int level, const std::string &arrName, const std::string &indexName, int offset, int hint);

};
#endif
