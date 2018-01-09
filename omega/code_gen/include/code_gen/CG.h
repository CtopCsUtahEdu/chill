#ifndef _CG_H
#define _CG_H

#include <omega/Relation.h>
#include <basic/boolset.h>
#include <code_gen/CG_outputBuilder.h>
#include <set>
#include <vector>

namespace omega {

class CodeGen;

/**
 * @brief Tree-like structure holding the iteration space
 */
struct CG_result {
  CodeGen *codegen_; //!< Reference to the codegen
  BoolSet<> active_; //!< Active set of statements

  CG_result() { codegen_ = NULL; }
  virtual ~CG_result() { /* not responsible for codegen_ */ }

  //! break down the complete iteration space condition to levels of bound/guard condtions
  virtual CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction) = 0;
  /**
   * @brief calculate each loop's nesting depth
   * Used in liftOverhead - depth start with 0 at leaf
   */
  virtual int populateDepth() = 0;
  //! redistribute guard condition locations by additional splittings
  virtual std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up) = 0;
  /**
   * @brief Hoist guard conditions for non-loop levels
   * Enables proper if-condition simplication when outputting actual code.
   */
  virtual Relation hoistGuard() = 0;
  virtual void removeGuard(const Relation &guard) = 0;

  //! Signature for printRepr of actual node types
  virtual CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString = false) const = 0;
  //! Main entry point for codegen
  CG_outputRepr *printRepr(CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, std::vector<std::map<std::string,std::vector<CG_outputRepr *> > >uninterpreted_symbols, bool printString = false) const;
  //! Using stringBuilder to generate loop representation
  std::string printString(std::vector<std::map<std::string, std::vector<CG_outputRepr *> > >uninterpreted_symbols = std::vector<std::map<std::string, std::vector<CG_outputRepr *> > >()) const;

  //! Total number of levels
  int num_level() const;
  //! A deep clone of the tree
  virtual CG_result *clone() const = 0;
  //! Dump content for debug information
  virtual void dump(int indent) const {}

  //! Add pragma info prior to code generation
  virtual void addPragma(int stmt, int loop_level, std::string name) = 0;
  //! Add omp pragma info prior to code generation
  virtual void addOmpPragma(int stmt, int loop_level, const std::vector<std::string>&, const std::vector<std::string>&) = 0;

  // These methods are for parallelization support
  virtual void collectIterationVariableNames(std::set<std::string>&) noexcept = 0;
  // TODO: read & write arrays - private
};

/**
 * @brief Statement sequence
 */
struct CG_split: public CG_result {
  std::vector<Relation> restrictions_;  //!< Restriction on each of the splits
  std::vector<CG_result *> clauses_;    //!< Sequence of splits on this level

  CG_split(CodeGen *codegen, const BoolSet<> &active, const std::vector<Relation> &restrictions, const std::vector<CG_result *> &clauses) {
    codegen_ = codegen;
    active_ = active;
    restrictions_ = restrictions;
    clauses_ = clauses;
  } 
  ~CG_split() {
    for (int i = 0; i < clauses_.size(); i++)
      delete clauses_[i];
  } 
  
  CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction);
  int populateDepth();
  std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up);
  Relation hoistGuard();
  void removeGuard(const Relation &guard);
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString=false) const;

  CG_result *clone() const;
  void dump(int indent) const;

  void addPragma(int stmt, int loop_level, std::string name);
  void addOmpPragma(int stnt, int loop_level, const std::vector<std::string>&, const std::vector<std::string>&);

  virtual void collectIterationVariableNames(std::set<std::string>&) noexcept;

private:
  std::vector<CG_result *> findNextLevel() const;
};

/**
 * @brief Loop
 */
struct CG_loop: public CG_result {
  int level_;               //!< Current level in the iteration space (1-based)
  CG_result *body_;         //!< Body node

  Relation known_;          //!< What is known globally/from parents
  Relation restriction_;    //!< Restriction based on split
  Relation bounds_;         //!< Iteration bounds
  Relation guard_;          //!< Conditions other than bounds
  bool needLoop_;
  int depth_;               //!< Current depth of loop - start with 0 at leaf(max)

  bool attachPragma_;       //!< Apply pragma to a loop
  std::string pragmaName_;  //!< Pragma text

  CG_loop(CodeGen *codegen, const BoolSet<> &active, int level, CG_result *body) {
    codegen_ = codegen;
    active_ = active;
    level_ = level;
    body_ = body;

    needLoop_     = false;
    depth_        = 0;
    attachPragma_ = false;
  }
  ~CG_loop() { delete body_; }
  
  CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction);
  int populateDepth();
  std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up);
  Relation hoistGuard();
  void removeGuard(const Relation &guard);
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,std::vector<std::map<std::string, std::vector<CG_outputRepr *> > >unin, bool printString = false) const;
  //! True implementation of printRepr to control of whether guard relation is printed
  CG_outputRepr *printRepr(bool do_print_guard, int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString = false) const;

  CG_result *clone() const;
  void dump(int indent) const;

  void addPragma(int stmt, int loop_level, std::string name);
  void addOmpPragma(int stnt, int loop_level, const std::vector<std::string>&, const std::vector<std::string>&);

  virtual void collectIterationVariableNames(std::set<std::string>&) noexcept;

};


/**
 * @brief Leaf - a basic code block
 */
struct CG_leaf: public CG_result {
  Relation known_;  //!< Global known/parents
  std::map<int, Relation> guards_;  //!< Guard relations for each active statements
  
  CG_leaf(CodeGen *codegen, const BoolSet<> &active) {
    codegen_ = codegen;
    active_ = active;
  }
  ~CG_leaf() {}
  
  CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction);
  int populateDepth() { return 0; }
  std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up);
  Relation hoistGuard();
  void removeGuard(const Relation &guard);
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly, std::vector<std::map<std::string, std::vector<CG_outputRepr *> > > unin, bool printString = false) const;

  CG_result *clone() const;
  void dump(int indent) const;

  void addPragma(int stmt, int loop_level, std::string name);
  void addOmpPragma(int stnt, int loop_level, const std::vector<std::string>&, const std::vector<std::string>&);

  virtual void collectIterationVariableNames(std::set<std::string>&) noexcept;

};

}

#endif
