#ifndef _CG_H
#define _CG_H

#include <omega/Relation.h>
#include <basic/boolset.h>
#include <code_gen/CG_outputBuilder.h>
#include <vector>

namespace omega {

class CodeGen;

struct CG_result {
  CodeGen *codegen_;
  BoolSet<> active_;

  CG_result() { codegen_ = NULL; }
  virtual ~CG_result() { /* not responsible for codegen_ */ }
  
  virtual CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction) = 0;
  virtual int populateDepth() = 0;
  virtual std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up) = 0;
  virtual Relation hoistGuard() = 0;
  virtual void removeGuard(const Relation &guard) = 0;
  virtual CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const = 0;
  CG_outputRepr *printRepr(CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts) const;
  std::string printString() const;
  int num_level() const;
  virtual CG_result *clone() const = 0;
  virtual void dump(int indent) const {}
  void dump() { dump(0); }
};


struct CG_split: public CG_result {
  std::vector<Relation> restrictions_;
  std::vector<CG_result *> clauses_;

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
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const;
  CG_result *clone() const;
  void dump(int indent) const;

private:
  std::vector<CG_result *> findNextLevel() const;
};


struct CG_loop: public CG_result {
  int level_;
  CG_result *body_;

  Relation known_;
  Relation restriction_;
  Relation bounds_;
  Relation guard_;
  bool needLoop_;
  int depth_;

  CG_loop(CodeGen *codegen, const BoolSet<> &active, int level, CG_result *body) {
    codegen_ = codegen;
    active_ = active;
    level_ = level;
    body_ = body;
  }
  ~CG_loop() { delete body_; }
  
  CG_result *recompute(const BoolSet<> &parent_active, const Relation &known, const Relation &restriction);
  int populateDepth();
  std::pair<CG_result *, Relation> liftOverhead(int depth, bool propagate_up);
  Relation hoistGuard();
  void removeGuard(const Relation &guard);
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const;
  CG_outputRepr *printRepr(bool do_print_guard, int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const;
  CG_result *clone() const;
  void dump(int indent) const;
};



struct CG_leaf: public CG_result {
  Relation known_;
  std::map<int, Relation> guards_;
  
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
  CG_outputRepr *printRepr(int indent, CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts, const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const;
  CG_result *clone() const;
  void dump(int indent) const;
};

}

#endif
