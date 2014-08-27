#ifndef DEP_HH
#define DEP_HH

#include <omega.h>
#include "graph.hh"
#include "ir_code.hh"
#include "chill_error.hh"

enum DependenceType { DEP_W2R, DEP_R2W, DEP_W2W, DEP_R2R, DEP_CONTROL, DEP_UNKNOWN };

class DependenceVector;
typedef std::vector<DependenceVector> DependenceList;

struct DependenceVector {
  DependenceType type;
  IR_Symbol *sym;
  
  bool is_reduction; // used to identify a class of flow dependence
                     // that can be broken
  std::vector<omega::coef_t> lbounds;
  std::vector<omega::coef_t> ubounds;
  
  bool quasi;
  bool is_scalar_dependence;
  DependenceVector() {
    type = DEP_UNKNOWN;
    sym = NULL;
    is_reduction = false;
    quasi = false;
    is_scalar_dependence = false;
  }
  // DependenceVector(int size);
  DependenceVector(const DependenceVector &that);
  ~DependenceVector() {delete sym;}
  DependenceVector &operator=(const DependenceVector &that);
  
  bool is_data_dependence() const;
  bool is_control_dependence() const;
  bool has_negative_been_carried_at(int dim) const;
  bool has_been_carried_at(int dim) const;
  bool has_been_carried_before(int dim) const;
  
  // the following functions will be cleaned up or removed later
  bool isZero() const;
  bool isPositive() const;
  bool isNegative() const;
  bool isAllPositive() const;
  bool isAllNegative() const;
  bool isZero(int dim) const;
  bool hasPositive(int dim) const;
  bool hasNegative(int dim) const;
  bool isCarried(int dim, omega::coef_t distance = posInfinity) const;
  bool canPermute(const std::vector<int> &pi) const;
  
  std::vector<DependenceVector> normalize() const;
  std::vector<DependenceVector> permute(const std::vector<int> &pi) const;
  DependenceVector reverse() const;
  // std::vector<DependenceVector> matrix(const std::vector<std::vector<int> > &M) const;
  DependenceType getType() const;
  friend std::ostream& operator<<(std::ostream &os, const DependenceVector &d);
};



class DependenceGraph: public Graph<Empty, DependenceVector> {
  
protected:
  int num_dim_;
  
public:
  DependenceGraph(int n) { num_dim_ = n; }
  DependenceGraph() { num_dim_ = 0; }
  ~DependenceGraph() {}
  int num_dim() const { return num_dim_; }
//   DependenceGraph permute(const std::vector<int> &pi) const;
  DependenceGraph permute(const std::vector<int> &pi,
                          const std::set<int> &active = std::set<int>()) const;
  // DependenceGraph matrix(const std::vector<std::vector<int> > &M) const;
  DependenceGraph subspace(int dim) const;
  bool isPositive() const;
  bool hasPositive(int dim) const;
  bool hasNegative(int dim) const;
};

#endif
