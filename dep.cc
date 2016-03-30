/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
 Data dependence vector and graph.

 Notes:
 All dependence vectors are normalized, i.e., the first non-zero distance
 must be positve. Thus the correct dependence meaning can be given based on
 source/destination pair's read/write type. Suppose for a dependence vector
 1, 0~5, -3), we want to permute the first and the second dimension,
 the result would be two dependence vectors (0, 1, -3) and (1~5, 1, -3).
 All operations on dependence vectors are non-destructive, i.e., new
 dependence vectors are returned.

 History:
 01/2006 Created by Chun Chen.
 03/2009 Use IR_Ref interface in source and destination arrays -chun
*****************************************************************************/

#include "dep.hh"

//-----------------------------------------------------------------------------
// Class: DependeceVector
//-----------------------------------------------------------------------------

std::ostream& operator<<(std::ostream &os, const DependenceVector &d) {
  if (d.sym != NULL) {
    os << d.sym->name();
    os << ':';
    if (d.quasi)
      os << "_quasi";
    
  }
  
  switch (d.type) {
  case DEP_W2R:
    os << "flow";
    // Check for reduction implemetation correctness
    if (d.is_reduction)
      os << "_reduction";
    break;
  case DEP_R2W:
    os << "anti";
    // TODO: Remove Check for reduction implemetation correctness
    if (d.is_reduction)
      os << "_reduction";
    break;
  case DEP_W2W:
    os << "output";
    // TODO: Remove Check for reduction implemetation correctness
    if (d.is_reduction)
      os << "_reduction";
    break;
  case DEP_R2R:
    os << "input";
    break;
  case DEP_CONTROL:
    os << "control";
    break;
  default:
    os << "unknown";
    break;
  }
  
  os << '(';
  
  for (int i = 0; i < d.lbounds.size(); i++) {
    omega::coef_t lbound = d.lbounds[i];
    omega::coef_t ubound = d.ubounds[i];
    
    if (lbound == ubound)
      os << lbound;
    else {
      if (lbound == -posInfinity)
        if (ubound == posInfinity)
          os << '*';
        else {
          if (ubound == -1)
            os << '-';
          else
            os << ubound << '-';
        }
      else if (ubound == posInfinity) {
        if (lbound == 1)
          os << '+';
        else
          os << lbound << '+';
      } else
        os << lbound << '~' << ubound;
    }
    
    if (i < d.lbounds.size() - 1)
      os << ", ";
  }
  
  os << ')';
  
  return os;
}

// DependenceVector::DependenceVector(int size):
//   lbounds(std::vector<coef_t>(size, 0)),
//   ubounds(std::vector<coef_t>(size, 0)) {
//   src = NULL;
//   dst = NULL;
// }

DependenceVector::DependenceVector(const DependenceVector &that) {
  if (that.sym != NULL)
    this->sym = that.sym->clone();
  else
    this->sym = NULL;
  this->type = that.type;
  this->lbounds = that.lbounds;
  this->ubounds = that.ubounds;
  quasi = that.quasi;
  is_scalar_dependence = that.is_scalar_dependence;
  is_reduction = that.is_reduction;
  is_reduction_cand = that.is_reduction_cand; // Manu
}

DependenceVector &DependenceVector::operator=(const DependenceVector &that) {
  if (this != &that) {
    delete this->sym;
    if (that.sym != NULL)
      this->sym = that.sym->clone();
    else
      this->sym = NULL;
    this->type = that.type;
    this->lbounds = that.lbounds;
    this->ubounds = that.ubounds;
    quasi = that.quasi;
    is_scalar_dependence = that.is_scalar_dependence;
    is_reduction = that.is_reduction;
    is_reduction_cand = that.is_reduction_cand;
  }
  return *this;
}
DependenceType DependenceVector::getType() const {
  return type;
}

bool DependenceVector::is_data_dependence() const {
  if (type == DEP_W2R || type == DEP_R2W || type == DEP_W2W
      || type == DEP_R2R)
    return true;
  else
    return false;
}

bool DependenceVector::is_control_dependence() const {
  if (type == DEP_CONTROL)
    return true;
  else
    return false;
}

bool DependenceVector::has_negative_been_carried_at(int dim) const {
  if (!is_data_dependence())
    throw std::invalid_argument("only works for data dependences");
  
  if (dim < 0 || dim >= lbounds.size())
    return false;
  
  for (int i = 0; i < dim; i++)
    if (lbounds[i] > 0 || ubounds[i] < 0)
      return false;
  
  if (lbounds[dim] < 0)
    return true;
  else
    return false;
}


bool DependenceVector::has_been_carried_at(int dim) const {
  if (!is_data_dependence())
    throw std::invalid_argument("only works for data dependences");
  
  if (dim < 0 || dim >= lbounds.size())
    return false;
  
  for (int i = 0; i < dim; i++)
    if (lbounds[i] > 0 || ubounds[i] < 0)
      return false;
  
  if ((lbounds[dim] != 0)  || (ubounds[dim] !=0))
    return true;
  
  return false;
}

bool DependenceVector::has_been_carried_before(int dim) const {
  if (!is_data_dependence())
    throw std::invalid_argument("only works for data dependences");
  
  if (dim < 0)
    return false;
  if (dim > lbounds.size())
    dim = lbounds.size();
  
  for (int i = 0; i < dim; i++) {
    if (lbounds[i] > 0)
      return true;
    if (ubounds[i] < 0)
      return true;
  }
  
  return false;
}

bool DependenceVector::isZero() const {
  return isZero(lbounds.size() - 1);
}

bool DependenceVector::isZero(int dim) const {
  if (dim >= lbounds.size())
    throw std::invalid_argument("invalid dependence dimension");
  
  for (int i = 0; i <= dim; i++)
    if (lbounds[i] != 0 || ubounds[i] != 0)
      return false;
  
  return true;
}

bool DependenceVector::isPositive() const {
  for (int i = 0; i < lbounds.size(); i++)
    if (lbounds[i] != 0 || ubounds[i] != 0) {
      if (lbounds[i] < 0)
        return false;
      else if (lbounds[i] > 0)
        return true;
    }
  
  return false;
}

bool DependenceVector::isNegative() const {
  for (int i = 0; i < lbounds.size(); i++)
    if (lbounds[i] != 0 || ubounds[i] != 0) {
      if (ubounds[i] > 0)
        return false;
      else if (ubounds[i] < 0)
        return true;
    }
  
  return false;
}

bool DependenceVector::isAllPositive() const {
  for (int i = 0; i < lbounds.size(); i++)
    if (lbounds[i] < 0)
      return false;
  
  return true;
}

bool DependenceVector::isAllNegative() const {
  for (int i = 0; i < ubounds.size(); i++)
    if (ubounds[i] > 0)
      return false;
  
  return true;
}

bool DependenceVector::hasPositive(int dim) const {
  if (dim >= lbounds.size())
    throw std::invalid_argument("invalid dependence dimension");
  
  if (lbounds[dim] > 0)
    //av: changed from ubounds to lbounds may have side effects
    return true;
  else
    return false;
}

bool DependenceVector::hasNegative(int dim) const {
  if (dim >= lbounds.size())
    throw std::invalid_argument("invalid dependence dimension");
  
  if (ubounds[dim] < 0)
    //av: changed from lbounds to ubounds may have side effects
    return true;
  else
    return false;
}

bool DependenceVector::isCarried(int dim, omega::coef_t distance) const {
  if (distance <= 0)
    throw std::invalid_argument("invalid dependence distance size");
  
  if (dim > lbounds.size())
    dim = lbounds.size();
  
  for (int i = 0; i < dim; i++)
    if (lbounds[i] > 0)
      return false;
    else if (ubounds[i] < 0)
      return false;
  
  if (dim >= lbounds.size())
    return true;
  
  if (lbounds[dim] > distance)
    return false;
  else if (ubounds[dim] < -distance)
    return false;
  
  return true;
}

bool DependenceVector::canPermute(const std::vector<int> &pi) const {
  if (pi.size() != lbounds.size())
    throw std::invalid_argument(
                                "permute dimensionality do not match dependence space");
  
  for (int i = 0; i < pi.size(); i++) {
    if (lbounds[pi[i]] > 0)
      return true;
    else if (lbounds[pi[i]] < 0)
      return false;
  }
  
  return true;
}

std::vector<DependenceVector> DependenceVector::normalize() const {
  std::vector<DependenceVector> result;
  
  DependenceVector dv(*this);
  for (int i = 0; i < dv.lbounds.size(); i++) {
    if (dv.lbounds[i] < 0 && dv.ubounds[i] >= 0) {
      omega::coef_t t = dv.ubounds[i];
      dv.ubounds[i] = -1;
      result.push_back(dv);
      dv.lbounds[i] = 0;
      dv.ubounds[i] = t;
    }
    if (dv.lbounds[i] == 0 && dv.ubounds[i] > 0) {
      dv.lbounds[i] = 1;
      result.push_back(dv);
      dv.lbounds[i] = 0;
      dv.ubounds[i] = 0;
    }
    if (dv.lbounds[i] == 0 && dv.ubounds[i] == 0)
      continue;
    else
      break;
  }
  
  result.push_back(dv);
  return result;
}

std::vector<DependenceVector> DependenceVector::permute(
                                                        const std::vector<int> &pi) const {
  if (pi.size() != lbounds.size())
    throw std::invalid_argument(
                                "permute dimensionality do not match dependence space");
  
  const int n = lbounds.size();
  
  DependenceVector dv(*this);
  for (int i = 0; i < n; i++) {
    dv.lbounds[i] = lbounds[pi[i]];
    dv.ubounds[i] = ubounds[pi[i]];
  }
  
  int violated = 0;
  
  for (int i = 0; i < n; i++) {
    if (dv.lbounds[i] > 0)
      break;
    else if (dv.lbounds[i] < 0)
      violated = 1;
  }
  
  if (((violated == 1) && !quasi) && !is_scalar_dependence) {
    throw ir_error("dependence violation");
    
  }
  
  return dv.normalize();
}

DependenceVector DependenceVector::reverse() const {
  const int n = lbounds.size();
  
  DependenceVector dv(*this);
  switch (type) {
  case DEP_W2R:
    dv.type = DEP_R2W;
    break;
  case DEP_R2W:
    dv.type = DEP_W2R;
    break;
  default:
    dv.type = type;
  }
  
  for (int i = 0; i < n; i++) {
    dv.lbounds[i] = -ubounds[i];
    dv.ubounds[i] = -lbounds[i];
  }
  dv.quasi = true;
  
  return dv;
}

// std::vector<DependenceVector> DependenceVector::matrix(const std::vector<std::vector<int> > &M) const {
//   if (M.size() != lbounds.size())
//     throw std::invalid_argument("(non)unimodular transformation dimensionality does not match dependence space");

//   const int n = lbounds.size();
//   DependenceVector dv;
//   if (sym != NULL)
//     dv.sym = sym->clone();
//   else
//     dv.sym = NULL;
//   dv.type = type;

//   for (int i = 0; i < n; i++) {
//     assert(M[i].size() == n+1 || M[i].size() == n);

//     omega::coef_t lb, ub;
//     if (M[i].size() == n+1)
//       lb = ub = M[i][n];
//     else
//       lb = ub = 0;

//     for (int j = 0; j < n; j++) {
//       int c = M[i][j];
//       if (c == 0)
//         continue;

//       if (c > 0) {
//         if (lbounds[j] == -posInfinity)
//           lb = -posInfinity;
//         else if (lb != -posInfinity)
//           lb += c * lbounds[j];
//         if (ubounds[j] == posInfinity)
//           ub = posInfinity;
//         else if (ub != posInfinity)
//           ub += c * ubounds[j];
//       }
//       else {
//         if (ubounds[j] == posInfinity)
//           lb = -posInfinity;
//         else if (lb != -posInfinity)
//           lb += c * ubounds[j];
//         if (lbounds[j] == -posInfinity)
//           ub = posInfinity;
//         else if (ub != posInfinity)
//           ub += c * lbounds[j];
//       }
//     }
//     dv.lbounds.push_back(lb);
//     dv.ubounds.push_back(ub);
//   }
//   dv.is_reduction = is_reduction;

//   return dv.normalize();
// }

//-----------------------------------------------------------------------------
// Class: DependenceGraph
//-----------------------------------------------------------------------------

DependenceGraph DependenceGraph::permute(const std::vector<int> &pi,
                                         const std::set<int> &active) const {
  DependenceGraph g;
  
  for (int i = 0; i < vertex.size(); i++)
    g.insert(vertex[i].first);
  
  for (int i = 0; i < vertex.size(); i++)
    for (EdgeList::const_iterator j = vertex[i].second.begin();
         j != vertex[i].second.end(); j++) {
      if (active.empty()
          || (active.find(i) != active.end()
              && active.find(j->first) != active.end())) {
        for (int k = 0; k < j->second.size(); k++) {
          std::vector<DependenceVector> dv = j->second[k].permute(pi);
          g.connect(i, j->first, dv);
        }
      } else if (active.find(i) == active.end()
                 && active.find(j->first) == active.end()) {
        std::vector<DependenceVector> dv = j->second;
        g.connect(i, j->first, dv);
      } else {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++)
          for (int d = 0; d < pi.size(); d++)
            if (pi[d] != d) {
              dv[k].lbounds[d] = -posInfinity;
              dv[k].ubounds[d] = posInfinity;
            }
        g.connect(i, j->first, dv);
      }
    }
  
  return g;
}

// DependenceGraph DependenceGraph::matrix(const std::vector<std::vector<int> > &M) const {
//   DependenceGraph g;

//   for (int i = 0; i < vertex.size(); i++)
//     g.insert(vertex[i].first);

//   for (int i = 0; i < vertex.size(); i++)
//     for (EdgeList::const_iterator j = vertex[i].second.begin(); j != vertex[i].second.end(); j++)
//       for (int k = 0; k < j->second.size(); k++)
//         g.connect(i, j->first, j->second[k].matrix(M));

//   return g;
// }

DependenceGraph DependenceGraph::subspace(int dim) const {
  DependenceGraph g;
  
  for (int i = 0; i < vertex.size(); i++)
    g.insert(vertex[i].first);
  
  for (int i = 0; i < vertex.size(); i++)
    for (EdgeList::const_iterator j = vertex[i].second.begin();
         j != vertex[i].second.end(); j++)
      
      for (int k = 0; k < j->second.size(); k++) {
        if(j->second[k].type != DEP_CONTROL){
          if (j->second[k].isCarried(dim))
            g.connect(i, j->first, j->second[k]);
        }else
          g.connect(i, j->first, j->second[k]);
        
      }
  
  return g;
}

bool DependenceGraph::isPositive() const {
  for (int i = 0; i < vertex.size(); i++)
    for (EdgeList::const_iterator j = vertex[i].second.begin();
         j != vertex[i].second.end(); j++)
      for (int k = 0; k < j->second.size(); k++)
        if (!j->second[k].isPositive())
          return false;
  
  return true;
}

bool DependenceGraph::hasPositive(int dim) const {
  for (int i = 0; i < vertex.size(); i++)
    for (EdgeList::const_iterator j = vertex[i].second.begin();
         j != vertex[i].second.end(); j++)
      for (int k = 0; k < j->second.size(); k++)
        if (!j->second[k].hasPositive(dim))
          return false;
  
  return true;
}

bool DependenceGraph::hasNegative(int dim) const {
  for (int i = 0; i < vertex.size(); i++)
    for (EdgeList::const_iterator j = vertex[i].second.begin();
         j != vertex[i].second.end(); j++)
      for (int k = 0; k < j->second.size(); k++)
        if (!j->second[k].hasNegative(dim))
          return false;
  
  return true;
}
