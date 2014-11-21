/*****************************************************************************
 Copyright (C) 2009 University of Utah 
 All Rights Reserved.

 Purpose:
   Class of set of bools where each element is indexed by a small integer.

 Notes:
   Set operands of binary operations can be of different sizes, missing
 elements are treated as false.

 History:
   03/30/2009 Created by Chun Chen.
*****************************************************************************/

#ifndef BOOLSET_H
#define BOOLSET_H

#include <vector>
#include <iostream>
#include <assert.h>

namespace omega {
  
template<typename T = unsigned int>
class BoolSet {
protected:
  unsigned int size_;
  std::vector<T> set_;
public:
  BoolSet(unsigned int size = 0);
  ~BoolSet() {}

  void set(unsigned int);
  void unset(unsigned int);
  bool get(unsigned int) const;
  unsigned int size() const {return size_;}
  unsigned int num_elem() const;
  bool imply(const BoolSet<T> &) const;
  bool empty() const;

  BoolSet<T> &operator|=(const BoolSet<T> &); 
  BoolSet<T> &operator&=(const BoolSet<T> &); 
  BoolSet<T> &operator-=(const BoolSet<T> &); 

  template<typename TT> friend BoolSet<TT> operator|(const BoolSet<TT> &, const BoolSet<TT> &);  // union
  template<typename TT> friend BoolSet<TT> operator&(const BoolSet<TT> &, const BoolSet<TT> &);  // intersection
  template<typename TT> friend BoolSet<TT> operator-(const BoolSet<TT> &, const BoolSet<TT> &);  // difference  
  template<typename TT> friend BoolSet<TT> operator~(const BoolSet<TT> &);                       // complement
  template<typename TT> friend bool operator==(const BoolSet<TT> &, const BoolSet<TT> &); 
  template<typename TT> friend bool operator!=(const BoolSet<TT> &, const BoolSet<TT> &); 
  template<typename TT> friend std::ostream& operator<<(std::ostream &, const BoolSet<TT> &);
};


template<typename T>
BoolSet<T>::BoolSet(unsigned int size) {
  assert(size >= 0);
  size_ = size;
  unsigned int n = size / (sizeof(T)*8);
  unsigned int r = size % (sizeof(T)*8);
  if (r != 0)
    n++;
  set_ = std::vector<T>(n, static_cast<T>(0));
}


template<typename T>
void BoolSet<T>::set(unsigned int i) {
  assert(i < size_ && i >= 0);
  unsigned int n = i / (sizeof(T)*8);
  unsigned int r = i % (sizeof(T)*8);

  T t = static_cast<T>(1) << r;
  set_[n] |= t;
}


template<typename T>
void BoolSet<T>::unset(unsigned int i) {
  assert(i < size_ && i >= 0);
  unsigned int n = i / (sizeof(T)*8);
  unsigned int r = i % (sizeof(T)*8);

  T t = static_cast<T>(1) << r;
  t = ~t;
  set_[n] &= t;
}


template<typename T>
bool BoolSet<T>::get(unsigned int i) const {
  assert(i < size_ && i >= 0);
  unsigned int n = i / (sizeof(T)*8);
  unsigned int r = i % (sizeof(T)*8);

  T t = static_cast<T>(1) << r;
  t = set_[n] & t;
  if (t)
    return true;
  else
    return false;
}


template<typename T>
unsigned int BoolSet<T>::num_elem() const {
  unsigned int n = size_;
  unsigned int c = 0;
  unsigned int p = 0;
  while (n != 0) {
    unsigned int m;
    if (n >= sizeof(T)*8) {
      m = sizeof(T)*8;
      n -= sizeof(T)*8;
    }
    else {
      m = n;
      n = 0;
    }

    T v = set_[p++];
    if (v != static_cast<T>(0)) {
      for (unsigned int i = 0; i < m; i++) {
        if (v & static_cast<T>(1))
          c++;
        v >>= 1;
      }
    }
  }

  return c;
}


template<typename T>
bool BoolSet<T>::imply(const BoolSet<T> &b) const {
  if (size_ >= b.size_) {
    for (unsigned int i = 0; i < b.set_.size(); i++)
      if ((set_[i] & b.set_[i]) != b.set_[i])
        return false;
  }
  else {
    for (unsigned int i = 0; i < set_.size(); i++)
      if ((set_[i] & b.set_[i]) != b.set_[i])
        return false;
    for (unsigned int i = set_.size(); i < b.set_.size(); i++)
      if (b.set_[i] != static_cast<T>(0))
        return false;
  }   

  return true;
}


template<typename T>
bool BoolSet<T>::empty() const {
  for (int i = 0; i < set_.size(); i++)
    if (set_[i] != static_cast<T>(0))
      return false;

  return true;
}


template<typename T>
BoolSet<T> operator|(const BoolSet<T> &a, const BoolSet<T> &b) {
  if (a.size_ >= b.size_) {
    BoolSet<T> c = a;
    for (unsigned int i = 0; i < b.set_.size(); i++)
      c.set_[i] |= b.set_[i];
    return c;
  }
  else {
    BoolSet<T> c = b;
    for (unsigned int i = 0; i < a.set_.size(); i++)
      c.set_[i] |= a.set_[i];
    return c;
  }
}


template<typename T>
BoolSet<T> operator&(const BoolSet<T> &a, const BoolSet<T> &b) {
  if (a.size_ >= b.size_) {
    BoolSet<T> c = a;
    for (unsigned int i = 0; i < b.set_.size(); i++)
      c.set_[i] &= b.set_[i];
    for (unsigned int i = b.set_.size(); i < a.set_.size(); i++)
      c.set_[i] = static_cast<T>(0);
    return c;
  }
  else {
    BoolSet<T> c = b;
    for (unsigned int i = 0; i < a.set_.size(); i++)
      c.set_[i] &= a.set_[i];
    for (unsigned int i = a.set_.size(); i < b.set_.size(); i++)
      c.set_[i] = static_cast<T>(0);
    return c;
  }
}
  

template<typename T>
BoolSet<T> operator-(const BoolSet<T> &a, const BoolSet<T> &b) {
  BoolSet<T> c(a.size_);
  
  int sz = a.set_.size();
  if (sz > b.set_.size())
    sz = b.set_.size();
  for (int i = 0; i < sz; i++)
    c.set_[i] = a.set_[i] ^ (a.set_[i] & b.set_[i]);
  for (int i = sz; i < a.set_.size(); i++)
    c.set_[i] = a.set_[i];

  return c;
}
  

template<typename T>
BoolSet<T> operator~(const BoolSet<T> &b) {
  unsigned int r = b.size_ % (sizeof(T)*8);
  BoolSet<T> a(b.size_);
  for (unsigned int i = 0; i < b.set_.size(); i++)
    a.set_[i] = ~b.set_[i];

  if (r != 0) {
    T t = static_cast<T>(1);
    for (unsigned int i = 1; i < r; i++)
      t = (t << 1) | static_cast<T>(1);
    a.set_[a.set_.size()-1] &= t;
  }
  return a;
}


template<typename T>
bool operator==(const BoolSet<T> &a, const BoolSet<T> &b) {
  return (a.size_ == b.size_) && (a.set_ == b.set_);
}


template<typename T>
bool operator!=(const BoolSet<T> &a, const BoolSet<T> &b) {
  return !(a == b);
}



template<typename T>
BoolSet<T> & BoolSet<T>::operator|=(const BoolSet<T> &b) {
  *this = *this | b;
  return *this;
}
  
  
template<typename T>
BoolSet<T> & BoolSet<T>::operator&=(const BoolSet<T> &b) {
  *this = *this & b;
  return *this;
}


template<typename T>
BoolSet<T> & BoolSet<T>::operator-=(const BoolSet<T> &b) {
  *this = *this - b;
  return *this;
}


template<typename T>
std::ostream& operator<<(std::ostream &os, const BoolSet<T> &b) {
  for (int i = b.size()-1; i >= 0; i--)
    if (b.get(i))
      os << '1';
    else
      os << '0';
  return os;
}

} // namespace

#endif
