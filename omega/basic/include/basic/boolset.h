/*****************************************************************************
 Copyright (C) 2009-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   BoolSet class, used as a set of integers from 0..n-1 where n is a very
 small integer.

 Notes:
   Set operands of binary operations can be of different sizes, missing
 elements are treated as false.

 History:
   03/30/09 Created by Chun Chen.
   03/26/11 iterator added, -chun
*****************************************************************************/

#ifndef _BOOLSET_H
#define _BOOLSET_H

#include <vector>
#include <iostream>
#include <assert.h>
#include <stdexcept>
#include <iterator>

namespace omega {

  /**
   * @brief BoolSet implemented using bitmasks
   * @tparam T Individual cell type holding the bits - defaults to unsigned int
   */
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
  void set_all();
  void unset_all();
  bool get(unsigned int) const;
  unsigned int size() const {return size_;}
  unsigned int num_elem() const;
  bool imply(const BoolSet<T> &) const;
  bool empty() const;
  void dump() const;

  BoolSet<T> &operator|=(const BoolSet<T> &); 
  BoolSet<T> &operator&=(const BoolSet<T> &); 
  BoolSet<T> &operator-=(const BoolSet<T> &); 

  //! Union
  template<typename TT> friend BoolSet<TT> operator|(const BoolSet<TT> &, const BoolSet<TT> &);
  //! intersection
  template<typename TT> friend BoolSet<TT> operator&(const BoolSet<TT> &, const BoolSet<TT> &);
  //! difference
  template<typename TT> friend BoolSet<TT> operator-(const BoolSet<TT> &, const BoolSet<TT> &);
  //! complement
  template<typename TT> friend BoolSet<TT> operator~(const BoolSet<TT> &);
  template<typename TT> friend bool operator==(const BoolSet<TT> &, const BoolSet<TT> &);
  template<typename TT> friend bool operator!=(const BoolSet<TT> &, const BoolSet<TT> &); 
  template<typename TT> friend std::ostream& operator<<(std::ostream &, const BoolSet<TT> &);
  template<typename TT> friend bool operator<(const BoolSet<TT> &, const BoolSet<TT> &);

// iterator related
public:
  class iterator;
  class const_iterator;
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
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
void BoolSet<T>::set_all() {
  unsigned int r = size_ % (sizeof(T)*8);
  if (r == 0) {
    for (unsigned int i = 0; i < set_.size(); i++)
      set_[i] = ~static_cast<T>(0);
  }
  else {
    for (unsigned int i = 0; i < set_.size()-1; i++)
      set_[i] = ~static_cast<T>(0);
    set_[set_.size()-1] = static_cast<T>(0);
    T t = static_cast<T>(1);
    for (unsigned int i = 0; i < r; i++) {
      set_[set_.size()-1] |= t;
      t = t<<1;
    }
  }
}


template<typename T>
void BoolSet<T>::unset_all() {
  for (unsigned int i = 0; i < set_.size(); i++)
    set_[i] = static_cast<T>(0);
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
void BoolSet<T>::dump() const {
  int j = 1;
  for (unsigned int i = 0; i < size(); i++) {
    if (get(i))
      std::cout << '1';
    else
      std::cout << '0';
    if (j%10 == 0 && i != size() - 1) {
      std::cout << ' ';
      j = 1;
    }
    else
      j++;
  }
  std::cout << std::endl;
  std::cout.flush();
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
  os << '{';
  for (typename BoolSet<T>::const_iterator i = b.begin(); i != b.end(); i++) {
    os << *i;
    if (i+1 != b.end())
      os << ',';
  }
  os << '}';
  
  return os;
}


template<typename T>
bool operator<(const BoolSet<T> &a, const BoolSet<T> &b) {
  unsigned int t1, t2;
  t1 = a.num_elem();
  t2 = b.num_elem();
  if (t1 < t2)
    return true;
  else if (t1 > t2)
    return false;
  else {
    t1 = a.size();
    t2 = b.size();
    if (t1 < t2)
      return true;
    else if (t1 > t2)
      return false;
    else 
      for (unsigned int i = 0; i < a.set_.size(); i++)
        if (a.set_[i] < b.set_[i])
          return true;
  }
  return false;
}


//
// iterator for BoolSet
//

template<typename T>
typename BoolSet<T>::iterator BoolSet<T>::begin() {
  typename BoolSet<T>::iterator it(this, 0);
  if (size_ == 0)
    return it;
  else if (set_[0] & static_cast<T>(1))
    return it;
  else
    return ++it;
}


template<typename T>
typename BoolSet<T>::iterator BoolSet<T>::end() {
  return typename BoolSet<T>::iterator(this, size_);
}


template<typename T>
typename BoolSet<T>::const_iterator BoolSet<T>::begin() const {
  typename BoolSet<T>::const_iterator it(this, 0);
  if (size_ == 0)
    return it;
  else if (set_[0] & static_cast<T>(1))
    return it;
  else
    return ++it;
}


template<typename T>
typename BoolSet<T>::const_iterator BoolSet<T>::end() const {
  return typename BoolSet<T>::const_iterator(this, size_);
}


template<typename T>
class BoolSet<T>::iterator: public std::iterator<std::forward_iterator_tag, T> {
protected:
  BoolSet<T> *s_;
  unsigned int pos_;

protected:
  iterator(BoolSet<T> *s, unsigned int pos) { s_ = s; pos_ = pos; }
  
public:
  ~iterator() {}
  
  typename BoolSet<T>::iterator &operator++();
  typename BoolSet<T>::iterator operator++(int);
  typename BoolSet<T>::iterator operator+(int) const;
  unsigned int operator*() const;
  bool operator==(const BoolSet<T>::iterator &) const;
  bool operator!=(const BoolSet<T>::iterator &) const;
  operator typename BoolSet<T>::const_iterator();

  friend class BoolSet<T>;
};


template<typename T>
typename BoolSet<T>::iterator &BoolSet<T>::iterator::operator++() {
  assert(pos_ < s_->size_);

  pos_++;
  unsigned int n = pos_ / (sizeof(T)*8);
  unsigned int r = pos_ % (sizeof(T)*8);
  while (pos_ < s_->size_) {
    if (s_->set_[n] == static_cast<T>(0)) {
      pos_ += sizeof(T)*8-r;
      n++;
      r = 0;
      if (pos_ >= s_->size_)
        break;
    }
    
    if (r == 0) {
      while (pos_ < s_->size_) {
        if (s_->set_[n] == static_cast<T>(0)) {
          pos_ += sizeof(T)*8;
          n++;
        }
        else
          break;
      }
      if (pos_ >= s_->size_)
        break;
    }

    for (unsigned int i = r; i < sizeof(T)*8; i++)
      if (s_->set_[n] & static_cast<T>(1) << i) {
        pos_ = pos_+i-r;
        return *this;
      }

    pos_ += sizeof(T)*8-r;
    n++;
    r = 0;
  }

  pos_ = s_->size_;
  return *this;
}


template<typename T>
typename BoolSet<T>::iterator BoolSet<T>::iterator::operator++(int) {
  typename BoolSet<T>::iterator it(*this);
  ++(*this);
  return it;
}


template<typename T>
typename BoolSet<T>::iterator BoolSet<T>::iterator::operator+(int n) const {
  assert(n >= 0);
  typename BoolSet<T>::iterator it(*this);
  while (n > 0) {
    ++it;
    --n;
  }
  return it;
}


template<typename T>
unsigned int BoolSet<T>::iterator::operator*() const {
  assert(pos_ < s_->size_);
  return pos_;
}


template<typename T>
bool BoolSet<T>::iterator::operator==(const BoolSet<T>::iterator &other) const {
  return s_ == other.s_ && pos_ == other.pos_;
}


template<typename T>
bool BoolSet<T>::iterator::operator!=(const BoolSet<T>::iterator &other) const {
  return !((*this) == other);
}


template<typename T>
BoolSet<T>::iterator::operator typename BoolSet<T>::const_iterator() {
  return BoolSet<T>::const_iterator(s_, pos_);
}


template<typename T>
class BoolSet<T>::const_iterator: public std::iterator<std::forward_iterator_tag, T> {
protected:
  const BoolSet<T> *s_;
  unsigned int pos_;

protected:
  const_iterator(const BoolSet<T> *s, unsigned int pos) { s_ = s; pos_ = pos; }
  
public:
  ~const_iterator() {}
  
  typename BoolSet<T>::const_iterator &operator++();
  typename BoolSet<T>::const_iterator operator++(int);
  typename BoolSet<T>::const_iterator operator+(int) const;
  unsigned int operator*() const;
  bool operator==(const BoolSet<T>::const_iterator &) const;
  bool operator!=(const BoolSet<T>::const_iterator &) const;

  friend class BoolSet<T>;
};


template<typename T>
typename BoolSet<T>::const_iterator &BoolSet<T>::const_iterator::operator++() {
  assert(pos_ < s_->size_);
  
  pos_++;
  unsigned int n = pos_ / (sizeof(T)*8);
  unsigned int r = pos_ % (sizeof(T)*8);
  while (pos_ < s_->size_) {
    if (s_->set_[n] == static_cast<T>(0)) {
      pos_ += sizeof(T)*8-r;
      n++;
      r = 0;
      if (pos_ >= s_->size_)
        break;
    }
    
    if (r == 0) {
      while (pos_ < s_->size_) {
        if (s_->set_[n] == static_cast<T>(0)) {
          pos_ += sizeof(T)*8;
          n++;
        }
        else
          break;
      }
      if (pos_ >= s_->size_)
        break;
    }

    for (unsigned int i = r; i < sizeof(T)*8; i++)
      if (s_->set_[n] & static_cast<T>(1) << i) {
        pos_ = pos_+i-r;
        return *this;
      }

    pos_ += sizeof(T)*8-r;
    n++;
    r = 0;
  }

  pos_ = s_->size_;
  return *this;
}


template<typename T>
typename BoolSet<T>::const_iterator BoolSet<T>::const_iterator::operator++(int) {
  typename BoolSet<T>::const_iterator it(*this);
  ++(*this);
  return it;
}


template<typename T>
typename BoolSet<T>::const_iterator BoolSet<T>::const_iterator::operator+(int n) const {
  assert(n >= 0);
  typename BoolSet<T>::const_iterator it(*this);
  while (n > 0) {
    ++it;
    --n;
  }
  return it;
}

  
template<typename T>
unsigned int BoolSet<T>::const_iterator::operator*() const {
  assert(pos_ < s_->size_);
  return pos_;
}


template<typename T>
bool BoolSet<T>::const_iterator::operator==(const BoolSet<T>::const_iterator &other) const {
  return s_ == other.s_ && pos_ == other.pos_;
}


template<typename T>
bool BoolSet<T>::const_iterator::operator!=(const BoolSet<T>::const_iterator &other) const {
  return !((*this) == other);
}

}

#endif
