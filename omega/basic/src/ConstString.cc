#include <basic/ConstString.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <string.h>

/* static const int CS_HashTable_Size = 1000; */
/* static ConstStringRep *hashTable[CS_HashTable_Size] = {0}; */

namespace omega {

const int CS_HashTable_Size = 1000;
class CS_HashTable {
public:
  ConstStringRep *p[CS_HashTable_Size];
  CS_HashTable();
  ~CS_HashTable();
};

namespace {
  CS_HashTable hashTable;
}

CS_HashTable::CS_HashTable() {
    for (int i = 0; i < CS_HashTable_Size; i++)
      p[i] = NULL;
  }

CS_HashTable::~CS_HashTable() {
  for (int i = 0; i < CS_HashTable_Size; i++) {
    ConstStringRep *t = p[i];
    while (t != NULL) {
      ConstStringRep *tt = t->nextInBucket;
      delete []t->name;
      delete t;
      t = tt;
    }
  }    
}

Const_String::Const_String() {
  rep = 0;
}

void Const_String::buildRep(const char* t) {
  int hash = 0;
  const char *s = t;
  while (*s != '\0') 
    hash = hash*33 + *s++;
  int hashBucket = hash % CS_HashTable_Size;
  if (hashBucket < 0) hashBucket += CS_HashTable_Size;
  assert(0 <= hashBucket && hashBucket < CS_HashTable_Size);
  ConstStringRep **q = &(hashTable.p[hashBucket]);
  ConstStringRep *p = *q;
  while (p != 0) {
    if (strcmp(p->name,t) == 0) break;
    q = &p->nextInBucket;  
    p = *q;
  }
  if (p!= 0) rep = p;
  else {
    rep = new ConstStringRep(t);
    *q = rep;
  }
}

Const_String::Const_String(const char * t) {
  buildRep(t);
}

Const_String::Const_String(const std::string &s) {
  buildRep(s.c_str());
}

Const_String::operator const char*() const {
  if (!rep) return 0;
  return rep->name;
}

Const_String::operator std::string() const {
  if (!rep) return std::string("");
  return std::string(rep->name);
}

int Const_String::operator++(int) {
  return rep->count++;
}

int Const_String::operator++() {
  return ++rep->count;
}

int Const_String:: operator--(int) {
  return rep->count--;
}

int Const_String:: operator--() {
  return --rep->count;
}

int operator ==(const Const_String &x, const Const_String &y) {
  return x.rep == y.rep;
}

int operator !=(const Const_String &x, const Const_String &y) {
  return x.rep != y.rep;
}

int operator <(const Const_String &x, const Const_String &y) {
  return (strcmp(x.rep->name,y.rep->name) < 0);
}

int operator >(const Const_String &x, const Const_String &y) {
  return (strcmp(x.rep->name,y.rep->name) > 0);
}

Const_String:: operator int() const {
  return rep != 0;
}

int Const_String::null() const {
  return rep == 0;
}

ConstStringRep:: ConstStringRep(const char *t) {
  count = 0;
  nextInBucket = 0;
  char *s = new char[1+strlen(t)];
  strcpy(s,t);
  name = s;
}

} // namespace
