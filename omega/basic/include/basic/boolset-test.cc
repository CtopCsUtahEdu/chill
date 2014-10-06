#include "boolset.h"
#include <iostream>

using namespace omega;

void foo(const BoolSet<> &B) {
  for (BoolSet<>::const_iterator i = B.begin(); i != B.end(); i++)
    std::cout << *i << ' ';
  std::cout << std::endl;
}

int main() {
  BoolSet<> A(13);
  
  A.set(2);
  std::cout << A << std::endl;
  
  A.set_all();
  std::cout << A << std::endl;

  A.unset_all();
  std::cout << A << std::endl;

  A.set(2);
  A.set(4);

  BoolSet<> B(13);
  B.set(2);

  std::cout << "A: " << A << std::endl;
  std::cout << "B: " << B << std::endl;
  
  std::cout << A.imply(B) << std::endl;
  std::cout << B.imply(A) << std::endl;

  B.set(10);
  std::cout << (A|B) << std::endl;
  std::cout << (A&B) << std::endl;

  BoolSet<> C(3);
  C.set(0);
  std::cout << (A|C) << std::endl;
  std::cout << ~(A|C) << std::endl;

  B = BoolSet<>(23);
  std::cout << "test iterator\n";
  B.set(12);
  B.set(11);
  B.set(0);
  std::cout << B << std::endl;
  for (BoolSet<>::const_iterator i = B.begin(); i != B.end(); i++) {
    std::cout << *i << ' ';
    if (*i == 11)
      B.unset(*i);
  }
  std::cout << std::endl;
  std::cout << B << std::endl;
  std::cout << std::endl;
  foo(B);

  std::cout << ~BoolSet<>(5) << std::endl;

  std::cout << "------\n";
  B.dump();
  std::cout << std::endl << *(B.begin()+1) << std::endl;

  for (BoolSet<>::iterator i = B.begin(); i != B.end(); i++)
    for (BoolSet<>::iterator j = i; j != B.end(); j++)
      if (j == i)
        std::cout << "ehh-";
  
}
