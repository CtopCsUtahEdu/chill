// this source is derived from CHILL AST originally from file 'include.c' as parsed by frontend compiler rose

#define uint64_t unsigned int

#define THISISDEFINED  buh 

#include "defines.h"   // local 

#include "included.c"  // local 

int main() {
  int t2;
  for (t2 = 0; t2 <= 99; t2 += 1) 
    ;
  return 0;
}
