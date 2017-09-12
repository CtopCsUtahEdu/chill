//#include <stdint.h>

#pragma  OMG 1 

#define uint64_t unsigned int
#define THISISDEFINED  buh 

#include "defines.h"   // local 
#include "included.c"  // local 

int main() { 
    int i, j = 0;
    for (i=0; i<100; i++) { 
        j = j+1;
	}
   return 0; 
}
