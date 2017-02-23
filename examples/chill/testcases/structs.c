//#include <stdint.h>

#define uint64_t unsigned int

struct { // unnamed  AKA __anonymous_0x239a400 todo typedef 
   struct internalNAMED{int i;}lowinside;  //  named inside unnamed 
} buh; // variable of type unnamed 

//struct {int box; }read, write;	// this dies



// this comment is before the function
void smooth_box_4_64( int sweep)
{
  buh.lowinside.i = sweep;
}

// this is after everything 
