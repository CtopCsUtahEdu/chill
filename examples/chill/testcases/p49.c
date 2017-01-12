
#define X 3
#define Y 7
#define N 10
#define M 10



void foo() { 
   int I;
   float A[100]; 
   float F[100]; 

   // known iteration count
   for (I=1; I <= N; I++) 
   {
      // loop carried dependence
      // S1 write to A in one iteration is read in the next iteration S2
      A[I+1] = F[I];    // S1
      F[I+1] = A[I];    // S2
   }

   
   return;
}

int main() {

   foo();
   
   return 0;
}

