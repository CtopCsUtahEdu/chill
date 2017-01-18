
#define X 3
#define Y 7
#define N 10
#define M 10



void foo() { 
   int I;
   float A[100]; 
   float F[100]; 

   A[0] = 0.0;
   // known iteration count
   for (I=1; I <= N; I++) 
   {
      // loop-independent dependence
      // S1 write to A in one iteration is read in same iteration S2
      A[I] = F[I] * 2.0;    // S1
      A[0] +=  A[I];    // S2
   }

   return;
}

int main() {

   foo();
   
   return 0;
}

