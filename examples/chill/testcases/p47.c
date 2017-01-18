
#define X 3
#define N 10
#define M 10



void foo() { 
   int I, J;
   float A[20][20]; 
  
   // known iteration count
   for (J=1; J <= 10; J++) 
   {
      for (I=1; I <= 10; I++) 
      {
         A[I+1][J] = A[I][J] + X;
      }
   }
   
   return;
}

int main() {

   foo();
   
   return 0;
}

