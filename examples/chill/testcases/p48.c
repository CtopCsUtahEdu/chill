
#define X 3
#define Y 7
#define N 10
#define M 10



void foo() { 
   int I, J;
   float A[100][100]; 
   float B[100][100]; 
   float C[100][100];

   // known iteration count
   for (J=1; J <= 10; J++) 
   {
      for (I=1; I <= 99; I++) 
      {
         A[I][J] = B[I][J] + X;
         C[I][J] = A[100-I][J] + Y;
      }
   }
   
   return;
}

int main() {

   foo();
   
   return 0;
}

