
#define N 16
#define M 20
#define L 25


void foo() { 
  int I, J, K;
  float A[100][100][100]; 

  // known iteration count
	for (I = 1; I<= N; I++) 
  {
     for (J=1; J<=M; J++) 
     {
       for (K=1; K<=L; K++) 
         {
           A[I+1][ J ][ K-1] = A[I][J][K] + 10;
         }
     }
  }

	return;
}

int main() {
   
   foo();
   
   return 0;
}

