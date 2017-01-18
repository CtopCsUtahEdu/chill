
#define N 16
#define M 20

void foo(float* A, float *B) {
   int I, J;
  
  // known iteration count
	for (I = 1; I < N; I++) 
  {
     for (J=0; J<M; J++) 
     {
        A[I+2] = A[I] + B[J];
     }
  }

	return;
}

int main() {
   float x[N], y[M];
   
   foo(x,y);
   
   return 0;
}

