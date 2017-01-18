
#define N 16

void foo(float* A, float *B) {
   int I;
  
  // known iteration count
	for (I = 1; I < N; I++) 
  {
     A[I+2] = A[I] + B[I];
  }

	return;
}

int main() {
   float x[N], y[N];
   
   foo(x,y);
   
   return 0;
}

