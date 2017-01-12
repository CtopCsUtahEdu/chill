
#define N 16

void foo(float* x, float *y) {
   int i,j;
  x[0] = 123;
  x[1] = 354;

  // known iteration count
	for (i = 1; i < N; i++) 
  {
     x[i] = 2 * x[i];
     x[i+1] = x[i];
     x[i+2] = 3;

     y[i] = x[i-1];
  }

	return;
}

int main() {
   float x[N], y[N];
   
   foo(x,y);
   
   return 0;
}

