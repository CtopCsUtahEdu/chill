
#define N 4

int xyz;

void zzfoo(float* A, float *B) {
   int I;
  
  // known iteration count
  for (I = 1; I < N;  I++) 
  {
     A[I+1] = A[I] + B[I];
     B[I-1] = 2.0f * B[I];
  }

  return;
}

int main() {
   float x[N], y[N];
   
   zzfoo(x,y);
   
   return 0;
}

