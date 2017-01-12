#define N 15

void foo(int n, float* x) {
   
   int i;
   
   // known iteration count
   for (i = 1; i <= N; i++) {
      x[i] = 2.0f *(float)i; 
   }
   return;
}



int main() {
   float x[N];
   
   foo(N, x);
   return 0;
}

