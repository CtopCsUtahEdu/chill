

#define N 64

// SIMPLE EXAMPLE FROM Allen/Kennedy page 256    currently FAILS KNOW FAIL 
void foo(float* a, float *b, float c, float *d, float e) {
   int i;

   // 2 loops with identical iterations 
   for (i = 0; i < N; i++) 
   {
      a[i] = b[i] + c;
   }
      
   for (i = 0; i < N; i++) 
   {
      d[i] = a[i] + e;
   }

  return;
}

